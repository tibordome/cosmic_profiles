#cython: language_level=3
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

cimport openmp
import numpy as np
cimport cython
from cosmic_profiles.cython_helpers.helper_class cimport CythonHelpers
from scipy.interpolate import interp1d
from cosmic_profiles.common.python_routines import respectPBCNoRef, calcMode, calcCoM
from cython.parallel import prange
from cosmic_profiles.common.caching import np_cache_factory

@cython.embedsignature(True)
@cython.binding(True)
def calcMassesCenters(float[:,:] xyz, float[:] masses, idx_cat, int MIN_NUMBER_PTCS, float L_BOX, str CENTER):
    """ Calculate total mass and centers of objects
    
    :param xyz: positions of all simulation particles
    :type xyz: (N2,3) floats, N2 >> N1
    :param masses: masses of all simulation particles
    :type masses: (N2,) floats
    :param idx_cat: list of indices defining the objects
    :type idx_cat: list of length N1, each consisting of a list of int indices
    :param MIN_NUMBER_PTCS: minimum number of particles for object to qualify for morphology calculation
    :type MIN_NUMBER_PTCS: int
    :param L_BOX: box size
    :type L_BOX: float
    :param CENTER: density profiles will be calculated with respect to CENTER = 'mode' (point of highest density)
        or 'com' (center of mass) of each halo
    :type CENTER: str
    :return centers, m: centers and masses
    :rtype: (N,3) and (N,) floats"""
    def inner(float[:,:] xyz, float[:] masses, idx_cat, int MIN_NUMBER_PTCS, float L_BOX, str CENTER):
        # Transform idx_cat to int[:,:]
        cdef int nb_objs = len(idx_cat)
        cdef int p
        cdef int[:] obj_size = np.zeros((nb_objs,), dtype = np.int32)
        cdef int[:] obj_pass = np.zeros((nb_objs,), dtype = np.int32)
        for p in range(nb_objs):
            if len(idx_cat[p]) >= MIN_NUMBER_PTCS: # Only add objects that have sufficient resolution
                obj_size[p] = len(idx_cat[p]) 
                obj_pass[p] = 1
        cdef int nb_pass = np.sum(obj_pass.base)
        cdef int[:] idxs_compr = np.zeros((nb_objs,), dtype = np.int32)
        idxs_compr.base[obj_pass.base.nonzero()[0]] = np.arange(np.sum(obj_pass.base))
        cdef int[:,:] idx_cat_arr = np.zeros((nb_pass,np.max([len(idx_cat[p]) for p in range(nb_objs)])), dtype = np.int32)
        for p in range(nb_objs):
            if obj_pass[p] == 1:
                idx_cat_arr.base[idxs_compr[p],:obj_size[p]] = np.array(idx_cat[p])
    
        # Calculate centers and total masses of objects
        del idx_cat
        cdef float[:] m = np.zeros((nb_pass,), dtype = np.float32)
        cdef int n
        cdef float[:,:] centers = np.zeros((nb_pass,3), dtype = np.float32)
        for p in range(nb_objs): # Calculate centers of objects
            if obj_pass[p] == 1:
                xyz_ = respectPBCNoRef(xyz.base[idx_cat_arr.base[idxs_compr[p],:obj_size[p]]], L_BOX)
                if CENTER == 'mode':
                    centers.base[idxs_compr[p]] = calcMode(xyz_, masses.base[idx_cat_arr.base[idxs_compr[p],:obj_size[p]]], 1000)
                else:
                    centers.base[idxs_compr[p]] = calcCoM(xyz_, masses.base[idx_cat_arr.base[idxs_compr[p],:obj_size[p]]])
        del xyz
        for p in prange(nb_objs, schedule = 'dynamic', nogil = True): # Calculate total mass of objects
            if obj_pass[p] == 1:
                for n in range(obj_size[p]):
                    m[idxs_compr[p]] = m[idxs_compr[p]] + masses[idx_cat_arr[idxs_compr[p],n]]
        del idx_cat_arr; del obj_pass; del idxs_compr; del obj_size; del masses
        return centers.base, m.base # Only rank = 0 content matters
    if(not hasattr(calcMassesCenters, "inner")):
        calcMassesCenters.inner = np_cache_factory(2,1)(inner)
    calcMassesCenters.inner(xyz.base, masses.base, idx_cat, MIN_NUMBER_PTCS, L_BOX, CENTER)
    return calcMassesCenters.inner(xyz.base, masses.base, idx_cat, MIN_NUMBER_PTCS, L_BOX, CENTER)

@cython.embedsignature(True)
@cython.binding(True)
def calcDensProfsSphDirectBinning(float[:,:] xyz, int[:] obj_keep, float[:] masses, float[:] r200s, float[:] ROverR200, idx_cat, int MIN_NUMBER_PTCS, float L_BOX, str CENTER):
    """ Calculates spherical shell-based density profiles for objects defined by indices found in `idx_cat`    
    
    Note: To calculate enclosed mass profiles, envoke ``CythonHelpers.calcMenclsBruteForce()`` instead of ``CythonHelpers.calcDensProfBruteForce()``
    
    :param xyz: positions of all simulation particles
    :type xyz: (N2,3) floats, N2 >> N1
    :param obj_keep: which objects among the N1 different ones to consider. 1: keep, 0: ignore
        This can be used to select objects within a certain mass range, for instance. Having
        a 1 where `idx_cat` has an empty list entry is not permitted.
    :type obj_keep: (N1,) ints
    :param masses: masses of all simulation particles
    :type masses: (N2,) floats
    :param r200s: R200 values of the objects
    :type r200s: (N1,) floats
    :param ROverR200: radii at which the density profiles should be calculated,
        normalized by R200
    :type ROverR200: (r_res,) float array
    :param idx_cat: list of indices defining the objects
    :type idx_cat: list of length N1, each consisting of a list of int indices
    :param MIN_NUMBER_PTCS: minimum number of particles for object to qualify for density profile calculation
    :type MIN_NUMBER_PTCS: int
    :param L_BOX: box size
    :type L_BOX: float
    :param CENTER: density profiles will be calculated with respect to CENTER = 'mode' (point of highest density)
        or 'com' (center of mass) of each halo
    :type CENTER: str
    :return: density profiles defined at ROverR200
    :rtype: (nb_keep, r_res) float array"""
    def inner(float[:,:] xyz, int[:] obj_keep, float[:] masses, float[:] r200s, float[:] ROverR200, idx_cat, int MIN_NUMBER_PTCS, float L_BOX, str CENTER):
        cdef int nb_objs = len(idx_cat)
        # Determine endpoints of radial bins
        cdef float[:] bin_edges = np.hstack(([np.float32(1e-8), (ROverR200.base[:-1] + ROverR200.base[1:])/2., ROverR200.base[-1]])) # Length = ROverR200.shape[0]+1
        cdef int r_res = ROverR200.shape[0]
        cdef int p
        cdef int n
        cdef int[:] obj_size = np.zeros((nb_objs,), dtype = np.int32)
        cdef int[:] obj_pass = np.zeros((nb_objs,), dtype = np.int32)
        for p in range(nb_objs):
            if len(idx_cat[p]) >= MIN_NUMBER_PTCS: # Only add objects that have sufficient resolution
                obj_size[p] = len(idx_cat[p])
                obj_pass[p] = 1
            else:
                if obj_keep[p] == 1:
                    raise ValueError("Having a 1 in obj_keep for an object for which idx_cat has insufficient resolution is not allowed.")
        cdef int nb_pass = np.sum(obj_pass.base)
        cdef int nb_keep = np.sum(obj_keep.base)
        if nb_objs == 0:
            return np.zeros((0,r_res), dtype = np.float32)
        # Transform idx_cat to int[:,:]
        cdef int[:,:] idx_cat_arr = np.zeros((nb_pass,np.max([len(idx_cat[p]) for p in range(nb_objs)])), dtype = np.int32)
        cdef int[:] idxs_compr = np.zeros((nb_objs,), dtype = np.int32)
        idxs_compr.base[obj_pass.base.nonzero()[0]] = np.arange(np.sum(obj_pass.base))
        for p in range(nb_objs):
            if obj_pass[p] == 1:
                idx_cat_arr.base[idxs_compr[p],:obj_size[p]] = np.array(idx_cat[p])
        # Define memoryviews
        cdef int[:] idxs_keep = np.zeros((nb_objs,), dtype = np.int32)
        idxs_keep.base[obj_keep.base.nonzero()[0]] = np.arange(np.sum(obj_keep.base))
        cdef float[:,:] dens_profs = np.zeros((nb_keep, r_res), dtype = np.float32)
        cdef float[:,:] bin_edges_tiled = np.reshape(np.tile(bin_edges.base, reps = openmp.omp_get_max_threads()), (openmp.omp_get_max_threads(), r_res+1))
        cdef int[:,:] shell = np.zeros((openmp.omp_get_max_threads(), idx_cat_arr.shape[1]), dtype = np.int32)
        cdef float[:,:,:] xyz_obj = np.zeros((openmp.omp_get_max_threads(), idx_cat_arr.shape[1], 3), dtype = np.float32)
        cdef float[:,:] m_obj = np.zeros((openmp.omp_get_max_threads(), idx_cat_arr.shape[1]), dtype = np.float32)
        cdef float[:,:] centers = np.zeros((nb_pass,3), dtype = np.float32)
        for p in range(nb_objs): # Calculate centers of objects
            if obj_pass[p] == 1:
                xyz_ = respectPBCNoRef(xyz.base[idx_cat_arr.base[idxs_compr[p],:obj_size[p]]], L_BOX)
                if CENTER == 'mode':
                    centers.base[idxs_compr[p]] = calcMode(xyz_, masses.base[idx_cat_arr.base[idxs_compr[p],:obj_size[p]]], 1000)
                else:
                    centers.base[idxs_compr[p]] = calcCoM(xyz_, masses.base[idx_cat_arr.base[idxs_compr[p],:obj_size[p]]])
        for p in prange(nb_objs, schedule = 'dynamic', nogil = True):
            if obj_keep[p] == 1:
                for n in range(obj_size[p]):
                    xyz_obj[openmp.omp_get_thread_num(),n] = xyz[idx_cat_arr[idxs_compr[p],n]]
                    m_obj[openmp.omp_get_thread_num(),n] = masses[idx_cat_arr[idxs_compr[p],n]]
                xyz_obj[openmp.omp_get_thread_num(),:obj_size[p]] = CythonHelpers.respectPBCNoRef(xyz_obj[openmp.omp_get_thread_num(),:obj_size[p]], L_BOX)
                dens_profs[idxs_keep[p]] = CythonHelpers.calcDensProfBruteForceSph(xyz_obj[openmp.omp_get_thread_num(),:obj_size[p]], m_obj[openmp.omp_get_thread_num(),:obj_size[p]], centers[idxs_compr[p]], r200s[p], bin_edges_tiled[openmp.omp_get_thread_num()], dens_profs[idxs_keep[p]], shell[openmp.omp_get_thread_num()])
        del bin_edges; del obj_size; del obj_pass; del idx_cat_arr; del idx_cat; del idxs_compr; del idxs_keep; del shell; del bin_edges_tiled; del xyz_obj; del m_obj; del centers
        return dens_profs.base
    if(not hasattr(calcDensProfsSphDirectBinning, "inner")):
        calcDensProfsSphDirectBinning.inner = np_cache_factory(5,1)(inner)
    calcDensProfsSphDirectBinning.inner(xyz.base, obj_keep.base, masses.base, r200s.base, ROverR200.base, idx_cat, MIN_NUMBER_PTCS, L_BOX, CENTER)
    return calcDensProfsSphDirectBinning.inner(xyz.base, obj_keep.base, masses.base, r200s.base, ROverR200.base, idx_cat, MIN_NUMBER_PTCS, L_BOX, CENTER)

@cython.embedsignature(True)
@cython.binding(True)
def calcDensProfsEllDirectBinning(float[:,:] xyz, int[:] obj_keep, float[:] masses, float[:] r200s, float[:] ROverR200, float[:,:] a, float[:,:] b, float[:,:] c, float[:,:,:] major, float[:,:,:] inter, float[:,:,:] minor, idx_cat, int MIN_NUMBER_PTCS, float L_BOX, str CENTER):
    """ Calculates ellipsoidal shell-based density profiles for objects defined by indices found in `idx_cat`    
    
    Note: To calculate enclosed mass profiles, envoke ``CythonHelpers.calcMenclsBruteForceEll()`` instead of ``CythonHelpers.calcDensProfBruteForceEll()``
    
    :param xyz: positions of all simulation particles
    :type xyz: (N2,3) floats, N2 >> N1
    :param obj_keep: which objects among the N1 different ones to consider. 1: keep, 0: ignore
        This can be used to select objects within a certain mass range, for instance. Having
        a 1 where `idx_cat` has an empty list entry is not permitted.
    :type obj_keep: (N1,) ints
    :param masses: masses of all simulation particles
    :type masses: (N2,) floats
    :param r200s: R200 values of the objects
    :type r200s: (N1,) floats
    :param ROverR200: radii at which the density profiles should be calculated,
        normalized by R200
    :type ROverR200: (r_res,) float array
    :param idx_cat: list of indices defining the objects
    :type idx_cat: list of length N1, each consisting of a list of int indices
    :param MIN_NUMBER_PTCS: minimum number of particles for object to qualify for density profile calculation
    :type MIN_NUMBER_PTCS: int
    :param L_BOX: box size
    :type L_BOX: float
    :param CENTER: density profiles will be calculated with respect to CENTER = 'mode' (point of highest density)
        or 'com' (center of mass) of each halo
    :type CENTER: str
    :param a: major axis eigenvalues
    :type a: (N1,D_BINS+1,) floats
    :param b: intermediate axis eigenvalues
    :type b: (N1,D_BINS+1,) floats
    :param c: minor axis eigenvalues
    :type c: (N1,D_BINS+1,) floats
    :param major: major axis eigenvectors
    :type major: (N1,D_BINS+1,3) floats
    :param inter: inter axis eigenvectors
    :type inter: (N1,D_BINS+1,3) floats
    :param minor: minor axis eigenvectors
    :type minor: (N1,D_BINS+1,3) floats
    :return: density profiles defined at ROverR200
    :rtype: (nb_keep, r_res) float array"""
    def inner(float[:,:] xyz, int[:] obj_keep, float[:] masses, float[:] r200s, float[:] ROverR200, float[:,:] a, float[:,:] b, float[:,:] c, float[:,:,:] major, float[:,:,:] inter, float[:,:,:] minor, idx_cat, int MIN_NUMBER_PTCS, float L_BOX, str CENTER):
        assert a.shape[0] == b.shape[0]
        assert b.shape[0] == c.shape[0]
        cdef int nb_objs = len(idx_cat)
        # Determine endpoints of radial bins
        cdef float[:] r_midpoints = (ROverR200.base[:-1] + ROverR200.base[1:])/2
        cdef float[:] bin_edges = np.hstack(([np.float32(r_midpoints[0]*r_midpoints[0]/r_midpoints[1]), r_midpoints.base, ROverR200.base[-1]])) # Length = ROverR200.shape[0]+1
        cdef int r_res = ROverR200.shape[0]
        cdef int p
        cdef int n
        cdef int[:] obj_size = np.zeros((nb_objs,), dtype = np.int32)
        cdef int[:] obj_pass = np.zeros((nb_objs,), dtype = np.int32)
        for p in range(nb_objs):
            if len(idx_cat[p]) >= MIN_NUMBER_PTCS: # Only add objects that have sufficient resolution
                obj_size[p] = len(idx_cat[p])
                obj_pass[p] = 1
            else:
                if obj_keep[p] == 1:
                    raise ValueError("Having a 1 in obj_keep for an object for which idx_cat has insufficient resolution is not allowed.")
        cdef int nb_pass = np.sum(obj_pass.base)
        cdef int nb_keep = np.sum(obj_keep.base)
        if nb_objs == 0:
            return np.zeros((0,r_res), dtype = np.float32)
        # Transform idx_cat to int[:,:]
        cdef int[:,:] idx_cat_arr = np.zeros((nb_pass,np.max([len(idx_cat[p]) for p in range(nb_objs)])), dtype = np.int32)
        cdef int[:] idxs_compr = np.zeros((nb_objs,), dtype = np.int32)
        idxs_compr.base[obj_pass.base.nonzero()[0]] = np.arange(np.sum(obj_pass.base))
        for p in range(nb_objs):
            if obj_pass[p] == 1:
                idx_cat_arr.base[idxs_compr[p],:obj_size[p]] = np.array(idx_cat[p])
        
        # Interpolate shape information to radii of interest
        cdef float[:,:] a_interpol = np.zeros((nb_keep,r_res+1), dtype = np.float32)
        cdef float[:,:] b_interpol = np.zeros((nb_keep,r_res+1), dtype = np.float32)
        cdef float[:,:] c_interpol = np.zeros((nb_keep,r_res+1), dtype = np.float32)
        cdef float[:,:,:] major_interpol = np.zeros((nb_keep,r_res,3), dtype = np.float32)
        cdef float[:,:,:] inter_interpol = np.zeros((nb_keep,r_res,3), dtype = np.float32)
        cdef float[:,:,:] minor_interpol = np.zeros((nb_keep,r_res,3), dtype = np.float32)
        cdef int[:] idxs_keep = np.zeros((nb_objs,), dtype = np.int32)
        idxs_keep.base[obj_keep.base.nonzero()[0]] = np.arange(np.sum(obj_keep.base))
        for p in range(nb_objs):
            if obj_pass[p] == 1:
                idx_cat_arr.base[idxs_compr[p],:obj_size[p]] = np.array(idx_cat[p])
                r_vec = np.float32(a.base[idxs_compr[p]])
                a_inter = interp1d(r_vec, a.base[idxs_compr[p]], bounds_error=False, fill_value='extrapolate')
                b_inter = interp1d(r_vec, b.base[idxs_compr[p]], bounds_error=False, fill_value='extrapolate')
                c_inter = interp1d(r_vec, c.base[idxs_compr[p]], bounds_error=False, fill_value='extrapolate')
                majorx_inter = interp1d(r_vec, major.base[idxs_compr[p],:,0], bounds_error=False, fill_value='extrapolate')
                majory_inter = interp1d(r_vec, major.base[idxs_compr[p],:,1], bounds_error=False, fill_value='extrapolate')
                majorz_inter = interp1d(r_vec, major.base[idxs_compr[p],:,2], bounds_error=False, fill_value='extrapolate')
                interx_inter = interp1d(r_vec, inter.base[idxs_compr[p],:,0], bounds_error=False, fill_value='extrapolate')
                intery_inter = interp1d(r_vec, inter.base[idxs_compr[p],:,1], bounds_error=False, fill_value='extrapolate')
                interz_inter = interp1d(r_vec, inter.base[idxs_compr[p],:,2], bounds_error=False, fill_value='extrapolate')
                minorx_inter = interp1d(r_vec, minor.base[idxs_compr[p],:,0], bounds_error=False, fill_value='extrapolate')
                minory_inter = interp1d(r_vec, minor.base[idxs_compr[p],:,1], bounds_error=False, fill_value='extrapolate')
                minorz_inter = interp1d(r_vec, minor.base[idxs_compr[p],:,2], bounds_error=False, fill_value='extrapolate')
                a_interpol.base[idxs_compr[p]] = a_inter(bin_edges.base*r200s[p])
                b_interpol.base[idxs_compr[p]] = b_inter(bin_edges.base*r200s[p])
                c_interpol.base[idxs_compr[p]] = c_inter(bin_edges.base*r200s[p])
                major_interpol.base[idxs_compr[p],:,0] = majorx_inter(ROverR200.base*r200s[p])
                major_interpol.base[idxs_compr[p],:,1] = majory_inter(ROverR200.base*r200s[p])
                major_interpol.base[idxs_compr[p],:,2] = majorz_inter(ROverR200.base*r200s[p])
                inter_interpol.base[idxs_compr[p],:,0] = interx_inter(ROverR200.base*r200s[p])
                inter_interpol.base[idxs_compr[p],:,1] = intery_inter(ROverR200.base*r200s[p])
                inter_interpol.base[idxs_compr[p],:,2] = interz_inter(ROverR200.base*r200s[p])
                minor_interpol.base[idxs_compr[p],:,0] = minorx_inter(ROverR200.base*r200s[p])
                minor_interpol.base[idxs_compr[p],:,1] = minory_inter(ROverR200.base*r200s[p])
                minor_interpol.base[idxs_compr[p],:,2] = minorz_inter(ROverR200.base*r200s[p])
        del idx_cat        
        # Calculate centers of objects
        cdef float[:,:] centers = np.zeros((nb_pass,3), dtype = np.float32)
        for p in range(nb_objs):
            if obj_pass[p] == 1:
                xyz_ = respectPBCNoRef(xyz.base[idx_cat_arr.base[idxs_compr[p],:obj_size[p]]], L_BOX)
                if CENTER == 'mode':
                    centers.base[idxs_compr[p]] = calcMode(xyz_, masses.base[idx_cat_arr.base[idxs_compr[p],:obj_size[p]]], 1000)
                else:
                    centers.base[idxs_compr[p]] = calcCoM(xyz_, masses.base[idx_cat_arr.base[idxs_compr[p],:obj_size[p]]])
        # Prepare density profile estimation
        cdef float[:,:] dens_profs = np.zeros((nb_keep, r_res), dtype = np.float32)
        cdef int[:,:] shell = np.zeros((openmp.omp_get_max_threads(), idx_cat_arr.shape[1]), dtype = np.int32)
        cdef float[:,:,:] xyz_obj = np.zeros((openmp.omp_get_max_threads(), idx_cat_arr.shape[1], 3), dtype = np.float32)
        cdef float[:,:,:] xyz_obj_princ = np.zeros((openmp.omp_get_max_threads(), idx_cat_arr.shape[1], 3), dtype = np.float32)
        cdef float[:,:] m_obj = np.zeros((openmp.omp_get_max_threads(), idx_cat_arr.shape[1]), dtype = np.float32)
        for p in prange(nb_objs, schedule = 'dynamic', nogil = True):
            if obj_keep[p] == 1:
                for n in range(obj_size[p]):
                    xyz_obj[openmp.omp_get_thread_num(),n] = xyz[idx_cat_arr[idxs_compr[p],n]]
                    m_obj[openmp.omp_get_thread_num(),n] = masses[idx_cat_arr[idxs_compr[p],n]]
                xyz_obj[openmp.omp_get_thread_num(),:obj_size[p]] = CythonHelpers.respectPBCNoRef(xyz_obj[openmp.omp_get_thread_num(),:obj_size[p]], L_BOX)
                dens_profs[idxs_keep[p]] = CythonHelpers.calcDensProfBruteForceEll(xyz_obj[openmp.omp_get_thread_num(),:obj_size[p]], xyz_obj_princ[openmp.omp_get_thread_num(),:obj_size[p]], m_obj[openmp.omp_get_thread_num(),:obj_size[p]], centers[idxs_compr[p]], r200s[p], a_interpol[idxs_compr[p]], b_interpol[idxs_compr[p]], c_interpol[idxs_compr[p]], major_interpol[idxs_compr[p]], inter_interpol[idxs_compr[p]], minor_interpol[idxs_compr[p]], dens_profs[idxs_keep[p]], shell[openmp.omp_get_thread_num()])
        del bin_edges; del obj_size; del obj_pass; del idx_cat_arr; del idxs_compr; del idxs_keep; del shell; del xyz_obj; del m_obj; del centers
        del a_interpol; del b_interpol; del c_interpol; del major_interpol; del inter_interpol; del minor_interpol; del xyz_obj_princ; del a; del b; del c; del major; del inter; del minor
        return dens_profs.base
    if(not hasattr(calcDensProfsEllDirectBinning, "inner")):
        calcDensProfsEllDirectBinning.inner = np_cache_factory(11,1)(inner)
    calcDensProfsEllDirectBinning.inner(xyz.base, obj_keep.base, masses.base, r200s.base, ROverR200.base, a.base, b.base, c.base, major.base, inter.base, minor.base, idx_cat, MIN_NUMBER_PTCS, L_BOX, CENTER)
    return calcDensProfsEllDirectBinning.inner(xyz.base, obj_keep.base, masses.base, r200s.base, ROverR200.base, a.base, b.base, c.base, major.base, inter.base, minor.base, idx_cat, MIN_NUMBER_PTCS, L_BOX, CENTER)

@cython.embedsignature(True)
@cython.binding(True)
def calcDensProfsKernelBased(float[:,:] xyz, int[:] obj_keep, float[:] masses, float[:] r200s, float[:] ROverR200, idx_cat, int MIN_NUMBER_PTCS, float L_BOX, str CENTER):
    """ Calculates kernel-based density profiles for objects defined by indices found in `idx_cat`
    
    Note: For background on this kernel-based method consult Reed et al. 2003, https://arxiv.org/abs/astro-ph/0312544.
    
    :param xyz: positions of all simulation particles
    :type xyz: (N2,3) floats, N2 >> N1
    :param obj_keep: which objects among the N1 different ones to consider. 1: keep, 0: ignore
        This can be used to select objects within a certain mass range, for instance. Having
        a 1 where `idx_cat` has an empty list entry is not permitted.
    :type obj_keep: (N1,) ints
    :param masses: masses of all simulation particles
    :type masses: (N2,) floats
    :param r200s: R200 values of the objects
    :type r200s: (N1,) floats
    :param ROverR200: radii at which the density profiles should be calculated,
        normalized by R200
    :type ROverR200: (r_res,) float array
    :param idx_cat: list of indices defining the objects
    :type idx_cat: list of length N1, each consisting of a list of int indices
    :param MIN_NUMBER_PTCS: minimum number of particles for object to qualify for density profile calculation
    :type MIN_NUMBER_PTCS: int
    :param L_BOX: box size
    :type L_BOX: float
    :param CENTER: density profiles will be calculated with respect to CENTER = 'mode' (point of highest density)
        or 'com' (center of mass) of each halo
    :type CENTER: str
    :return: density profiles defined at ROverR200
    :rtype: (nb_keep, r_res) float array"""
    
    def inner(float[:,:] xyz, int[:] obj_keep, float[:] masses, float[:] r200s, float[:] ROverR200, idx_cat, int MIN_NUMBER_PTCS, float L_BOX, str CENTER):
        cdef int nb_objs = len(idx_cat)
        cdef int r_res = ROverR200.shape[0]
        cdef int p
        cdef int r_idx
        cdef int n
        cdef int[:] obj_size = np.zeros((nb_objs,), dtype = np.int32)
        cdef int[:] obj_pass = np.zeros((nb_objs,), dtype = np.int32)
        for p in range(nb_objs):
            if len(idx_cat[p]) >= MIN_NUMBER_PTCS: # Only add objects that have sufficient resolution
                obj_size[p] = len(idx_cat[p])
                obj_pass[p] = 1
            else:
                if obj_keep[p] == 1:
                    raise ValueError("Having a 1 in obj_keep for an object for which idx_cat has insufficient resolution is not allowed.")
        cdef int nb_pass = np.sum(obj_pass.base)
        cdef int nb_keep = np.sum(obj_keep.base)
        if nb_objs == 0:
            return np.zeros((0,r_res), dtype = np.float32)
        # Transform idx_cat to int[:,:]
        cdef int[:,:] idx_cat_arr = np.zeros((nb_pass,np.max([len(idx_cat[p]) for p in range(nb_objs)])), dtype = np.int32)
        cdef int[:] idxs_compr = np.zeros((nb_objs,), dtype = np.int32)
        idxs_compr.base[obj_pass.base.nonzero()[0]] = np.arange(np.sum(obj_pass.base))
        for p in range(nb_objs):
            if obj_pass[p] == 1:
                idx_cat_arr.base[idxs_compr[p],:obj_size[p]] = np.array(idx_cat[p])
        # Define memoryviews
        cdef int[:] idxs_keep = np.zeros((nb_objs,), dtype = np.int32)
        idxs_keep.base[obj_keep.base.nonzero()[0]] = np.arange(np.sum(obj_keep.base))
        cdef float[:,:] dens_profs = np.zeros((nb_keep, r_res), dtype = np.float32)
        cdef int[:] shell = np.zeros((idx_cat_arr.shape[1]), dtype = np.int32)
        cdef float[:,:] xyz_obj = np.zeros((idx_cat_arr.shape[1], 3), dtype = np.float32)
        cdef float[:] m_obj = np.zeros((idx_cat_arr.shape[1]), dtype = np.float32)
        cdef float[:,:] centers = np.zeros((nb_pass,3), dtype = np.float32)
        cdef float[:] dists = np.zeros((idx_cat_arr.shape[1],), dtype = np.float32) # Distances from center of halo
        cdef float[:] hs = np.zeros((idx_cat_arr.shape[1],), dtype = np.float32) # Kernel widths
        for p in range(nb_objs): # Calculate centers of objects
            if obj_pass[p] == 1:
                xyz_ = respectPBCNoRef(xyz.base[idx_cat_arr.base[idxs_compr[p],:obj_size[p]]], L_BOX)
                if CENTER == 'mode':
                    centers.base[idxs_compr[p]] = calcMode(xyz_, masses.base[idx_cat_arr.base[idxs_compr[p],:obj_size[p]]], 1000)
                else:
                    centers.base[idxs_compr[p]] = calcCoM(xyz_, masses.base[idx_cat_arr.base[idxs_compr[p],:obj_size[p]]])
            if obj_keep[p] == 1:
                for n in range(obj_size[p]):
                    xyz_obj[n] = xyz[idx_cat_arr[idxs_compr[p],n]]
                    m_obj[n] = masses[idx_cat_arr[idxs_compr[p],n]]
                xyz_obj[:obj_size[p]] = CythonHelpers.respectPBCNoRef(xyz_obj[:obj_size[p]], L_BOX)
                dists = np.linalg.norm(xyz_obj.base[:obj_size[p]]-centers.base[idxs_compr[p]], axis = 1)
                hs = 0.005*ROverR200[-1]*r200s[p]*(dists.base/(ROverR200[-1]*r200s[p]))**(0.5)
                for r_idx in prange(r_res, schedule = 'dynamic', nogil = True):
                    for n in range(obj_size[p]):
                        dens_profs[idxs_keep[p]][r_idx] += m_obj[n]*CythonHelpers.calcKTilde(ROverR200[r_idx]*r200s[p], dists[n], hs[n])/(hs[n]**3)
        del obj_size; del obj_pass; del idx_cat_arr; del idx_cat; del idxs_compr; del idxs_keep; del shell; del xyz_obj; del m_obj; del centers
        del dists; del hs
        return dens_profs.base
    if(not hasattr(calcDensProfsKernelBased, "inner")):
        calcDensProfsKernelBased.inner = np_cache_factory(5,1)(inner)
    calcDensProfsKernelBased.inner(xyz.base, obj_keep.base, masses.base, r200s.base, ROverR200.base, idx_cat, MIN_NUMBER_PTCS, L_BOX, CENTER)
    return calcDensProfsKernelBased.inner(xyz.base, obj_keep.base, masses.base, r200s.base, ROverR200.base, idx_cat, MIN_NUMBER_PTCS, L_BOX, CENTER)
