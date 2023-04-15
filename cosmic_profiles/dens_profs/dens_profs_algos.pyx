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
def calcMassesCenters(float[:,:] xyz, float[:] masses, int[:] idx_cat, int[:] obj_size, float L_BOX, str CENTER):
    """ Calculate total mass and centers of objects
    
    :param xyz: positions of all simulation particles
    :type xyz: (N2,3) floats, N2 >> N1
    :param masses: masses of all simulation particles
    :type masses: (N2,) floats
    :param idx_cat: contains indices of particles belonging to an object
    :type idx_cat: (N3,) integers
    :param obj_size: indicates how many particles are in each object
    :type obj_size: (N1,) integers
    :param L_BOX: box size
    :type L_BOX: float
    :param CENTER: density profiles will be calculated with respect to CENTER = 'mode' (point of highest density)
        or 'com' (center of mass) of each halo
    :type CENTER: str
    :return centers, m: centers and masses
    :rtype: (N,3) and (N,) floats"""
    # Calculate centers and total masses of objects
    cdef int p
    cdef int n
    cdef int[:] offsets = np.int32(np.hstack((np.array([0]), np.cumsum(obj_size.base))))
    cdef float[:] m = np.zeros((obj_size.shape[0],), dtype = np.float32)
    cdef float[:,:] centers = np.zeros((obj_size.shape[0],3), dtype = np.float32)
    for p in range(obj_size.shape[0]): # Calculate centers of objects
        xyz_ = respectPBCNoRef(xyz.base[idx_cat.base[offsets[p]:offsets[p+1]]], L_BOX)
        if CENTER == 'mode':
            centers.base[p] = calcMode(xyz_, masses.base[idx_cat.base[offsets[p]:offsets[p+1]]], 1000)
        else:
            centers.base[p] = calcCoM(xyz_, masses.base[idx_cat.base[offsets[p]:offsets[p+1]]])
    for p in prange(obj_size.shape[0], schedule = 'dynamic', nogil = True): # Calculate total mass of objects
        for n in range(obj_size[p]):
            m[p] = m[p] + masses[idx_cat[offsets[p]+n]]
    del idx_cat; del obj_size; del masses
    return centers.base, m.base # Only rank = 0 content matters
   
@cython.embedsignature(True)
def calcDensProfsSphDirectBinning(float[:,:] xyz, float[:] masses, float[:] r200s, float[:] ROverR200, int[:] idx_cat, int[:] obj_size, float L_BOX, str CENTER):
    """ Calculates spherical shell-based density profiles for objects defined by indices found in `idx_cat`    
    
    Note: To calculate enclosed mass profiles, envoke ``CythonHelpers.calcMenclsBruteForce()`` instead of ``CythonHelpers.calcDensProfBruteForce()``
    
    :param xyz: positions of all simulation particles
    :type xyz: (N2,3) floats, N2 >> N1
    :param masses: masses of all simulation particles
    :type masses: (N2,) floats
    :param r200s: R200 values of the objects
    :type r200s: (N1,) floats
    :param ROverR200: radii at which the density profiles should be calculated,
        normalized by R200
    :type ROverR200: (r_res,) float array
    :param idx_cat: contains indices of particles belonging to an object
    :type idx_cat: (N3,) integers
    :param obj_size: indicates how many particles are in each object
    :type obj_size: (N1,) integers
    :param L_BOX: box size
    :type L_BOX: float
    :param CENTER: density profiles will be calculated with respect to CENTER = 'mode' (point of highest density)
        or 'com' (center of mass) of each halo
    :type CENTER: str
    :return: density profiles defined at ROverR200
    :rtype: (N1, r_res) float array"""
    cdef int nb_objs = obj_size.shape[0]
    cdef int[:] offsets = np.int32(np.hstack((np.array([0]), np.cumsum(obj_size.base))))
    cdef int largest_size = np.max(obj_size.base)
    # Determine endpoints of radial bins
    cdef float[:] bin_edges = np.hstack(([np.float32(1e-8), (ROverR200.base[:-1] + ROverR200.base[1:])/2., ROverR200.base[-1]])) # Length = ROverR200.shape[0]+1
    cdef int r_res = ROverR200.shape[0]
    cdef int p
    cdef int n
    if nb_objs == 0:
        return np.zeros((0,r_res), dtype = np.float32)
    # Define memoryviews
    cdef float[:,:] dens_profs = np.zeros((nb_objs, r_res), dtype = np.float32)
    cdef float[:,:] bin_edges_tiled = np.reshape(np.tile(bin_edges.base, reps = openmp.omp_get_max_threads()), (openmp.omp_get_max_threads(), r_res+1))
    cdef int[:,:] shell = np.zeros((openmp.omp_get_max_threads(), largest_size), dtype = np.int32)
    cdef float[:,:,:] xyz_obj = np.zeros((openmp.omp_get_max_threads(), largest_size, 3), dtype = np.float32)
    cdef float[:,:] m_obj = np.zeros((openmp.omp_get_max_threads(), largest_size), dtype = np.float32)
    cdef float[:,:] centers = np.zeros((nb_objs,3), dtype = np.float32)
    for p in range(nb_objs): # Calculate centers of objects
        xyz_ = respectPBCNoRef(xyz.base[idx_cat.base[offsets[p]:offsets[p+1]]], L_BOX)
        if CENTER == 'mode':
            centers.base[p] = calcMode(xyz_, masses.base[idx_cat.base[offsets[p]:offsets[p+1]]], 1000)
        else:
            centers.base[p] = calcCoM(xyz_, masses.base[idx_cat.base[offsets[p]:offsets[p+1]]])
    for p in prange(nb_objs, schedule = 'dynamic', nogil = True):
        for n in range(obj_size[p]):
            xyz_obj[openmp.omp_get_thread_num(),n] = xyz[idx_cat[offsets[p]+n]]
            m_obj[openmp.omp_get_thread_num(),n] = masses[idx_cat[offsets[p]+n]]
        xyz_obj[openmp.omp_get_thread_num(),:obj_size[p]] = CythonHelpers.respectPBCNoRef(xyz_obj[openmp.omp_get_thread_num(),:obj_size[p]], L_BOX)
        dens_profs[p] = CythonHelpers.calcDensProfBruteForceSph(xyz_obj[openmp.omp_get_thread_num(),:obj_size[p]], m_obj[openmp.omp_get_thread_num(),:obj_size[p]], centers[p], r200s[p], bin_edges_tiled[openmp.omp_get_thread_num()], dens_profs[p], shell[openmp.omp_get_thread_num()])
    del bin_edges; del obj_size; del idx_cat; del shell; del bin_edges_tiled; del xyz_obj; del m_obj; del centers
    return dens_profs.base
   
@cython.embedsignature(True)
def calcDensProfsEllDirectBinning(float[:,:] xyz, float[:] masses, float[:] r200s, float[:] ROverR200, float[:,:] a, float[:,:] b, float[:,:] c, float[:,:,:] major, float[:,:,:] inter, float[:,:,:] minor, int[:] idx_cat, int[:] obj_size, float L_BOX, str CENTER):
    """ Calculates ellipsoidal shell-based density profiles for objects defined by indices found in `idx_cat`    
    
    Note: To calculate enclosed mass profiles, envoke ``CythonHelpers.calcMenclsBruteForceEll()`` instead of ``CythonHelpers.calcDensProfBruteForceEll()``
    
    :param xyz: positions of all simulation particles
    :type xyz: (N2,3) floats, N2 >> N1
    :param masses: masses of all simulation particles
    :type masses: (N2,) floats
    :param r200s: R200 values of the objects
    :type r200s: (N1,) floats
    :param ROverR200: radii at which the density profiles should be calculated,
        normalized by R200
    :type ROverR200: (r_res,) float array
    :param idx_cat: contains indices of particles belonging to an object
    :type idx_cat: (N3,) integers
    :param obj_size: indicates how many particles are in each object
    :type obj_size: (N1,) integers
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
    :rtype: (N1, r_res) float array"""
    assert a.shape[0] == b.shape[0]
    assert b.shape[0] == c.shape[0]
    cdef int nb_objs = obj_size.shape[0]
    cdef int[:] offsets = np.int32(np.hstack((np.array([0]), np.cumsum(obj_size.base))))
    cdef int largest_size = np.max(obj_size.base)
    # Determine endpoints of radial bins
    cdef float[:] r_midpoints = (ROverR200.base[:-1] + ROverR200.base[1:])/2
    cdef float[:] bin_edges = np.hstack(([np.float32(r_midpoints[0]*r_midpoints[0]/r_midpoints[1]), r_midpoints.base, ROverR200.base[-1]])) # Length = ROverR200.shape[0]+1
    cdef int r_res = ROverR200.shape[0]
    cdef int p
    cdef int n
    if nb_objs == 0:
        return np.zeros((0,r_res), dtype = np.float32)
    # Interpolate shape information to radii of interest
    cdef float[:,:] a_interpol = np.zeros((nb_objs,r_res+1), dtype = np.float32)
    cdef float[:,:] b_interpol = np.zeros((nb_objs,r_res+1), dtype = np.float32)
    cdef float[:,:] c_interpol = np.zeros((nb_objs,r_res+1), dtype = np.float32)
    cdef float[:,:,:] major_interpol = np.zeros((nb_objs,r_res,3), dtype = np.float32)
    cdef float[:,:,:] inter_interpol = np.zeros((nb_objs,r_res,3), dtype = np.float32)
    cdef float[:,:,:] minor_interpol = np.zeros((nb_objs,r_res,3), dtype = np.float32)
    for p in range(nb_objs):
        idx_cat.base[offsets[p]:offsets[p+1]] = np.array(idx_cat[p])
        r_vec = np.float32(a.base[p])
        a_inter = interp1d(r_vec, a.base[p], bounds_error=False, fill_value='extrapolate')
        b_inter = interp1d(r_vec, b.base[p], bounds_error=False, fill_value='extrapolate')
        c_inter = interp1d(r_vec, c.base[p], bounds_error=False, fill_value='extrapolate')
        majorx_inter = interp1d(r_vec, major.base[p,:,0], bounds_error=False, fill_value='extrapolate')
        majory_inter = interp1d(r_vec, major.base[p,:,1], bounds_error=False, fill_value='extrapolate')
        majorz_inter = interp1d(r_vec, major.base[p,:,2], bounds_error=False, fill_value='extrapolate')
        interx_inter = interp1d(r_vec, inter.base[p,:,0], bounds_error=False, fill_value='extrapolate')
        intery_inter = interp1d(r_vec, inter.base[p,:,1], bounds_error=False, fill_value='extrapolate')
        interz_inter = interp1d(r_vec, inter.base[p,:,2], bounds_error=False, fill_value='extrapolate')
        minorx_inter = interp1d(r_vec, minor.base[p,:,0], bounds_error=False, fill_value='extrapolate')
        minory_inter = interp1d(r_vec, minor.base[p,:,1], bounds_error=False, fill_value='extrapolate')
        minorz_inter = interp1d(r_vec, minor.base[p,:,2], bounds_error=False, fill_value='extrapolate')
        a_interpol.base[p] = a_inter(bin_edges.base*r200s[p])
        b_interpol.base[p] = b_inter(bin_edges.base*r200s[p])
        c_interpol.base[p] = c_inter(bin_edges.base*r200s[p])
        major_interpol.base[p,:,0] = majorx_inter(ROverR200.base*r200s[p])
        major_interpol.base[p,:,1] = majory_inter(ROverR200.base*r200s[p])
        major_interpol.base[p,:,2] = majorz_inter(ROverR200.base*r200s[p])
        inter_interpol.base[p,:,0] = interx_inter(ROverR200.base*r200s[p])
        inter_interpol.base[p,:,1] = intery_inter(ROverR200.base*r200s[p])
        inter_interpol.base[p,:,2] = interz_inter(ROverR200.base*r200s[p])
        minor_interpol.base[p,:,0] = minorx_inter(ROverR200.base*r200s[p])
        minor_interpol.base[p,:,1] = minory_inter(ROverR200.base*r200s[p])
        minor_interpol.base[p,:,2] = minorz_inter(ROverR200.base*r200s[p])
    # Calculate centers of objects
    cdef float[:,:] centers = np.zeros((nb_objs,3), dtype = np.float32)
    for p in range(nb_objs):
        xyz_ = respectPBCNoRef(xyz.base[idx_cat.base[offsets[p]:offsets[p+1]]], L_BOX)
        if CENTER == 'mode':
            centers.base[p] = calcMode(xyz_, masses.base[idx_cat.base[offsets[p]:offsets[p+1]]], 1000)
        else:
            centers.base[p] = calcCoM(xyz_, masses.base[idx_cat.base[offsets[p]:offsets[p+1]]])
    # Prepare density profile estimation
    cdef float[:,:] dens_profs = np.zeros((nb_objs, r_res), dtype = np.float32)
    cdef int[:,:] shell = np.zeros((openmp.omp_get_max_threads(), largest_size), dtype = np.int32)
    cdef float[:,:,:] xyz_obj = np.zeros((openmp.omp_get_max_threads(), largest_size, 3), dtype = np.float32)
    cdef float[:,:,:] xyz_obj_princ = np.zeros((openmp.omp_get_max_threads(), largest_size, 3), dtype = np.float32)
    cdef float[:,:] m_obj = np.zeros((openmp.omp_get_max_threads(), largest_size), dtype = np.float32)
    for p in prange(nb_objs, schedule = 'dynamic', nogil = True):
        for n in range(obj_size[p]):
            xyz_obj[openmp.omp_get_thread_num(),n] = xyz[idx_cat[offsets[p]+n]]
            m_obj[openmp.omp_get_thread_num(),n] = masses[idx_cat[offsets[p]+n]]
        xyz_obj[openmp.omp_get_thread_num(),:obj_size[p]] = CythonHelpers.respectPBCNoRef(xyz_obj[openmp.omp_get_thread_num(),:obj_size[p]], L_BOX)
        dens_profs[p] = CythonHelpers.calcDensProfBruteForceEll(xyz_obj[openmp.omp_get_thread_num(),:obj_size[p]], xyz_obj_princ[openmp.omp_get_thread_num(),:obj_size[p]], m_obj[openmp.omp_get_thread_num(),:obj_size[p]], centers[p], r200s[p], a_interpol[p], b_interpol[p], c_interpol[p], major_interpol[p], inter_interpol[p], minor_interpol[p], dens_profs[p], shell[openmp.omp_get_thread_num()])
    del bin_edges; del obj_size; del idx_cat; del shell; del xyz_obj; del m_obj; del centers
    del a_interpol; del b_interpol; del c_interpol; del major_interpol; del inter_interpol; del minor_interpol; del xyz_obj_princ; del a; del b; del c; del major; del inter; del minor
    return dens_profs.base
  
@cython.embedsignature(True)
def calcDensProfsKernelBased(float[:,:] xyz, float[:] masses, float[:] r200s, float[:] ROverR200, int[:] idx_cat, int[:] obj_size, float L_BOX, str CENTER):
    """ Calculates kernel-based density profiles for objects defined by indices found in `idx_cat`
    
    Note: For background on this kernel-based method consult Reed et al. 2003, https://arxiv.org/abs/astro-ph/0312544.
    
    :param xyz: positions of all simulation particles
    :type xyz: (N2,3) floats, N2 >> N1
    :param masses: masses of all simulation particles
    :type masses: (N2,) floats
    :param r200s: R200 values of the objects
    :type r200s: (N1,) floats
    :param ROverR200: radii at which the density profiles should be calculated,
        normalized by R200
    :type ROverR200: (r_res,) float array
    :param idx_cat: contains indices of particles belonging to an object
    :type idx_cat: (N3,) integers
    :param obj_size: indicates how many particles are in each object
    :type obj_size: (N1,) integers
    :param L_BOX: box size
    :type L_BOX: float
    :param CENTER: density profiles will be calculated with respect to CENTER = 'mode' (point of highest density)
        or 'com' (center of mass) of each halo
    :type CENTER: str
    :return: density profiles defined at ROverR200
    :rtype: (N1, r_res) float array"""
    
    cdef int nb_objs = obj_size.shape[0]
    cdef int[:] offsets = np.int32(np.hstack((np.array([0]), np.cumsum(obj_size.base))))
    cdef int largest_size = np.max(obj_size.base)
    cdef int r_res = ROverR200.shape[0]
    cdef int p
    cdef int r_idx
    cdef int n
    if nb_objs == 0:
        return np.zeros((0,r_res), dtype = np.float32)
    # Define memoryviews
    cdef float[:,:] dens_profs = np.zeros((nb_objs, r_res), dtype = np.float32)
    cdef int[:] shell = np.zeros((largest_size), dtype = np.int32)
    cdef float[:,:] xyz_obj = np.zeros((largest_size, 3), dtype = np.float32)
    cdef float[:] m_obj = np.zeros((largest_size), dtype = np.float32)
    cdef float[:,:] centers = np.zeros((nb_objs,3), dtype = np.float32)
    cdef float[:] dists = np.zeros((largest_size,), dtype = np.float32) # Distances from center of halo
    cdef float[:] hs = np.zeros((largest_size,), dtype = np.float32) # Kernel widths
    for p in range(nb_objs): # Calculate centers of objects
        xyz_ = respectPBCNoRef(xyz.base[idx_cat.base[offsets[p]:offsets[p+1]]], L_BOX)
        if CENTER == 'mode':
            centers.base[p] = calcMode(xyz_, masses.base[idx_cat.base[offsets[p]:offsets[p+1]]], 1000)
        else:
            centers.base[p] = calcCoM(xyz_, masses.base[idx_cat.base[offsets[p]:offsets[p+1]]])
        for n in range(obj_size[p]):
            xyz_obj[n] = xyz[idx_cat[offsets[p]+n]]
            m_obj[n] = masses[idx_cat[offsets[p]+n]]
        xyz_obj[:obj_size[p]] = CythonHelpers.respectPBCNoRef(xyz_obj[:obj_size[p]], L_BOX)
        dists = np.linalg.norm(xyz_obj.base[:obj_size[p]]-centers.base[p], axis = 1)
        hs = 0.005*ROverR200[-1]*r200s[p]*(dists.base/(ROverR200[-1]*r200s[p]))**(0.5)
        for r_idx in prange(r_res, schedule = 'dynamic', nogil = True):
            for n in range(obj_size[p]):
                dens_profs[p][r_idx] += m_obj[n]*CythonHelpers.calcKTilde(ROverR200[r_idx]*r200s[p], dists[n], hs[n])/(hs[n]**3)
    del obj_size; del idx_cat; del shell; del xyz_obj; del m_obj; del centers
    del dists; del hs
    return dens_profs.base