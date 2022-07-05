#cython: language_level=3
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

cimport openmp
import numpy as np
cimport cython
from cosmic_profiles.cython_helpers.helper_class cimport CythonHelpers
from cosmic_profiles.common.python_routines import respectPBCNoRef, calcMode, calcCoM
from cython.parallel import prange
from cosmic_profiles.common.caching import np_cache_factory

@cython.embedsignature(True)
@np_cache_factory(2,1)
def calcMassesCenters(float[:,:] xyz, float[:] masses, cat, int MIN_NUMBER_PTCS, float L_BOX, str CENTER):
    """ Calculate total mass and centers of objects
    
    :param xyz: positions of all simulation particles
    :type xyz: (N2,3) floats, N2 >> N1
    :param masses: masses of all simulation particles
    :type masses: (N2,) floats
    :param cat: list of indices defining the objects
    :type cat: list of length N1, each consisting of a list of int indices
    :param MIN_NUMBER_PTCS: minimum number of particles for object to qualify for morphology calculation
    :type MIN_NUMBER_PTCS: int
    :param L_BOX: box size
    :type L_BOX: float
    :param CENTER: density profiles will be calculated with respect to CENTER = 'mode' (point of highest density)
        or 'com' (center of mass) of each halo
    :type CENTER: str
    :return centers, m: centers and masses
    :rtype: (N,3) and (N,) floats"""
    # Transform cat to int[:,:]
    cdef int nb_objs = len(cat)
    cdef int p
    cdef int[:] obj_size = np.zeros((nb_objs,), dtype = np.int32)
    cdef int[:] obj_pass = np.zeros((nb_objs,), dtype = np.int32)
    for p in range(nb_objs):
        if len(cat[p]) >= MIN_NUMBER_PTCS: # Only add objects that have sufficient resolution
            obj_size[p] = len(cat[p]) 
            obj_pass[p] = 1
    cdef int nb_pass = np.sum(obj_pass.base)
    cdef int[:] idxs_compr = np.zeros((nb_objs,), dtype = np.int32)
    idxs_compr.base[obj_pass.base.nonzero()[0]] = np.arange(np.sum(obj_pass.base))
    cdef int[:,:] cat_arr = np.zeros((nb_pass,np.max([len(cat[p]) for p in range(nb_objs)])), dtype = np.int32)
    for p in range(nb_objs):
        if obj_pass[p] == 1:
            cat_arr.base[idxs_compr[p],:obj_size[p]] = np.array(cat[p])

    # Calculate centers and total masses of objects
    cdef float[:] m = np.zeros((nb_pass,), dtype = np.float32)
    cdef int n
    cdef float[:,:] centers = np.zeros((nb_pass,3), dtype = np.float32)
    for p in range(nb_objs): # Calculate centers of objects
        if obj_pass[p] == 1:
            xyz_ = respectPBCNoRef(xyz.base[cat_arr.base[idxs_compr[p],:obj_size[p]]], L_BOX)
            if CENTER == 'mode':
                centers.base[idxs_compr[p]] = calcMode(xyz_, masses.base[cat_arr.base[idxs_compr[p],:obj_size[p]]], 1000)
            else:
                centers.base[idxs_compr[p]] = calcCoM(xyz_, masses.base[cat_arr.base[idxs_compr[p],:obj_size[p]]])
    for p in prange(nb_objs, schedule = 'dynamic', nogil = True): # Calculate total mass of objects
        if obj_pass[p] == 1:
            for n in range(obj_size[p]):
                m[idxs_compr[p]] = m[idxs_compr[p]] + masses[cat_arr[idxs_compr[p],n]]
    
    return centers.base, m.base # Only rank = 0 content matters
    
@cython.embedsignature(True)
@np_cache_factory(5,1)
def calcDensProfsDirectBinning(float[:,:] xyz, int[:] obj_keep, float[:] masses, float[:] r200s, float[:] ROverR200, idx_cat, int MIN_NUMBER_PTCS, float L_BOX, str CENTER):
    """ Calculates density profiles for objects defined by indices found in `idx_cat`    
    
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
    
    cdef int nb_objs = len(idx_cat)
    # Determine endpoints of radial bins
    cdef float[:] rad_bins = np.hstack(([np.float32(1e-8), (ROverR200.base[:-1] + ROverR200.base[1:])/2., ROverR200.base[-1]])) # Length = ROverR200.shape[0]+1
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
    cdef float[:,:] Mencls = np.zeros((nb_keep, r_res), dtype = np.float32)
    cdef float[:,:] rad_bins_tiled = np.reshape(np.tile(rad_bins.base, reps = openmp.omp_get_max_threads()), (openmp.omp_get_max_threads(), r_res+1))
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
            dens_profs[idxs_keep[p]] = CythonHelpers.calcDensProfBruteForce(xyz_obj[openmp.omp_get_thread_num(),:obj_size[p]], m_obj[openmp.omp_get_thread_num(),:obj_size[p]], centers[idxs_compr[p]], r200s[p], rad_bins_tiled[openmp.omp_get_thread_num()], dens_profs[idxs_keep[p]], shell[openmp.omp_get_thread_num()])
    
    return dens_profs.base

@cython.embedsignature(True)
@np_cache_factory(5,1)
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
    cdef float[:,:] Mencls = np.zeros((nb_keep, r_res), dtype = np.float32)
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
    
    return dens_profs.base