#cython: language_level=3
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 13:17:51 2021
"""

cimport openmp
import numpy as np
cimport cython
from cosmic_profiles.cython_helpers.helper_class cimport CythonHelpers
from cosmic_profiles.common.python_routines import respectPBCNoRef, findMode, getCoM
from cython.parallel import prange

@cython.embedsignature(True)
def getDensProfsDirectBinning(cat, float[:,:] xyz, int[:] obj_keep, float[:] masses, float[:] r200s, float[:] ROverR200, int MIN_NUMBER_PTCS, float L_BOX, str CENTER):
    
    cdef int nb_objs = len(cat)
    # Determine endpoints of radial bins
    cdef float[:] rad_bins = np.hstack(([np.float32(1e-8), (ROverR200.base[:-1] + ROverR200.base[1:])/2., ROverR200.base[-1]])) # Length = ROverR200.shape[0]+1
    cdef int r_res = ROverR200.shape[0]
    cdef int p
    cdef int n
    cdef int[:] obj_size = np.zeros((nb_objs,), dtype = np.int32)
    cdef int[:] obj_pass = np.zeros((nb_objs,), dtype = np.int32)
    for p in range(nb_objs):
        if len(cat[p]) >= MIN_NUMBER_PTCS: # Only add objects that have sufficient resolution
            obj_size[p] = len(cat[p])
            obj_pass[p] = 1
        else:
            if obj_keep[p] == 1:
                raise ValueError("Having a 1 in obj_keep for an object for which cat is empty is not allowed.")
    cdef int nb_pass = np.sum(obj_pass.base)
    cdef int nb_keep = np.sum(obj_keep.base)
    if nb_objs == 0:
        return ROverR200.base, np.zeros((0,r_res), dtype = np.float32), np.zeros((0,r_res), dtype = np.float32), np.zeros((0,r_res), dtype = np.float32), np.zeros((0,r_res), dtype = np.float32)
    # Transform cat to int[:,:]
    cdef int[:,:] cat_arr = np.zeros((nb_pass,np.max([len(cat[p]) for p in range(nb_objs)])), dtype = np.int32)
    cdef int[:] idxs_compr = np.zeros((nb_objs,), dtype = np.int32)
    idxs_compr.base[obj_pass.base.nonzero()[0]] = np.arange(np.sum(obj_pass.base))
    for p in range(nb_objs):
        if obj_pass[p] == 1:
            cat_arr.base[idxs_compr[p],:obj_size[p]] = np.array(cat[p])
    # Define memoryviews
    cdef int[:] idxs_keep = np.zeros((nb_objs,), dtype = np.int32)
    idxs_keep.base[obj_keep.base.nonzero()[0]] = np.arange(np.sum(obj_keep.base))
    cdef float[:,:] dens_profs = np.zeros((nb_keep, r_res), dtype = np.float32)
    cdef float[:,:] Mencls = np.zeros((nb_keep, r_res), dtype = np.float32)
    cdef float[:,:] rad_bins_tiled = np.reshape(np.tile(rad_bins.base, reps = openmp.omp_get_max_threads()), (openmp.omp_get_max_threads(), r_res+1))
    cdef int[:,:] shell = np.zeros((openmp.omp_get_max_threads(), cat_arr.shape[1]), dtype = np.int32)
    cdef float[:,:,:] xyz_obj = np.zeros((openmp.omp_get_max_threads(), cat_arr.shape[1], 3), dtype = np.float32)
    cdef float[:,:] m_obj = np.zeros((openmp.omp_get_max_threads(), cat_arr.shape[1]), dtype = np.float32)
    cdef float[:,:] centers = np.zeros((nb_pass,3), dtype = np.float32)
    for p in range(nb_objs): # Calculate centers of objects
        if obj_pass[p] == 1:
            xyz_ = respectPBCNoRef(xyz.base[cat_arr.base[idxs_compr[p],:obj_size[p]]], L_BOX)
            if CENTER == 'mode':
                centers.base[idxs_compr[p]] = findMode(xyz_, masses.base[cat_arr.base[idxs_compr[p],:obj_size[p]]], 1000)
            else:
                centers.base[idxs_compr[p]] = getCoM(xyz_, masses.base[cat_arr.base[idxs_compr[p],:obj_size[p]]])
    for p in prange(nb_objs, schedule = 'dynamic', nogil = True):
        if obj_keep[p] == 1:
            for n in range(obj_size[p]):
                xyz_obj[openmp.omp_get_thread_num(),n] = xyz[cat_arr[idxs_compr[p],n]]
                m_obj[openmp.omp_get_thread_num(),n] = masses[cat_arr[idxs_compr[p],n]]
            xyz_obj[openmp.omp_get_thread_num(),:obj_size[p]] = CythonHelpers.respectPBCNoRef(xyz_obj[openmp.omp_get_thread_num(),:obj_size[p]], L_BOX)
            dens_profs[idxs_keep[p]] = CythonHelpers.getDensProfBruteForce(xyz_obj[openmp.omp_get_thread_num(),:obj_size[p]], m_obj[openmp.omp_get_thread_num(),:obj_size[p]], centers[idxs_compr[p]], r200s[p], rad_bins_tiled[openmp.omp_get_thread_num()], dens_profs[idxs_keep[p]], shell[openmp.omp_get_thread_num()])
    
    return ROverR200.base, dens_profs.base

@cython.embedsignature(True)
def getDensProfsKernelBased(cat, float[:,:] xyz, int[:] obj_keep, float[:] masses, float[:] r200s, float[:] ROverR200, int MIN_NUMBER_PTCS, float L_BOX, str CENTER):
    
    cdef int nb_objs = len(cat)
    cdef int r_res = ROverR200.shape[0]
    cdef int p
    cdef int r_idx
    cdef int n
    cdef int[:] obj_size = np.zeros((nb_objs,), dtype = np.int32)
    cdef int[:] obj_pass = np.zeros((nb_objs,), dtype = np.int32)
    for p in range(nb_objs):
        if len(cat[p]) >= MIN_NUMBER_PTCS: # Only add objects that have sufficient resolution
            obj_size[p] = len(cat[p])
            obj_pass[p] = 1
        else:
            if obj_keep[p] == 1:
                raise ValueError("Having a 1 in obj_keep for an object for which cat is empty is not allowed.")
    cdef int nb_pass = np.sum(obj_pass.base)
    cdef int nb_keep = np.sum(obj_keep.base)
    if nb_objs == 0:
        return ROverR200.base, np.zeros((0,r_res), dtype = np.float32), np.zeros((0,r_res), dtype = np.float32), np.zeros((0,r_res), dtype = np.float32), np.zeros((0,r_res), dtype = np.float32)
    # Transform cat to int[:,:]
    cdef int[:,:] cat_arr = np.zeros((nb_pass,np.max([len(cat[p]) for p in range(nb_objs)])), dtype = np.int32)
    cdef int[:] idxs_compr = np.zeros((nb_objs,), dtype = np.int32)
    idxs_compr.base[obj_pass.base.nonzero()[0]] = np.arange(np.sum(obj_pass.base))
    for p in range(nb_objs):
        if obj_pass[p] == 1:
            cat_arr.base[idxs_compr[p],:obj_size[p]] = np.array(cat[p])
    # Define memoryviews
    cdef int[:] idxs_keep = np.zeros((nb_objs,), dtype = np.int32)
    idxs_keep.base[obj_keep.base.nonzero()[0]] = np.arange(np.sum(obj_keep.base))
    cdef float[:,:] dens_profs = np.zeros((nb_keep, r_res), dtype = np.float32)
    cdef float[:,:] Mencls = np.zeros((nb_keep, r_res), dtype = np.float32)
    cdef int[:] shell = np.zeros((cat_arr.shape[1]), dtype = np.int32)
    cdef float[:,:] xyz_obj = np.zeros((cat_arr.shape[1], 3), dtype = np.float32)
    cdef float[:] m_obj = np.zeros((cat_arr.shape[1]), dtype = np.float32)
    cdef float[:,:] centers = np.zeros((nb_pass,3), dtype = np.float32)
    cdef float[:] dists = np.zeros((cat_arr.shape[1],), dtype = np.float32) # Distances from center of halo
    cdef float[:] hs = np.zeros((cat_arr.shape[1],), dtype = np.float32) # Kernel widths
    for p in range(nb_objs): # Calculate centers of objects
        if obj_pass[p] == 1:
            xyz_ = respectPBCNoRef(xyz.base[cat_arr.base[idxs_compr[p],:obj_size[p]]], L_BOX)
            if CENTER == 'mode':
                centers.base[idxs_compr[p]] = findMode(xyz_, masses.base[cat_arr.base[idxs_compr[p],:obj_size[p]]], 1000)
            else:
                centers.base[idxs_compr[p]] = getCoM(xyz_, masses.base[cat_arr.base[idxs_compr[p],:obj_size[p]]])
        if obj_keep[p] == 1:
            for n in range(obj_size[p]):
                xyz_obj[n] = xyz[cat_arr[idxs_compr[p],n]]
                m_obj[n] = masses[cat_arr[idxs_compr[p],n]]
            xyz_obj[:obj_size[p]] = CythonHelpers.respectPBCNoRef(xyz_obj[:obj_size[p]], L_BOX)
            dists = np.linalg.norm(xyz_obj.base[:obj_size[p]]-centers.base[idxs_compr[p]], axis = 1)
            hs = 0.005*ROverR200[-1]*r200s[p]*(dists.base/(ROverR200[-1]*r200s[p]))**(0.5)
            for r_idx in prange(r_res, schedule = 'dynamic', nogil = True):
                for n in range(obj_size[p]):
                    dens_profs[idxs_keep[p]][r_idx] += m_obj[n]*CythonHelpers.getKTilde(ROverR200[r_idx]*r200s[p], dists[n], hs[n])/(hs[n]**3)
    
    return ROverR200.base, dens_profs.base
