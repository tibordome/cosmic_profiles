#cython: language_level=3
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
cimport cython
cimport openmp
from cython.parallel import prange
from cosmic_profiles.common.caching import np_cache_factory

@cython.embedsignature(True) 
cdef int[:] calcCSHIdxs(int[:] obj_idxs, int start_idx, int nb_shs, int csh_size, int MIN_NUMBER_PTCS) nogil:
    """ Return the indices of the particles that belong to the CSH
    
    :param obj_idxs: array to store the indices
    :type obj_idxs: int array
    :param start_idx: first index that belongs to this CSH
    :type start_idx: int
    :param nb_shs: number of SHs in each FoF-halo
    :type nb_shs: (N1,) ints
    :param csh_size: number of DM particles in the SHs
    :type csh_size: (N2,) ints
    :param MIN_NUMBER_PTCS: minimum number of particles for CSH to be valid
    :type MIN_NUMBER_PTCS: int
    :return: obj_idxs filled partially with indices
    :rtype: int array"""
    
    cdef int l
    if nb_shs > 0:
        l = start_idx
        if csh_size >= MIN_NUMBER_PTCS: # Only add objects that have sufficient resolution
            while l < start_idx+csh_size: # Add content of Object = Subfind 0 == CSH
                obj_idxs[l-start_idx] = l
                l += 1
    return obj_idxs

@cython.embedsignature(True)
@cython.binding(True)
def calcObjCat(int[:] nb_shs, int[:] sh_len, int[:] fof_sizes, float[:] group_r200, int MIN_NUMBER_PTCS):
    """ Construct central subhalo (CSH) catalogue from FoF/SH info
    
    Note that the indices returned in each CSH are 'true index + 1'
    
    :param nb_shs: number of SHs in each FoF-halo
    :type nb_shs: (N1,) ints
    :param sh_len: number of DM particles in each subhalo
    :type sh_len: (N2,) ints, N2>N1
    :param fof_sizes: number of particles in the FoF-halos
    :type fof_sizes: (N1,) ints
    :param group_r200: R200-radius of FoF-halos
    :type group_r200: (N1,) floats
    :param MIN_NUMBER_PTCS: minimum number of particles for CSH to be valid
    :type MIN_NUMBER_PTCS: int
    :return: obj_cat: indices,
        obj_r200: R200-radii, obj_size: number of particles in each object
    :rtype: int array, float array, int array"""
    def inner(int[:] nb_shs, int[:] sh_len, int[:] fof_sizes, float[:] group_r200, int MIN_NUMBER_PTCS):
        cdef int nb_halos = len(nb_shs)
        cdef int[:] obj_pass = np.zeros((nb_halos,), dtype = np.int32)
        cdef int[:] obj_size = np.zeros((nb_halos,), dtype = np.int32) # CSH size
        cdef float[:] h_r200 = np.zeros((nb_halos,), dtype = np.float32) # Note: = 0.0 for object that lacks SHs
        cdef int p
        cdef int q
        cdef int idx_sum
        cdef int start_idx
        for p in prange(nb_halos, schedule = 'dynamic', nogil = True):
            if nb_shs[p] > 0: # There is at least 1 subhalo, which is needed for R200 to be non-zero
                idx_sum = 0
                for q in range(p):
                    idx_sum = idx_sum+nb_shs[q]
                if sh_len[idx_sum] >= MIN_NUMBER_PTCS: # Only add halos that have sufficient resolution
                    obj_pass[p] = 1
                    obj_size[p] = sh_len[idx_sum]
            h_r200[p] = group_r200[p]
        cdef int[:,:] obj_cat = np.zeros((np.sum(obj_pass.base),np.max(obj_size.base)), dtype = np.int32) # Object catalogue (1 halo ~ Halo is the unit), particle indices in each object, empty list entry [] if object is empty 
        cdef int[:] idxs_compr = np.zeros((nb_halos,), dtype = np.int32)
        idxs_compr.base[obj_pass.base.nonzero()[0]] = np.arange(np.sum(obj_pass.base))
        for p in prange(nb_halos, schedule = 'dynamic', nogil = True):
            if obj_pass[p] == 1:
                idx_sum = 0
                start_idx = 0
                for q in range(p):
                    idx_sum = idx_sum+nb_shs[q]
                    start_idx = start_idx+fof_sizes[q]
                obj_cat[idxs_compr[p]] = calcCSHIdxs(obj_cat[idxs_compr[p]], start_idx, nb_shs[p], sh_len[idx_sum], MIN_NUMBER_PTCS)
        # Creating 1D index cat here is not very efficient
        obj_cat_out = np.empty(0, np.int32)
        for p in range(len(obj_cat)):
            obj_cat_out = np.hstack((obj_cat_out, obj_cat.base[p, :obj_size.base[obj_pass.base.nonzero()[0]][p]]))
        del obj_cat
        return obj_cat_out, h_r200.base[obj_pass.base.nonzero()[0]], obj_size.base[obj_pass.base.nonzero()[0]]
    if(not hasattr(calcObjCat, "inner")):
        calcObjCat.inner = np_cache_factory(4,0)(inner)
    calcObjCat.inner(nb_shs.base, sh_len.base, fof_sizes.base, group_r200.base, MIN_NUMBER_PTCS)
    return calcObjCat.inner(nb_shs.base, sh_len.base, fof_sizes.base, group_r200.base, MIN_NUMBER_PTCS)