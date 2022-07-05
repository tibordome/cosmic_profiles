#cython: language_level=3
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
cimport cython
cimport openmp
from cython.parallel import prange
from cosmic_profiles.common.caching import np_cache_factory

@cython.embedsignature(True) 
cdef int[:] calcCSHIdxs(int[:] h_idxs, int start_idx, int fof_dm_size, int nb_shs, int csh_size, int MIN_NUMBER_DM_PTCS) nogil:
    """ Return the indices of the DM particles that belong to the CSH
    
    :param h_idxs: array to store the indices
    :type h_idxs: int array
    :param start_idx: first index that belongs to this CSH
    :type start_idx: int
    :param fof_dm_size: number of DM particles in the FoF-halos
    :type fof_dm_size: (N1,) ints
    :param nb_shs: number of SHs in each FoF-halo
    :type nb_shs: (N1,) ints
    :param csh_size: number of DM particles in the SHs
    :type csh_size: (N2,) ints
    :param MIN_NUMBER_DM_PTCS: minimum number of DM particles for CSH to be valid
    :type MIN_NUMBER_DM_PTCS: int
    :return: h_idxs filled partially with indices (+1, to allow 0 to be interpreted as no index)
    :rtype: int array"""
    
    cdef int l
    if nb_shs == 0: # There is no Halo, so add all the "inner fuzz" to the catalogue
        if fof_dm_size != 0: # If there is not even any "inner fuzz", return nothing
            l = start_idx
            if fof_dm_size >= MIN_NUMBER_DM_PTCS: # Only add halos that have sufficient resolution
                while l < start_idx+fof_dm_size: # Add content of inner fuzz (after last Subfind, but since there is no Subfind, all of FOF)
                    h_idxs[l-start_idx] = l+1
                    l += 1
    else:
        l = start_idx
        if csh_size >= MIN_NUMBER_DM_PTCS: # Only add halos that have sufficient resolution
            while l < start_idx+csh_size: # Add content of Halo = Subfind 0 == CSH
                h_idxs[l-start_idx] = l+1
                l += 1
    return h_idxs

@cython.embedsignature(True)
@np_cache_factory(3,0)
def calcGxCat(int[:] nb_shs, int[:] sh_len_gx, int[:] fof_gx_size, int MIN_NUMBER_STAR_PTCS):
    """ Construct galaxy catalogue
    
     Note that the indices returned in each gx are 'true index + 1'
    
    :param nb_shs: number of SHs in each FoF-halo
    :type nb_shs: (N1,) ints
    :param sh_len_gx: number of star particles in each subhalo
    :type sh_len_gx: (N2,) ints, N2>N1
    :param fof_gx_size: number of star particles in the FoF-halos
    :type fof_gx_size: (N1,) ints
    :param MIN_NUMBER_STAR_PTCS: minimum number of star particles for gx to be valid
    :type MIN_NUMBER_STAR_PTCS: int
    :return: galaxy catalogue, containing indices of star particles belong to each galaxy
    :rtype: list of N1 int lists containing indices"""
    cdef int nb_halos = len(nb_shs)
    cdef int[:] gx_pass = np.zeros((nb_halos,), dtype = np.int32)
    cdef int[:] gx_size = np.zeros((nb_halos,), dtype = np.int32) # Either CSH or halo size (if no SHs exist)
    cdef int p
    cdef int q
    cdef int idx_sum
    cdef int start_idx
    for p in prange(nb_halos, schedule = 'dynamic', nogil = True):
        if nb_shs[p] == 0: # There is no Halo, so add all the "inner fuzz" to the catalogue
            if fof_gx_size[p] != 0: # If there is not even any "inner fuzz", return nothing
                if fof_gx_size[p] >= MIN_NUMBER_STAR_PTCS: # Only add gxs that have sufficient resolution
                    gx_pass[p] = 1
                    gx_size[p] = fof_gx_size[p]
        else:
            idx_sum = 0
            for q in range(p):
                idx_sum = idx_sum+nb_shs[q]
            if sh_len_gx[idx_sum] >= MIN_NUMBER_STAR_PTCS: # Only add gxs that have sufficient resolution
                gx_pass[p] = 1
                gx_size[p] = sh_len_gx[idx_sum]
    if nb_halos == 0:
        return np.zeros((0,0), dtype = np.int32), np.zeros((0,), dtype = np.float32)
    cdef int[:,:] gx_cat = np.zeros((np.sum(gx_pass.base),np.max(gx_size.base)), dtype = np.int32) # Gx catalogue, empty list entry [] if gx has too low resolution
    cdef int[:] idxs_compr = np.zeros((nb_halos,), dtype = np.int32)
    idxs_compr.base[gx_pass.base.nonzero()[0]] = np.arange(np.sum(gx_pass.base))
    for p in prange(nb_halos, schedule = 'dynamic', nogil = True):
        if gx_pass[p] == 1:
            idx_sum = 0
            start_idx = 0
            for q in range(p):
                idx_sum = idx_sum+nb_shs[q]
                start_idx = start_idx+fof_gx_size[q]
            gx_cat[idxs_compr[p]] = calcCSHIdxs(gx_cat[idxs_compr[p]], start_idx, fof_gx_size[p], nb_shs[p], sh_len_gx[idx_sum], MIN_NUMBER_STAR_PTCS)
    return gx_cat.base, gx_pass.base

@cython.embedsignature(True)
@np_cache_factory(5,0)
def calcCSHCat(int[:] nb_shs, int[:] sh_len, int[:] fof_dm_sizes, float[:] group_r200, float[:] halo_masses, int MIN_NUMBER_DM_PTCS):
    """ Construct central subhalo (CSH) catalogue from FoF/SH info
    
    Note that the indices returned in each CSH are 'true index + 1'
    
    :param nb_shs: number of SHs in each FoF-halo
    :type nb_shs: (N1,) ints
    :param sh_len: number of DM particles in each subhalo
    :type sh_len: (N2,) ints, N2>N1
    :param fof_dm_size: number of particles in the FoF-halos
    :type fof_dm_size: (N1,) ints
    :param group_r200: R200-radius of FoF-halos
    :type group_r200: (N1,) floats
    :param halo_masses: masses of FoF-halos
    :type halo_masses: (N1,) floats
    :param MIN_NUMBER_DM_PTCS: minimum number of DM particles for CSH to be valid
    :type MIN_NUMBER_DM_PTCS: int
    :return: h_cat: indices (+1, to allow 0 to be interpreted as no index),
        h_r200: R200-radii, h_pass: passed `MIN_NUMBER_DM_PTCS`-threshold or not
    :rtype: int array, float array, int array"""
    cdef int nb_halos = len(nb_shs)
    cdef int[:] h_pass = np.zeros((nb_halos,), dtype = np.int32)
    cdef int[:] h_size = np.zeros((nb_halos,), dtype = np.int32) # Either CSH or halo size (if no SHs exist)
    cdef float[:] h_r200 = np.zeros((nb_halos,), dtype = np.float32) # H R_200 (mean, not critical) values. Note: = 0.0 for halo that lacks SHs
    cdef int p
    cdef int q
    cdef int idx_sum
    cdef int start_idx
    for p in prange(nb_halos, schedule = 'dynamic', nogil = True):
        if nb_shs[p] == 0: # There is no Halo, so add all the "inner fuzz" to the catalogue
            if fof_dm_sizes[p] != 0: # If there is not even any "inner fuzz", return nothing
                if fof_dm_sizes[p] >= MIN_NUMBER_DM_PTCS: # Only add halos that have sufficient resolution
                    h_pass[p] = 1
                    h_size[p] = fof_dm_sizes[p]
        else:
            idx_sum = 0
            for q in range(p):
                idx_sum = idx_sum+nb_shs[q]
            if sh_len[idx_sum] >= MIN_NUMBER_DM_PTCS: # Only add halos that have sufficient resolution
                h_pass[p] = 1
                h_size[p] = sh_len[idx_sum]
        h_r200[p] = group_r200[p]
    cdef int[:,:] h_cat = np.zeros((np.sum(h_pass.base),np.max(h_size.base)), dtype = np.int32) # Halo catalogue (1 halo ~ Halo is the unit), DM particle indices in each Halo, empty list entry [] if Halo is empty 
    cdef int[:] idxs_compr = np.zeros((nb_halos,), dtype = np.int32)
    idxs_compr.base[h_pass.base.nonzero()[0]] = np.arange(np.sum(h_pass.base))
    for p in prange(nb_halos, schedule = 'dynamic', nogil = True):
        if h_pass[p] == 1:
            idx_sum = 0
            start_idx = 0
            for q in range(p):
                idx_sum = idx_sum+nb_shs[q]
                start_idx = start_idx+fof_dm_sizes[q]
            h_cat[idxs_compr[p]] = calcCSHIdxs(h_cat[idxs_compr[p]], start_idx, fof_dm_sizes[p], nb_shs[p], sh_len[idx_sum], MIN_NUMBER_DM_PTCS)
    return h_cat.base, h_r200.base, h_pass.base