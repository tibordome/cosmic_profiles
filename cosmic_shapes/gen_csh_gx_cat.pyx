#cython: language_level=3
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 12:24:04 2022
"""

from cython_helpers cimport CythonHelpers
import numpy as np
cimport cython
cimport openmp
from cython.parallel import prange

@cython.embedsignature(True)
cdef int createGxs(float[:] star_xyz, float[:,:] fof_com, int[:] nb_shs, float[:,:] sh_com, float L_BOX) nogil:
    """ Find closest FoF-halo for star particle at position `star_xyz`
    
    :param star_xyz: position of star particle
    :type star_xyz: (3,) float
    :param fof_com: COMs of FoF-halos
    :type fof_com: (N1, 3) floats
    :param nb_shs: number of SHs in each FoF-halo
    :type nb_shs: (N1,) ints
    :param sh_com: COMs of the SHs
    :type sh_com: (N2, 3) floats
    :param L_BOX: side length of simulation box, in cMpc/h
    :type L_BOX: float
    :return: index of FoF-halo to which star particle is closest, -1
        if star particle is to be discarded (since closer to some SH other than the CSH)
    :rtype: int"""
    
    cdef bint discard = False
    cdef int argmin = 0
    cdef int argmin_2 = 0
    cdef float dist_x
    cdef float dist_y
    cdef float dist_z
    cdef float sh_com_x
    cdef float sh_com_y
    cdef float sh_com_z    
    cdef float dist = 0.0
    cdef int idx_base = 0
    cdef int run
        
    # Find closest FoF-halo
    for run in range(fof_com.shape[0]):
        dist_x = CythonHelpers.cython_abs(star_xyz[0]-fof_com[run,0])
        if dist_x > L_BOX/2:
            dist_x = L_BOX-dist_x
        dist_y = CythonHelpers.cython_abs(star_xyz[1]-fof_com[run,1])
        if dist_y > L_BOX/2:
            dist_y = L_BOX-dist_y
        dist_z = CythonHelpers.cython_abs(star_xyz[2]-fof_com[run,2])
        if dist_z > L_BOX/2:
            dist_z = L_BOX-dist_z
        if run == 0:
            dist = dist_x**2+dist_y**2+dist_z**2
        else:
            if dist_x**2+dist_y**2+dist_z**2 < dist:
                argmin = run
                dist = dist_x**2+dist_y**2+dist_z**2
    
    # If star ptc is closer to some SH other than the CSH, discard star particle
    if nb_shs[argmin] != 0: # There are > 0 SHs in FoF halo. Otherwise there is nothing to check
        idx_base = 0
        for run in range(argmin):
            idx_base = idx_base + nb_shs[run]
        for run in range(nb_shs[argmin]):
            sh_com_x = sh_com[idx_base+run,0]
            dist_x = CythonHelpers.cython_abs(star_xyz[0]-sh_com_x)
            if dist_x > L_BOX/2:
                dist_x = L_BOX-dist_x
            sh_com_y = sh_com[idx_base+run,1]
            dist_y = CythonHelpers.cython_abs(star_xyz[1]-sh_com_y)
            if dist_y > L_BOX/2:
                dist_y = L_BOX-dist_y
            sh_com_z = sh_com[idx_base+run,2]
            dist_z = CythonHelpers.cython_abs(star_xyz[2]-sh_com_z)
            if dist_z > L_BOX/2:
                dist_z = L_BOX-dist_z
            if run == 0:
                dist = dist_x**2+dist_y**2+dist_z**2
            else:
                if dist_x**2+dist_y**2+dist_z**2 < dist:
                    argmin_2 = run
                    dist = dist_x**2+dist_y**2+dist_z**2        
        if argmin_2 != 0: # Star ptc is closer to some SH != CSH
            discard = True
    if discard == False:
        return argmin
    else:
        return -1
    
@cython.embedsignature(True)
def getGxCat(float[:,:] star_xyz, float[:,:] fof_com, int[:] nb_shs, float[:,:] sh_com, float L_BOX, int MIN_NUMBER_STAR_PTCS):
    """ Construct galaxy catalogue
    
    The approach is very simple: Assign each star particle to its nearest FoF-halo, 
    and bundle the star particles into newly obtained 'galaxies'. If the star particle
    is closer to a subhalo (SH) that is not the central subhalo (CSH), discard the star particle.
    
    :param star_xyz: position of star particles
    :type star_xyz: (N, 3,) floats
    :param fof_com: COMs of FoF-halos
    :type fof_com: (N1, 3) floats
    :param nb_shs: number of SHs in each FoF-halo
    :type nb_shs: (N1,) ints
    :param sh_com: COMs of the SHs
    :type sh_com: (N2, 3) floats
    :param L_BOX: side length of simulation box, in cMpc/h
    :type L_BOX: float
    :param MIN_NUMBER_STAR_PTCS: minimum number of star particles for galaxy to be valid
    :type MIN_NUMBER_STAR_PTCS: int
    :return: galaxy catalogue, containing indices of star particles belong to each galaxy
    :rtype: list of N1 int lists containing indices"""
    cdef int nb_halos = fof_com.shape[0]
    cdef int nb_stars = star_xyz.shape[0]
    cdef int[:] gx_pass = np.zeros((nb_halos,), dtype = np.int32)
    cdef int[:] belongs_to = np.zeros((nb_stars,), dtype = np.int32)
    cdef int[:] occ = np.zeros((nb_halos,), dtype = np.int32)
    cdef float[:,:,:] fof_com_ext = np.repeat(fof_com.base[np.newaxis,:,:], openmp.omp_get_max_threads(), axis = 0) 
    cdef float[:,:,:] sh_com_ext = np.repeat(sh_com.base[np.newaxis,:,:], openmp.omp_get_max_threads(), axis = 0) 
    cdef int[:,:] nb_shs_ext = np.repeat(nb_shs.base[np.newaxis,:], openmp.omp_get_max_threads(), axis = 0) 
    cdef int p
    for p in prange(nb_stars, schedule = 'dynamic', nogil = True):
        belongs_to[p] = createGxs(star_xyz[p], fof_com_ext[openmp.omp_get_thread_num()], nb_shs_ext[openmp.omp_get_thread_num()], sh_com_ext[openmp.omp_get_thread_num()], L_BOX)
        if belongs_to[p] != -1:
            occ[belongs_to[p]] += 1
    for p in prange(nb_halos, schedule = 'dynamic', nogil = True):
        if occ[p] >= MIN_NUMBER_STAR_PTCS: # Only add halos that have sufficient resolution
            gx_pass[p] = 1  
    gx_cat = [[] for i in range(nb_halos)]
    for p in range(nb_stars):
        if belongs_to[p] != -1 and gx_pass[belongs_to[p]] == 1:
            gx_cat[belongs_to[p]].append(p)
    return gx_cat
  
@cython.embedsignature(True)       
cdef int[:] getCSHIdxs(int[:] h_idxs, int start_idx, int fof_dm_size, int nb_shs, int csh_size, int MIN_NUMBER_DM_PTCS) nogil:
    """ Return the indices of the DM particles that belong to the CSH
    
    :param h_idxs: array to store the indices
    :type h_idxs: int array
    :param start_idx: first index that belongs to this CSH
    :type start_idx: int
    :param fof_dm_size: number of particles in the FoF-halos
    :type fof_dm_size: (N1,) ints
    :param nb_shs: number of SHs in each FoF-halo
    :type nb_shs: (N1,) ints
    :param csh_size: number of particles in the SHs
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
def getCSHCat(int[:] nb_shs, int[:] sh_len, int[:] fof_dm_sizes, float[:] group_r200, float[:] halo_masses, int MIN_NUMBER_DM_PTCS):
    """ Construct central subhalo (CSH) catalogue from FoF/SH info
    
    Note that the indices returned in each CSH are 'true index + 1'
    
    :param nb_shs: number of SHs in each FoF-halo
    :type nb_shs: (N1,) ints
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
            h_cat[idxs_compr[p]] = getCSHIdxs(h_cat[idxs_compr[p]], start_idx, fof_dm_sizes[p], nb_shs[p], sh_len[idx_sum], MIN_NUMBER_DM_PTCS)
    return h_cat.base, h_r200.base, h_pass.base
