#cython: language_level=3
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
cimport cython
cimport openmp
from cython.parallel import prange

@cython.embedsignature(True) 
def calcCSHIdxs(int[:] obj_idxs, int start_idx, int nb_shs, int csh_size, int MIN_NUMBER_PTCS):
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
    return

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
    return