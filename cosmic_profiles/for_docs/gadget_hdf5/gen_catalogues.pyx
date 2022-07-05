#cython: language_level=3
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
cimport cython
cimport openmp
from cython.parallel import prange

@cython.embedsignature(True) 
def calcCSHIdxs(int[:] h_idxs, int start_idx, int fof_dm_size, int nb_shs, int csh_size, int MIN_NUMBER_DM_PTCS):
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
    
    return

@cython.embedsignature(True)
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
    return

@cython.embedsignature(True)
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
    return