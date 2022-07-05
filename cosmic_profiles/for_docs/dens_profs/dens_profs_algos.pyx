#cython: language_level=3
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
cimport cython

@cython.embedsignature(True)
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
    return
    
@cython.embedsignature(True)
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
    
    return

@cython.embedsignature(True)
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
    
    return