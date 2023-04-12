#cython: language_level=3
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
cimport cython

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
    return
   
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
    return
   
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
    return
  
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
    return