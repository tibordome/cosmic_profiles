#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def getEpsilon(idx_cat, xyz, masses, L_BOX, CENTER, angle=0.0):
    """ Calculate the complex ellipticity (z-projected)
    
    It is obtained from the shape tensor = centred (wrt mode) second mass moment tensor
    
    :param idx_cat: catalogue of objects (halos/gxs)
    :type idx_cat: list of lists of ints
    :param xyz: coordinates of particles of type 1 or type 4
    :type xyz: (N^3x3) floats
    :param masses: masses of particles of type 1 or type 4
    :type masses: (N^3x1) floats
    :param L_BOX: simulation box side length
    :type L_BOX: float, units: Mpc/h
    :param CENTER: shape quantities will be calculated with respect to CENTER = 'mode' (point of highest density)
        or 'com' (center of mass) of each halo
    :type CENTER: str
    :param angle: rotation of objects around z-axis before ellipticity is calculated (z-projected)
    :type angle: float
    :return: complex ellipticity
    :rtype: complex scalar
    """
    return

def getShape(Rs, d, param_interest, ERROR_METHOD, D_LOGSTART, D_LOGEND, D_BINS):
    """ Get average profile for param_interest (which is defined at all values of d)
    at all elliptical radii Rs
    
    :param Rs: elliptical radii of interest
    :type Rs: (N,) floats
    :param d: param_interest is defined at all elliptical radii d
    :type d: (N2,) floats
    :param param_interest: the quantity of interest defined at all elliptical radii d
    :type param_interest: (N2,) floats
    :param ERROR_METHOD: mean (if ERROR_METHOD == "bootstrap" or "SEM") or median
        (if ERROR_METHOD == "median_quantile") and the +- 1 sigma error attached
    :type ERROR_METHOD: string
    :param D_LOGSTART: logarithm of minimum ellipsoidal radius of interest, in units of R200 of parent halo
    :type D_LOGSTART: int
    :param D_LOGEND: logarithm of maximum ellipsoidal radius of interest, in units of R200 of parent halo
    :type D_LOGEND: int
    :param D_BINS: number of ellipsoidal radii of interest minus 1 (i.e. number of bins)
    :type D_BINS: int
    :return: mean/median, err_low, err_high
    :rtype: float, float, float"""
    return

def getShapeMs(Rs, d, idx_groups, group, param_interest, ERROR_METHOD, D_LOGSTART, D_LOGEND, D_BINS):
    """ Similar to getShape, but with mass-splitting"""
    return

def getShapeProfs(VIZ_DEST, SNAP, D_LOGSTART, D_LOGEND, D_BINS, start_time, obj_masses, obj_centers, d, q, s, major_full, MASS_UNIT=1e10, suffix = '_'):
    """
    Create a series of plots to analyze object shapes
    
    Plot intertial tensor axis ratios, triaxialities and ellipticity histograms.
    
    :param VIZ_DEST: visualisation folder destination
    :type VIZ_DEST: string
    :param SNAP: e.g. '024'
    :type SNAP: string
    :param D_LOGSTART: logarithm of minimum ellipsoidal radius of interest, in units of R200 of parent halo
    :type D_LOGSTART: int
    :param D_LOGEND: logarithm of maximum ellipsoidal radius of interest, in units of R200 of parent halo
    :type D_LOGEND: int
    :param D_BINS: number of ellipsoidal radii of interest minus 1 (i.e. number of bins)
    :type D_BINS: int
    :param start_time: time of start of shape analysis
    :type start_time: float
    :param obj_masses: total mass of objects, in internal units
    :type obj_masses: (N,) floats
    :param obj_centers: positions of centers of objects, in Mpc/h
    :type obj_centers: (N,3) floats
    :param d: param_interest is defined at all elliptical radii d
    :type d: (N, D_BINS+1) floats
    :param q: intermediate-to-major axis ratios
    :type q: (N, D_BINS+1) floats
    :param s: minor-to-major axis ratios
    :type s: (N, D_BINS+1) floats
    :param major_full: major axis vectors
    :type major_full: (N,D_BINS+1,3) floats
    :param MASS_UNIT: conversion factor from previous mass unit to M_sun/h
    :type MASS_UNIT: float
    :param suffix: either '_dm_' or '_gx_' or '' (latter for CosmicProfsDirect)
    :type suffix: string"""
    
    return

def getLocalTHist(VIZ_DEST, SNAP, D_LOGSTART, D_LOGEND, D_BINS, start_time, obj_masses, obj_centers, d, q, s, major_full, HIST_NB_BINS, frac_r200, MASS_UNIT, suffix = '_'):
    """ Plot triaxiality T histogram
    
    :param VIZ_DEST: visualisation folder destination
    :type VIZ_DEST: string
    :param SNAP: e.g. '024'
    :type SNAP: string
    :param D_LOGSTART: logarithm of minimum ellipsoidal radius of interest, in units of R200 of parent halo
    :type D_LOGSTART: int
    :param D_LOGEND: logarithm of maximum ellipsoidal radius of interest, in units of R200 of parent halo
    :type D_LOGEND: int
    :param D_BINS: number of ellipsoidal radii of interest minus 1 (i.e. number of bins)
    :type D_BINS: int
    :param start_time: time of start of shape analysis
    :type start_time: float
    :param obj_masses: masses of objects, in internal units
    :type obj_masses: (N,) floats
    :param obj_centers: centers of objects, each coordinate in Mpc/h
    :type obj_centers: (N,3) floats
    :param d: ellipsoidal radii at which shape profiles have been calculated
    :type d: (N, D_BINS+1) floats
    :param q: q-values
    :type q: (N, D_BINS+1) floats
    :param s: s-values
    :type s: (N, D_BINS+1) floats
    :param major_full: major axes at each radii
    :type major_full: (N, D_BINS+1, 3) floats
    :param HIST_NB_BINS: Number of histogram bins
    :type HIST_NB_BINS: int
    :param frac_r200: depth of objects to plot triaxiality, in units of R200
    :type frac_r200: float
    :param MASS_UNIT: conversion factor from previous mass unit to M_sun/h
    :type MASS_UNIT: float
    :param suffix: either '_dm_' or '_gx_' or '' (latter for CosmicProfsDirect)
    :type suffix: string
    """
    return

def getGlobalTHist(VIZ_DEST, SNAP, start_time, obj_masses, obj_centers, d, q, s, major_full, HIST_NB_BINS, MASS_UNIT, suffix = '_'):
    """ Plot triaxiality T histogram
    
    :param VIZ_DEST: visualisation folder destination
    :type VIZ_DEST: string
    :param SNAP: e.g. '024'
    :type SNAP: string
    :param start_time: time of start of shape analysis
    :type start_time: float
    :param obj_masses: masses of objects in M_sun/h
    :type obj_masses: (N,) floats
    :param obj_centers: centers of objects, each coordinate in Mpc/h
    :type obj_centers: (N,3) floats
    :param d: ellipsoidal radii at which shape profiles have been calculated
    :type d: (N, D_BINS+1) floats
    :param q: q-values
    :type q: (N, D_BINS+1) floats
    :param s: s-values
    :type s: (N, D_BINS+1) floats
    :param major_full: major axes at each radii
    :type major_full: (N, D_BINS+1, 3) floats
    :param HIST_NB_BINS: Number of histogram bins
    :type HIST_NB_BINS: int
    :param MASS_UNIT: conversion factor from previous mass unit to M_sun/h
    :type MASS_UNIT: float
    :param suffix: either '_dm_' or '_gx_' or '' (latter for CosmicProfsDirect)
    :type suffix: string
    """
    return

def getGlobalEpsHist(idx_cat, xyz, masses, L_BOX, CENTER, VIZ_DEST, SNAP, suffix = '_', HIST_NB_BINS = 11):
    """ Plot ellipticity histogram
    
    :param idx_cat: catalogue of objects (objs/gxs)
    :type idx_cat: list of lists of ints
    :param xyz: coordinates of particles of type 1 or type 4
    :type xyz: (N^3x3) floats
    :param masses: masses of particles of type 1 or type 4, internal units
    :type masses: (N^3x1) floats
    :param L_BOX: simulation box side length
    :type L_BOX: float, units: Mpc/h
    :param CENTER: shape quantities will be calculated with respect to CENTER = 'mode' (point of highest density)
            or 'com' (center of mass) of each halo
    :type CENTER: str
    :param VIZ_DEST: visualisation folder destination
    :type VIZ_DEST: string
    :param SNAP: e.g. '024'
    :type SNAP: string
    :param suffix: either '_dm_' or '_gx_' or '' (latter for CosmicProfsDirect)
    :type suffix: string
    :param HIST_NB_BINS: Number of histogram bins
    :type HIST_NB_BINS: int"""
    
    return
        
def getLocalEpsHist(idx_cat, xyz, masses, r200, L_BOX, CENTER, VIZ_DEST, SNAP, frac_r200, suffix = '_', HIST_NB_BINS = 11):
    """ Plot ellipticity histogram
    
    :param idx_cat: catalogue of objects (objs/gxs)
    :type idx_cat: list of lists of ints
    :param xyz: coordinates of particles of type 1 or type 4, in Mpc/h
    :type xyz: (N^3x3) floats
    :param masses: masses of particles of type 1 or type 4, internal units
    :type masses: (N^3x1) floats
    :param r200: R_200 radii of the parent halos
    :type r200: (N1,) floats
    :param L_BOX: simulation box side length
    :type L_BOX: float, units: Mpc/h
    :param CENTER: shape quantities will be calculated with respect to CENTER = 'mode' (point of highest density)
            or 'com' (center of mass) of each halo
    :type CENTER: str
    :param VIZ_DEST: visualisation folder destination
    :type VIZ_DEST: string
    :param SNAP: e.g. '024'
    :type SNAP: string
    :param frac_r200: depth of objects to plot triaxiality, in units of R200
    :type frac_r200: float
    :param suffix: either '_dm_' or '_gx_' or '' (latter for CosmicProfsDirect)
    :type suffix: string
    :param HIST_NB_BINS: Number of histogram bins
    :type HIST_NB_BINS: int"""
    
    return