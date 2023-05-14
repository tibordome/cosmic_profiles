#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def getEpsilon(idx_cat, obj_size, xyz, masses, L_BOX, CENTER, angle=0.0):
    """ Calculate the complex ellipticity (z-projected)
    
    It is obtained from the shape tensor = centred (wrt mode) second mass moment tensor
    
    :param idx_cat: contains indices of particles belonging to an object
    :type idx_cat: (N3) integers
    :param obj_size: indicates how many particles are in each object
    :type obj_size: (N1,) integers
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

def getShape(d, param_interest, ERROR_METHOD, r_over_r200, r200):
    """ Get average profile for param_interest (which is defined at all values of d)
    at all ellipsoidal radii Rs
    
    :param d: param_interest is defined at all ellipsoidal radii d
    :type d: (N1,N2) floats
    :param param_interest: the quantity of interest defined at all ellipsoidal radii d
    :type param_interest: (N1,N2) floats
    :param ERROR_METHOD: mean (if ERROR_METHOD == "bootstrap" or "SEM") or median
        (if ERROR_METHOD == "median_quantile") and the +- 1 sigma error attached
    :type ERROR_METHOD: string
    :param r_over_r200: normalized radii at which shape profiles are estimated
    :type r_over_r200: (N2,) floats
    :param r200: each entry gives the R_200 radius of the parent halo in Mpc/h (internal length units)
    :type r200: (N1,) floats
    :return: mean/median, err_low, err_high
    :rtype: float, float, float"""
    return

def getShapeMs(d, idx_groups, group, param_interest, ERROR_METHOD, r_over_r200, r200):
    """ Similar to getShape, but with mass-splitting"""
    return

def getShapeProfs(VIZ_DEST, SNAP, r_over_r200, r200, start_time, obj_masses, obj_centers, d, q, s, major_full, nb_bins, MASS_UNIT=1e10, suffix = '_'):
    """
    Create a series of plots to analyze object shapes
    
    Plot intertial tensor axis ratios, triaxialities and ellipticity histograms.
    
    :param VIZ_DEST: visualisation folder destination
    :type VIZ_DEST: string
    :param SNAP: e.g. '024'
    :type SNAP: string
    :param r_over_r200: normalized radii at which shape profiles are estimated
    :type r_over_r200: (D_BINS+1,) floats
    :param r200: each entry gives the R_200 radius of the parent halo in Mpc/h (internal length units)
    :type r200: (N,) floats
    :param start_time: time of start of shape analysis
    :type start_time: float
    :param obj_masses: total mass of objects, in 10^10*M_sun/h
    :type obj_masses: (N,) floats
    :param obj_centers: positions of centers of objects, in Mpc/h
    :type obj_centers: (N,3) floats
    :param d: param_interest is defined at all ellipsoidal radii d
    :type d: (N, D_BINS+1) floats
    :param q: intermediate-to-major axis ratios
    :type q: (N, D_BINS+1) floats
    :param s: minor-to-major axis ratios
    :type s: (N, D_BINS+1) floats
    :param major_full: major axis vectors
    :type major_full: (N,D_BINS+1,3) floats
    :param nb_bins: Number of mass bins to plot density profiles for
    :type nb_bins: int
    :param MASS_UNIT: conversion factor from previous mass unit to M_sun/h
    :type MASS_UNIT: float
    :param suffix: either '_dm_' or '_gx_' or '' (latter for DensShapeProfs)
    :type suffix: string"""
    
    return

def getLocalTHist(VIZ_DEST, SNAP, r_over_r200, r200, start_time, obj_masses, obj_centers, d, q, s, major_full, HIST_NB_BINS, frac_r200, MASS_UNIT, suffix = '_'):
    """ Plot triaxiality T histogram
    
    :param VIZ_DEST: visualisation folder destination
    :type VIZ_DEST: string
    :param SNAP: e.g. '024'
    :type SNAP: string
    :param r_over_r200: normalized radii at which shape profiles are estimated
    :type r_over_r200: (D_BINS+1,) floats
    :param r200: each entry gives the R_200 radius of the parent halo in Mpc/h (internal length units)
    :type r200: (N,) floats
    :param start_time: time of start of shape analysis
    :type start_time: float
    :param obj_masses: masses of objects, in 10^10*M_sun/h
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
    :param suffix: either '_dm_' or '_gx_' or '' (latter for DensShapeProfs)
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
    :param obj_masses: masses of objects in 10^10*M_sun/h
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
    :param suffix: either '_dm_' or '_gx_' or '' (latter for DensShapeProfs)
    :type suffix: string
    """    
    return

def getGlobalEpsHist(xyz, masses, idx_cat, obj_size, L_BOX, CENTER, VIZ_DEST, SNAP, suffix = '_', HIST_NB_BINS = 11):
    """ Plot ellipticity histogram
    
    :param xyz: coordinates of particles of type 1 or type 4, in Mpc/h
    :type xyz: (N^3x3) floats
    :param masses: masses of particles of type 1 or type 4, in 10^10*M_sun/h
    :type masses: (N^3x1) floats
    :param idx_cat: contains indices of particles belonging to an object
    :type idx_cat: (N3) integers
    :param obj_size: indicates how many particles are in each object
    :type obj_size: (N1,) integers
    :param L_BOX: simulation box side length
    :type L_BOX: float, units: Mpc/h
    :param CENTER: shape quantities will be calculated with respect to CENTER = 'mode' (point of highest density)
            or 'com' (center of mass) of each halo
    :type CENTER: str
    :param VIZ_DEST: visualisation folder destination
    :type VIZ_DEST: string
    :param SNAP: e.g. '024'
    :type SNAP: string
    :param suffix: either '_dm_' or '_gx_' or '' (latter for DensShapeProfs)
    :type suffix: string
    :param HIST_NB_BINS: Number of histogram bins
    :type HIST_NB_BINS: int"""
    
    return

def getLocalEpsHist(xyz, masses, r200, idx_cat, obj_size, L_BOX, CENTER, VIZ_DEST, SNAP, frac_r200, suffix = '_', HIST_NB_BINS = 11):
    """ Plot ellipticity histogram
    
    :param xyz: coordinates of particles of type 1 or type 4, in Mpc/h
    :type xyz: (N^3x3) floats
    :param masses: masses of particles of type 1 or type 4, in 10^10*M_sun/h
    :type masses: (N^3x1) floats
    :param r200: R_200 radii of the parent halos
    :type r200: (N1,) floats
    :param idx_cat: contains indices of particles belonging to an object
    :type idx_cat: (N3) integers
    :param obj_size: indicates how many particles are in each object
    :type obj_size: (N1,) integers
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
    :param suffix: either '_dm_' or '_gx_' or '' (latter for DensShapeProfs)
    :type suffix: string
    :param HIST_NB_BINS: Number of histogram bins
    :type HIST_NB_BINS: int"""
    
    return