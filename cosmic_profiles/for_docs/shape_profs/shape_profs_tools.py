#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def getEpsilon(cat, xyz, masses, L_BOX, angle=0.0):
    """ Calculate the complex ellipticity (z-projected)
    
    It is obtained from the shape tensor = centred (wrt mode) second mass moment tensor
    
    :param cat: catalogue of objects (halos/gxs)
    :type cat: list of lists of ints
    :param xyz: coordinates of particles of type 1 or type 4
    :type xyz: (N^3x3) floats
    :param masses: masses of particles of type 1 or type 4
    :type masses: (N^3x1) floats
    :param L_BOX: simulation box side length
    :type L_BOX: float, units: Mpc/h
    :param angle: rotation of objects around z-axis before ellipticity is calculated (z-projected)
    :type angle: float
    :return: complex ellipticity
    :rtype: complex scalar
    """
    return

def getShape(R, d, param_interest, ERROR_METHOD, D_LOGSTART, D_LOGEND, D_BINS):
    """ Get average profile for param_interest (which is defined at all values of d)
    at all elliptical radii R
    
    :param R: elliptical radii of interest
    :type R: (N,) floats
    :param d: param_interest is defined at all elliptical radii d
    :type d: (N2,) floats
    :param param_interest: the quantity of interest defined at all elliptical radii d
    :type param_interest: (N2,) floats
    :return: mean, err_low, err_high
    :rtype: float, float, float"""
    return

def getShapeMs(R, d, idx_groups, group, param_interest, ERROR_METHOD, D_LOGSTART, D_LOGEND, D_BINS):
    """ Similar to getShape, but with mass-splitting"""
    return

def getShapeProfiles(VIZ_DEST, SNAP, D_LOGSTART, D_LOGEND, D_BINS, start_time, obj_masses, obj_centers, d, q, s, major_full, MASS_UNIT=1e10, suffix = '_'):
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
    :param MASS_UNIT: conversion factor from previous mass unit to M_sun/h
    :type MASS_UNIT: float
    :param suffix: either '_dm_' or '_gx_' or '' (latter for CosmicProfilesDirect)
    :type suffix: string"""
    
    return

def getLocalTHisto(CAT_DEST, VIZ_DEST, SNAP, D_LOGSTART, D_LOGEND, D_BINS, start_time, obj_masses, obj_centers, d, q, s, major_full, HIST_NB_BINS, MASS_UNIT=1e10, suffix = '_', inner = False):
    """ Plot triaxiality T histogram
    
    :param CAT_DEST: catalogue destination
    :type CAT_DEST: string
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
    :param suffix: either '_dm_' or '_gx_' or '' (latter for CosmicProfilesDirect)
    :type suffix: string
    :param inner: whether T histogram is to be plotted for R200 (False) or at R200*0.15 (True, e.g. 
        Milky Way radius is about R200*0.15)
    :type inner: boolean
    """
    return

def getGlobalEpsHisto(cat, xyz, masses, L_BOX, VIZ_DEST, SNAP, suffix = '_', HIST_NB_BINS = 11):
    """ Plot ellipticity histogram
    
    :param cat: catalogue of objects (objs/gxs)
    :type cat: list of lists of ints
    :param xyz: coordinates of particles of type 1 or type 4
    :type xyz: (N^3x3) floats
    :param masses: masses of particles of type 1 or type 4
    :type masses: (N^3x1) floats
    :param L_BOX: simulation box side length
    :type L_BOX: float, units: Mpc/h
    :param VIZ_DEST: visualisation folder destination
    :type VIZ_DEST: string
    :param SNAP: e.g. '024'
    :type SNAP: string
    :param suffix: either '_dm_' or '_gx_' or '' (latter for CosmicProfilesDirect)
    :type suffix: string
    :param HIST_NB_BINS: Number of histogram bins
    :type HIST_NB_BINS: int"""
    
    return
