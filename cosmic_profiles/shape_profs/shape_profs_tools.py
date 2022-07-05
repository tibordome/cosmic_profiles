#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
from cosmic_profiles.common.python_routines import respectPBCNoRef, calcCoM, calcMode, print_status, eTo10, getCatWithinFracR200
from cosmic_profiles.common.cosmo_tools import M_split, getMeanOrMedianAndError
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
    if rank == 0:
        eps = []
        rot_matrix = R.from_rotvec(angle * np.array([0, 0, 1])).as_matrix()
        for obj in idx_cat:
            if obj != []:
                xyz_ = respectPBCNoRef(xyz[obj], L_BOX)
                if CENTER == 'mode':
                    center = calcMode(xyz_, masses[obj], max((max(xyz_[:,0])-min(xyz_[:,0]), max(xyz_[:,1])-min(xyz_[:,1]), max(xyz_[:,2])-min(xyz_[:,2]))))
                else:
                    center = calcCoM(xyz_, masses[obj])
                masses_ = masses[obj]
                xyz_new = np.zeros((xyz_.shape[0],3))
                for i in range(xyz_new.shape[0]):
                    xyz_new[i] = np.dot(rot_matrix, xyz_[i]-center)
                shape_tensor = np.sum((masses_)[:,np.newaxis,np.newaxis]*(np.matmul(xyz_new[:,:,np.newaxis],xyz_new[:,np.newaxis,:])),axis=0)/np.sum(masses_)
                qxx = shape_tensor[0,0]
                qyy = shape_tensor[1,1]
                qxy = shape_tensor[0,1]
                eps.append((qxx-qyy)/(qxx+qyy) + complex(0,1)*2*qxy/(qxx+qyy))
        eps = np.array(eps)
        return eps
    else:
        return None

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
    y = [[] for i in range(D_BINS+1)]
    for obj in range(param_interest.shape[0]):
        for rad in range(D_BINS+1):
            closest_idx = (np.abs(Rs - d[obj,rad]/d[obj,-int(D_LOGEND/((D_LOGEND-D_LOGSTART)/D_BINS))-1])).argmin() # Determine which point in Rs is closest
            if np.isnan(param_interest[obj][rad]) or np.log10(d[obj][rad]/d[obj,-int(D_LOGEND/((D_LOGEND-D_LOGSTART)/D_BINS))-1]) > D_LOGEND:
                continue
            else:
                y[closest_idx].append(param_interest[obj][rad])
    mean, err_low, err_high = getMeanOrMedianAndError(y, ERROR_METHOD)
    return mean, err_low, err_high

def getShapeMs(Rs, d, idx_groups, group, param_interest, ERROR_METHOD, D_LOGSTART, D_LOGEND, D_BINS):
    """ Similar to getShape, but with mass-splitting"""
    y = [[] for i in range(D_BINS+1)]
    for obj in idx_groups[group]:
        for rad in range(D_BINS+1):
            closest_idx = (np.abs(Rs - d[obj,rad]/d[obj,-int(D_LOGEND/((D_LOGEND-D_LOGSTART)/D_BINS))-1])).argmin() # Determine which point in Rs is closest
            if np.isnan(param_interest[obj][rad]) or np.log10(d[obj][rad]/d[obj,-int(D_LOGEND/((D_LOGEND-D_LOGSTART)/D_BINS))-1]) > D_LOGEND:
                continue
            else:
                y[closest_idx].append(param_interest[obj][rad])
    mean, err_low, err_high = getMeanOrMedianAndError(y, "median_quantile")
    return mean, err_low, err_high

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
    
    print_status(rank,start_time,'Starting getShapeProfs() with snap {0}'.format(SNAP))
    
    if rank == 0:
        print_status(rank, start_time, "The number of objects considered is {0}".format(d.shape[0]))
        
        # Mass splitting
        max_min_m, obj_m_groups, obj_center_groups, idx_groups = M_split(MASS_UNIT*obj_masses, obj_centers, start_time)
            
        # Maximal elliptical radii
        Rs = np.logspace(D_LOGSTART,D_LOGEND,D_BINS+1)
        ERROR_METHOD = "median_quantile"
        
        # Q
        plt.figure()
        mean_median, err_low, err_high = getShape(Rs, d, q, ERROR_METHOD, D_LOGSTART, D_LOGEND, D_BINS)
        plt.semilogx(Rs, mean_median)
        plt.fill_between(Rs, mean_median-err_low, mean_median+err_high, edgecolor='g', alpha = 0.5, label = 'All objects')
        # Formatting
        plt.xlabel(r"$r/R_{200}$")
        plt.ylabel(r"q")
        plt.ylim(0.0, 1.0)
        plt.savefig("{0}/q{1}{2}.pdf".format(VIZ_DEST, suffix, SNAP), bbox_inches="tight")
        
        # S
        plt.figure()
        mean_median, err_low, err_high = getShape(Rs, d, s, ERROR_METHOD, D_LOGSTART, D_LOGEND, D_BINS)
        plt.semilogx(Rs, mean_median)
        plt.fill_between(Rs, mean_median-err_low, mean_median+err_high, edgecolor='g', alpha = 0.5, label = 'All objects')
        # Formatting
        plt.xlabel(r"$r/R_{200}$")
        plt.ylabel(r"s")
        plt.ylim(0.0, 1.0)
        plt.savefig("{0}/s{1}{2}.pdf".format(VIZ_DEST, suffix, SNAP), bbox_inches="tight")
        
        # T
        plt.figure()
        if q.ndim == 2:
            T = np.zeros((q.shape[0], q.shape[1]))
            for obj in range(q.shape[0]):
                T[obj] = (1-q[obj]**2)/(1-s[obj]**2) # Triaxiality
        else:
            T = np.empty(0)
        mean_median, err_low, err_high = getShape(Rs, d, T, ERROR_METHOD, D_LOGSTART, D_LOGEND, D_BINS)
        plt.semilogx(Rs, mean_median)
        plt.fill_between(Rs, mean_median-err_low, mean_median+err_high, edgecolor='g', alpha = 0.5, label = 'All objects')
        
        # Formatting
        plt.xlabel(r"$r/R_{200}$")
        plt.ylabel(r"T")
        plt.axhline(2/3, label=r"$T$ > 2/3: prolate", linestyle='--', color = "y")
        plt.legend(loc="upper right", fontsize="x-small")
        plt.ylim(0.0, 1.0)
        plt.savefig("{0}/T{1}{2}.pdf".format(VIZ_DEST, suffix, SNAP), bbox_inches="tight")
        
        # Q: M-splitting
        for group in range(len(obj_m_groups)):
            plt.figure()
            mean_median, err_low, err_high = getShapeMs(Rs, d, idx_groups, group, q, ERROR_METHOD, D_LOGSTART, D_LOGEND, D_BINS)
            if len(idx_groups[group]) != 0:
                plt.semilogx(Rs, mean_median)
                plt.fill_between(Rs, mean_median-err_low, mean_median+err_high, label = r"$M: {0} - {1} \ M_{{\odot}}/h$".format(eTo10("{:.2E}".format(max_min_m[group])), eTo10("{:.2E}".format(max_min_m[group+1]))), alpha = 0.5)
            plt.legend(loc="upper right", fontsize="x-small")
            plt.xlabel(r"$r/R_{200}$")
            plt.ylabel(r"q")
            plt.ylim(0.0, 1.0)
            plt.savefig("{}/qM{:.2f}{}{}.pdf".format(VIZ_DEST, np.float32(np.log10(max_min_m[group])), suffix, SNAP), bbox_inches="tight")
        
        # S: M-splitting
        for group in range(len(obj_m_groups)):
            plt.figure()
            mean_median, err_low, err_high = getShapeMs(Rs, d, idx_groups, group, s, ERROR_METHOD, D_LOGSTART, D_LOGEND, D_BINS)
            if len(idx_groups[group]) != 0:
                plt.semilogx(Rs, mean_median)
                plt.fill_between(Rs, mean_median-err_low, mean_median+err_high, label = r"$M: {0} - {1} \ M_{{\odot}}/h$".format(eTo10("{:.2E}".format(max_min_m[group])), eTo10("{:.2E}".format(max_min_m[group+1]))), alpha = 0.5)
            plt.legend(loc="upper right", fontsize="x-small")
            plt.xlabel(r"$r/R_{200}$")
            plt.ylabel(r"s")
            plt.ylim(0.0, 1.0)
            plt.savefig("{}/sM{:.2f}{}{}.pdf".format(VIZ_DEST, np.float32(np.log10(max_min_m[group])), suffix, SNAP), bbox_inches="tight")
        
        # T: M-splitting
        for group in range(len(obj_m_groups)):
            plt.figure()
            mean_median, err_low, err_high = getShapeMs(Rs, d, idx_groups, group, T, ERROR_METHOD, D_LOGSTART, D_LOGEND, D_BINS)
            if len(idx_groups[group]) != 0:
                plt.semilogx(Rs, mean_median)
                plt.fill_between(Rs, mean_median-err_low, mean_median+err_high, label = r"$M: {0} - {1} \ M_{{\odot}}/h$".format(eTo10("{:.2E}".format(max_min_m[group])), eTo10("{:.2E}".format(max_min_m[group+1]))), alpha = 0.5)
            plt.axhline(2/3, label=r"$T$ > 2/3: prolate", linestyle='--', color = "y")
            plt.legend(loc="upper right", fontsize="x-small")            
            plt.xlabel(r"$r/R_{200}$")
            plt.ylabel(r"T")
            plt.ylim(0.0, 1.0)
            plt.savefig("{}/TM{:.2f}{}{}.pdf".format(VIZ_DEST, np.float32(np.log10(max_min_m[group])), suffix, SNAP), bbox_inches="tight")

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
    print_status(rank, start_time, "Starting getLocalTHisto(). The number of objects considered is {0}".format(d.shape[0]))
    
    if rank == 0:
        idx = np.zeros((d.shape[0],), dtype = np.int32)
        for obj in range(idx.shape[0]):
            idx[obj] = np.argmin(abs(d[obj] - d[obj,-int(D_LOGEND/((D_LOGEND-D_LOGSTART)/D_BINS))-1]*frac_r200))        
        
        t = np.zeros((d.shape[0],))
        for obj in range(d.shape[0]):
            t[obj] = (1-q[obj,idx[obj]]**2)/(1-s[obj,idx[obj]]**2) # Triaxiality
        t = np.nan_to_num(t)
        
        # T counting
        plt.figure()
        t[t == 0.] = np.nan
        n, bins, patches = plt.hist(x=t, bins = np.linspace(0, 1, HIST_NB_BINS), alpha=0.7, density=True)
        plt.axvline(1/3, label="oblate-triaxial transition", color = "g")
        plt.axvline(2/3, label="triaxial-prolate transition", color = "r")
        plt.xlabel(r"T")
        plt.ylabel('Normalized Bin Count')
        plt.grid(axis='y', alpha=0.75)
        plt.xlim(0.0, 1.0)
        plt.legend(loc="upper left", fontsize="x-small")
        plt.savefig("{0}/LocalTCount{1}{2}.pdf".format(VIZ_DEST, suffix, SNAP), bbox_inches="tight")
        
        t = t[np.logical_not(np.isnan(t))]
        print_status(rank, start_time, "In degrees: The average T value for the objects is {0} and the standard deviation (assuming T is Gaussian distributed) is {1}".format(round(np.average(t),2), round(np.std(t),2)))
     
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
    print_status(rank, start_time, "Starting getLocalTHisto(). The number of objects considered is {0}".format(d.shape[0]))
    
    if rank == 0:
        idx = np.array([np.int32(x) for x in list(np.ones((d.shape[0],))*(-1))])
        
        t = np.zeros((d.shape[0],))
        for obj in range(d.shape[0]):
            t[obj] = (1-q[obj,idx[obj]]**2)/(1-s[obj,idx[obj]]**2) # Triaxiality
        t = np.nan_to_num(t)
        
        # T counting
        plt.figure()
        t[t == 0.] = np.nan
        n, bins, patches = plt.hist(x=t, bins = np.linspace(0, 1, HIST_NB_BINS), alpha=0.7, density=True)
        plt.axvline(1/3, label="oblate-triaxial transition", color = "g")
        plt.axvline(2/3, label="triaxial-prolate transition", color = "r")
        plt.xlabel(r"T")
        plt.ylabel('Normalized Bin Count')
        plt.grid(axis='y', alpha=0.75)
        plt.xlim(0.0, 1.0)
        plt.legend(loc="upper left", fontsize="x-small")
        plt.savefig("{0}/GlobalTCount{1}{2}.pdf".format(VIZ_DEST, suffix, SNAP), bbox_inches="tight")
        
        t = t[np.logical_not(np.isnan(t))]
        print_status(rank, start_time, "In degrees: The average T value for the objects is {0} and the standard deviation (assuming T is Gaussian distributed) is {1}".format(round(np.average(t),2), round(np.std(t),2)))
     

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
    
    if rank == 0:
        eps = getEpsilon(idx_cat, xyz, masses, L_BOX, CENTER)
        plt.figure()
        n, bins, patches = plt.hist(x=abs(eps), bins = np.linspace(0, 1, HIST_NB_BINS), alpha=0.7, density=True)
        plt.xlabel(r"$\epsilon$")
        plt.ylabel('Normalized Bin Count')
        plt.grid(axis='y', alpha=0.75)
        plt.xlim(0.0, 1.0)
        plt.savefig("{0}/EpsCount{1}{2}.pdf".format(VIZ_DEST, suffix, SNAP), bbox_inches="tight")
        
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
    
    if rank == 0:
        # Update h_cat so that only particles within r200 are considered
        idx_cat_new = getCatWithinFracR200(idx_cat, xyz, L_BOX, CENTER, r200, frac_r200)
        
        # Direct fitting result prep, needed for both D and A
        eps = getEpsilon(idx_cat_new, xyz, masses, L_BOX, CENTER)
        plt.figure()
        n, bins, patches = plt.hist(x=abs(eps), bins = np.linspace(0, 1, HIST_NB_BINS), alpha=0.7, density=True)
        plt.xlabel(r"$\epsilon$")
        plt.ylabel('Normalized Bin Count')
        plt.grid(axis='y', alpha=0.75)
        plt.xlim(0.0, 1.0)
        plt.savefig("{0}/EpsCount{1}{2}.pdf".format(VIZ_DEST, suffix, SNAP), bbox_inches="tight")