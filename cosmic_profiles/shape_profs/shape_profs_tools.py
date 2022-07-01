#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 17:39:15 2022
"""

from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
from cosmic_profiles.common.python_routines import respectPBCNoRef, findMode, print_status, eTo10
from cosmic_profiles.common.cosmo_tools import M_split, getMeanOrMedianAndError
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
    if rank == 0:
        eps = []
        rot_matrix = R.from_rotvec(angle * np.array([0, 0, 1])).as_matrix()
        for obj in cat:
            if obj != []:
                xyz_ = respectPBCNoRef(xyz[obj], L_BOX)
                masses_ = masses[obj]
                mode = findMode(xyz_, masses_, max((max(xyz_[:,0])-min(xyz_[:,0]), max(xyz_[:,1])-min(xyz_[:,1]), max(xyz_[:,2])-min(xyz_[:,2]))))
                xyz_new = np.zeros((xyz_.shape[0],3))
                for i in range(xyz_new.shape[0]):
                    xyz_new[i] = np.dot(rot_matrix, xyz_[i]-mode)
                shape_tensor = np.sum((masses_)[:,np.newaxis,np.newaxis]*(np.matmul(xyz_new[:,:,np.newaxis],xyz_new[:,np.newaxis,:])),axis=0)/np.sum(masses_)
                qxx = shape_tensor[0,0]
                qyy = shape_tensor[1,1]
                qxy = shape_tensor[0,1]
                eps.append((qxx-qyy)/(qxx+qyy) + complex(0,1)*2*qxy/(qxx+qyy))
        eps = np.array(eps)
        return eps
    else:
        return None

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
    y = [[] for i in range(D_BINS+1)]
    for obj in range(param_interest.shape[0]):
        for rad in range(D_BINS+1):
            closest_idx = (np.abs(R - d[obj,rad]/d[obj,-int(D_LOGEND/((D_LOGEND-D_LOGSTART)/D_BINS))-1])).argmin() # Determine which point in R is closest
            if np.isnan(param_interest[obj][rad]) or np.log10(d[obj][rad]/d[obj,-int(D_LOGEND/((D_LOGEND-D_LOGSTART)/D_BINS))-1]) > D_LOGEND:
                continue
            else:
                y[closest_idx].append(param_interest[obj][rad])
    mean, err_low, err_high = getMeanOrMedianAndError(y, ERROR_METHOD)
    return mean, err_low, err_high

def getShapeMs(R, d, idx_groups, group, param_interest, ERROR_METHOD, D_LOGSTART, D_LOGEND, D_BINS):
    """ Similar to getShape, but with mass-splitting"""
    y = [[] for i in range(D_BINS+1)]
    for obj in idx_groups[group]:
        for rad in range(D_BINS+1):
            closest_idx = (np.abs(R - d[obj,rad]/d[obj,-int(D_LOGEND/((D_LOGEND-D_LOGSTART)/D_BINS))-1])).argmin() # Determine which point in R is closest
            if np.isnan(param_interest[obj][rad]) or np.log10(d[obj][rad]/d[obj,-int(D_LOGEND/((D_LOGEND-D_LOGSTART)/D_BINS))-1]) > D_LOGEND:
                continue
            else:
                y[closest_idx].append(param_interest[obj][rad])
    mean, err_low, err_high = getMeanOrMedianAndError(y, "median_quantile")
    return mean, err_low, err_high

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
    
    print_status(rank,start_time,'Starting getShapeProfiles() with snap {0}'.format(SNAP))
    
    if rank == 0:
        print_status(rank, start_time, "The number of objects considered is {0}".format(d.shape[0]))
        
        # Mass splitting
        max_min_m, obj_m_groups, obj_center_groups, idx_groups = M_split(MASS_UNIT*obj_masses, obj_centers, start_time)
            
        # Maximal elliptical radii
        R = np.logspace(D_LOGSTART,D_LOGEND,D_BINS+1)
        ERROR_METHOD = "median_quantile"
        
        # Q
        plt.figure()
        mean_median, err_low, err_high = getShape(R, d, q, ERROR_METHOD, D_LOGSTART, D_LOGEND, D_BINS)
        plt.semilogx(R, mean_median)
        plt.fill_between(R, mean_median-err_low, mean_median+err_high, edgecolor='g', alpha = 0.5, label = 'All objects')
        # Formatting
        plt.xlabel(r"$r/R_{200}$")
        plt.ylabel(r"q")
        plt.ylim(0.0, 1.0)
        plt.savefig("{0}/q{1}{2}.pdf".format(VIZ_DEST, suffix, SNAP), bbox_inches="tight")
        
        # S
        plt.figure()
        mean_median, err_low, err_high = getShape(R, d, s, ERROR_METHOD, D_LOGSTART, D_LOGEND, D_BINS)
        plt.semilogx(R, mean_median)
        plt.fill_between(R, mean_median-err_low, mean_median+err_high, edgecolor='g', alpha = 0.5, label = 'All objects')
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
        mean_median, err_low, err_high = getShape(R, d, T, ERROR_METHOD, D_LOGSTART, D_LOGEND, D_BINS)
        plt.semilogx(R, mean_median)
        plt.fill_between(R, mean_median-err_low, mean_median+err_high, edgecolor='g', alpha = 0.5, label = 'All objects')
        
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
            mean_median, err_low, err_high = getShapeMs(R, d, idx_groups, group, q, ERROR_METHOD, D_LOGSTART, D_LOGEND, D_BINS)
            if len(idx_groups[group]) != 0:
                plt.semilogx(R, mean_median)
                plt.fill_between(R, mean_median-err_low, mean_median+err_high, label = r"$M: {0} - {1} \ M_{{\odot}}/h$".format(eTo10("{:.2E}".format(max_min_m[group])), eTo10("{:.2E}".format(max_min_m[group+1]))), alpha = 0.5)
            plt.legend(loc="upper right", fontsize="x-small")
            plt.xlabel(r"$r/R_{200}$")
            plt.ylabel(r"q")
            plt.ylim(0.0, 1.0)
            plt.savefig("{}/qM{:.2f}{}{}.pdf".format(VIZ_DEST, np.float32(np.log10(max_min_m[group])), suffix, SNAP), bbox_inches="tight")
        
        # S: M-splitting
        for group in range(len(obj_m_groups)):
            plt.figure()
            mean_median, err_low, err_high = getShapeMs(R, d, idx_groups, group, s, ERROR_METHOD, D_LOGSTART, D_LOGEND, D_BINS)
            if len(idx_groups[group]) != 0:
                plt.semilogx(R, mean_median)
                plt.fill_between(R, mean_median-err_low, mean_median+err_high, label = r"$M: {0} - {1} \ M_{{\odot}}/h$".format(eTo10("{:.2E}".format(max_min_m[group])), eTo10("{:.2E}".format(max_min_m[group+1]))), alpha = 0.5)
            plt.legend(loc="upper right", fontsize="x-small")
            plt.xlabel(r"$r/R_{200}$")
            plt.ylabel(r"s")
            plt.ylim(0.0, 1.0)
            plt.savefig("{}/sM{:.2f}{}{}.pdf".format(VIZ_DEST, np.float32(np.log10(max_min_m[group])), suffix, SNAP), bbox_inches="tight")
        
        # T: M-splitting
        for group in range(len(obj_m_groups)):
            plt.figure()
            mean_median, err_low, err_high = getShapeMs(R, d, idx_groups, group, T, ERROR_METHOD, D_LOGSTART, D_LOGEND, D_BINS)
            if len(idx_groups[group]) != 0:
                plt.semilogx(R, mean_median)
                plt.fill_between(R, mean_median-err_low, mean_median+err_high, label = r"$M: {0} - {1} \ M_{{\odot}}/h$".format(eTo10("{:.2E}".format(max_min_m[group])), eTo10("{:.2E}".format(max_min_m[group+1]))), alpha = 0.5)
            plt.axhline(2/3, label=r"$T$ > 2/3: prolate", linestyle='--', color = "y")
            plt.legend(loc="upper right", fontsize="x-small")            
            plt.xlabel(r"$r/R_{200}$")
            plt.ylabel(r"T")
            plt.ylim(0.0, 1.0)
            plt.savefig("{}/TM{:.2f}{}{}.pdf".format(VIZ_DEST, np.float32(np.log10(max_min_m[group])), suffix, SNAP), bbox_inches="tight")

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
    print_status(rank, start_time, "Starting getLocalTHisto(). The number of objects considered is {0}".format(d.shape[0]))
    
    if rank == 0:
        idx = np.zeros((d.shape[0],), dtype = np.int32)
        for obj in range(idx.shape[0]):
            if inner == True:
                idx[obj] = np.argmin(abs(d[obj] - d[obj,-int(D_LOGEND/((D_LOGEND-D_LOGSTART)/D_BINS))-1]*0.15))
            else:
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
        plt.savefig("{0}/TCount{1}{2}.pdf".format(VIZ_DEST, suffix, SNAP), bbox_inches="tight")
        
        t = t[np.logical_not(np.isnan(t))]
        print_status(rank, start_time, "In degrees: The average T value for the objects is {0} and the standard deviation (assuming T is Gaussian distributed) is {1}".format(round(np.average(t),2), round(np.std(t),2)))
     
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
    
    if rank == 0:
        eps = getEpsilon(cat, xyz, masses, L_BOX)
        plt.figure()
        n, bins, patches = plt.hist(x=abs(eps), bins = np.linspace(0, 1, HIST_NB_BINS), alpha=0.7, density=True)
        plt.xlabel(r"$\epsilon$")
        plt.ylabel('Normalized Bin Count')
        plt.grid(axis='y', alpha=0.75)
        plt.xlim(0.0, 1.0)
        plt.savefig("{0}/EpsCount{1}{2}.pdf".format(VIZ_DEST, suffix, SNAP), bbox_inches="tight")
