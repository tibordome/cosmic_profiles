#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from cosmic_profiles.common.python_routines import print_status, eTo10
from cosmic_profiles.common.cosmo_tools import M_split
from pathos.multiprocessing import ProcessingPool as Pool
from functools import partial
import os
from cosmic_profiles.common.caching import np_cache_factory
from scipy import optimize
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def getEinastoProf(r, model_pars):
    """
    Get Einasto density profile at radius ``r``
    
    :param r: radius of interest
    :type r: float
    :param model_pars: model parameters
    :type model_pars: (n,) float array
    :return: profile value at ``r``
    :rtype: float"""
    rho_2, r_2, alpha = model_pars
    return rho_2*np.exp(-2/alpha*((r/r_2)**alpha-1))

def getAlphaBetaGammaProf(r, model_pars):
    """
    Get alpha-beta-gamma density profile at radius ``r``
    
    :param r: radius of interest
    :type r: float
    :param model_pars: model parameters
    :type model_pars: (n,) float array
    :return: profile value at ``r``
    :rtype: float"""
    rho_0, alpha, beta, gamma, r_s = model_pars
    return rho_0/((r/r_s)**gamma*(1+(r/r_s)**alpha)**((beta-gamma)/alpha))

def getNFWProf(r, model_pars):
    """
    Get NFW density profile at radius ``r``
    
    :param r: radius of interest
    :type r: float
    :param model_pars: model parameters
    :type model_pars: (n,) float array
    :return: profile value at ``r``
    :rtype: float"""
    rho_s, r_s = model_pars
    return rho_s/((r/r_s)*(1+r/r_s)**2)

def getHernquistProf(r, model_pars):
    """
    Get Hernquist density profile at radius ``r``
    
    :param r: radius of interest
    :type r: float
    :param model_pars: model parameters
    :type model_pars: (n,) float array
    :return: profile value at ``r``
    :rtype: float"""
    rho_s, r_s = model_pars
    return rho_s/((r/r_s)*(1+r/r_s)**3)

def getDensProfs(VIZ_DEST, SNAP, cat, r200s, dens_profs_fit, ROverR200_fit, dens_profs, ROverR200, obj_masses, obj_centers, method, start_time, MASS_UNIT, suffix = '_'):
    """
    Create a series of plots to analyze object shapes
    
    Plot intertial tensor axis ratios, triaxialities and ellipticity histograms.
    
    :param VIZ_DEST: visualisation folder destination
    :type VIZ_DEST: string
    :param SNAP: e.g. '024'
    :type SNAP: string
    :param cat: catalogue of objects (halos/gxs)
    :type cat: N2-long list of lists of ints, N2 > N
    :param r200s: catalogue of virial radii (of parent halos in case of gxs)
    :type r200s: N2-long float array
    :param dens_profs_fit: density profiles, defined at ``ROverR200``, in M_sun*h^2/(Mpc)**3
    :type dens_profs_fit: (N, r_res2) floats
    :param ROverR200_fit: normalized radii at which the mass-decomposed density
        profile fits shall be calculated
    :type ROverR200_fit: (r_res2,) floats
    :param dens_profs: density profiles, defined at ``ROverR200``, in M_sun*h^2/(Mpc)**3
    :type dens_profs: (N, r_res1) floats
    :param ROverR200: normalized radii at which ``dens_profs`` are defined
    :type ROverR200: (N, r_res1) floats
    :param obj_masses: masses of objects in M_sun/h
    :type obj_masses: (N,) floats
    :param obj_centers: centers of objects, each coordinate in Mpc/h
    :type obj_centers: (N,3) floats
    :param method: string describing density profile model assumed for fitting
    :type method: string, either `einasto`, `alpha_beta_gamma`, `hernquist`, `nfw`
    :param start_time: time of start of shape analysis
    :type start_time: float
    :param MASS_UNIT: conversion factor from previous mass unit to M_sun/h
    :type MASS_UNIT: float
    :param suffix: either '_dm_' or '_gx_' or '' (latter for CosmicProfsDirect)
    :type suffix: string"""
    
    print_status(rank,start_time,'Starting getDensProfs() with snap {0}'.format(SNAP))
    
    if rank == 0:
        print_status(rank, start_time, "The number of objects considered is {0}".format(obj_masses.shape[0]))
        
        # Mass splitting
        max_min_m, obj_m_groups, obj_center_groups, idx_groups = M_split(MASS_UNIT*obj_masses, obj_centers, start_time)
        
        obj_pass = np.int32(np.array([1 if x != [] else 0 for x in cat]))
        idxs_compr = np.zeros((len(cat),), dtype = np.int32)
        idxs_compr[obj_pass.nonzero()[0]] = np.arange(np.sum(obj_pass)) 
        prof_models = {'einasto': getEinastoProf, 'alpha_beta_gamma': getAlphaBetaGammaProf, 'nfw': getNFWProf, 'hernquist': getHernquistProf}
        model_name = {'einasto': 'Einasto', 'alpha_beta_gamma': r'\alpha \beta \gamma', 'nfw': 'NFW', 'hernquist': 'Hernquist'}
        
        # Average over all objects' density profiles
        if np.sum(obj_pass) > 0:
            dens_prof_ = dens_profs[idxs_compr[np.nonzero(obj_pass > 0)[0]]]
        else:
            dens_prof_ = np.zeros((0,ROverR200.shape[0]), dtype = np.float32)
        y = [list(dens_prof_[:,i]) for i in range(ROverR200.shape[0])]
        prof_median = np.array([np.median(z) if z != [] else np.nan for z in y])
        err_low = np.array([np.quantile(np.array(z), 0.25)/(np.sqrt(len(z))) if z != [] else np.nan for z in y])
        err_high = np.array([np.quantile(np.array(z), 0.75)/(np.sqrt(len(z))) if z != [] else np.nan for z in y])
        r200 = np.average(r200s[np.arange(r200s.shape[0])[obj_pass.nonzero()[0]]])
        # Prepare median for fitting
        if np.sum(obj_pass) > 0:
            dens_prof_fit = dens_profs_fit[idxs_compr[np.nonzero(obj_pass > 0)[0]]]
        else:
            dens_prof_fit = np.zeros((0,ROverR200_fit.shape[0]), dtype = np.float32)
        y = [list(dens_prof_fit[:,i]) for i in range(ROverR200_fit.shape[0])]
        prof_median_fit = np.array([np.median(z) if z != [] else np.nan for z in y])
        best_fit, obj_nb = fitDensProf(ROverR200_fit, method, (prof_median_fit, r200, 0)) # Fit median
        # Plotting
        plt.figure()
        plt.loglog(ROverR200_fit, prof_models[method](ROverR200_fit*np.average(r200s[np.arange(r200s.shape[0])[obj_pass.nonzero()[0]]]), best_fit), 'o--', color = 'r', linewidth=2, markersize=4, label=r'${}$-profile fit'.format(model_name[method]))
        plt.loglog(ROverR200, prof_median, color = 'blue')
        plt.fill_between(ROverR200, prof_median-err_low, prof_median+err_high, facecolor = 'blue', edgecolor='g', alpha = 0.5, label = r"All objects")
        plt.xlabel(r"$r/R_{200}$")
        plt.ylabel(r"$\rho$ [$h^2M_{{\odot}}$ / Mpc${{}}^3$]")
        plt.legend(loc="upper right", fontsize="x-small")
        plt.savefig("{}/RhoProf_{}.pdf".format(VIZ_DEST, SNAP), bbox_inches="tight")
        
        for group in range(len(obj_m_groups)):
            obj_pass_m = np.int32([1 if (obj_pass[i] == 1 and obj_masses[idxs_compr[i]]*MASS_UNIT > max_min_m[group] and obj_masses[idxs_compr[i]]*MASS_UNIT < max_min_m[group+1]) else 0 for i in range(len(cat))])
            # Find profile median
            y = [list(dens_prof_[idxs_compr[np.nonzero(obj_pass_m > 0)[0]],i]) for i in range(ROverR200.shape[0])]
            prof_median = np.array([np.median(z) if z != [] else np.nan for z in y])
            err_low = np.array([np.quantile(np.array(z), 0.25)/(np.sqrt(len(z))) if z != [] else np.nan for z in y])
            err_high = np.array([np.quantile(np.array(z), 0.75)/(np.sqrt(len(z))) if z != [] else np.nan for z in y])
            r200_m = np.average(r200s[np.arange(r200s.shape[0])[obj_pass_m.nonzero()[0]]])
            # Prepare median for fitting
            y = [list(dens_prof_fit[idxs_compr[np.nonzero(obj_pass_m > 0)[0]],i]) for i in range(ROverR200_fit.shape[0])]
            prof_median_fit = np.array([np.median(z) if z != [] else np.nan for z in y])
            best_fit_m, obj_nb = fitDensProf(ROverR200_fit, method, (prof_median_fit, r200_m, 0))
            # Plotting
            plt.figure()
            plt.loglog(ROverR200_fit, prof_models[method](ROverR200_fit*np.average(r200s[np.arange(r200s.shape[0])[obj_pass_m.nonzero()[0]]]), best_fit_m), 'o--', color = 'r', linewidth=2, markersize=4, label=r'${}$-profile fit'.format(model_name[method]))
            plt.loglog(ROverR200, prof_median, color = 'blue')
            plt.fill_between(ROverR200, prof_median-err_low, prof_median+err_high, facecolor = 'blue', edgecolor='g', alpha = 0.5, label = r"$M: {0} - {1} \ M_{{\odot}}/h$".format(eTo10("{:.2E}".format(max_min_m[group])), eTo10("{:.2E}".format(max_min_m[group+1]))))
            plt.xlabel(r"$r/R_{200}$")
            plt.ylabel(r"$\rho$ [$h^2M_{{\odot}}$ / Mpc${{}}^3$]")
            plt.legend(loc="upper right", fontsize="x-small")
            plt.savefig("{}/RhoProfM{:.2f}_{}.pdf".format(VIZ_DEST, np.float32(np.log10(max_min_m[group])), SNAP), bbox_inches="tight")
        return

def fitDensProf(ROverR200, method, median_r200_obj_nb):
    """
    Fit density profile according to model provided
    
    Note that ``median`` which is defined at ``ROverR200``
    must be in units of UnitMass/(Mpc/h)**3.
    
    :param ROverR200: normalized radii where ``median`` is defined and 
        fitting should be carried out
    :type ROverR200: (N,) floats
    :param method: string describing density profile model assumed for fitting
    :type method: string, either `einasto`, `alpha_beta_gamma`, `hernquist`, `nfw`
    :param median_r200_obj_nb: density profile (often a median of many profiles combined,
        in units of UNIT_MASS/(Mpc/h)**3), virial radius (of parent halo in case of gx) 
        and object number
    :type median_r200_obj_nb: tuple of (N,) floats, float, int
    :return best_fit, obj_nb: best-fit results, object number
    :rtype: (n,) floats, int"""
    
    median, r200, obj_nb = median_r200_obj_nb
    prof_models = {'einasto': getEinastoProf, 'alpha_beta_gamma': getAlphaBetaGammaProf, 'nfw': getNFWProf, 'hernquist': getHernquistProf}
    def toMinimize(model_pars, median, rbin_centers, method):
        psi_2 = np.sum(np.array([(np.log(median[i])-np.log(prof_models[method](rbin, model_pars)))**2/rbin_centers.shape[0] for i, rbin in enumerate(rbin_centers)]))
        return psi_2
    R_to_min = ROverR200*r200 # Mpc/h
    # Discard nan values in median
    R_to_min = R_to_min[~np.isnan(median)] # Note: np.isnan returns a boolean
    median = median[~np.isnan(median)]
    # Set initial guess and minimize scalar function
    try:
        if method == 'einasto':
            iguess = np.array([0.1, r200/5, 0.18]) # Note: alpha = 0.18 gives ~ NFW
            res = optimize.minimize(toMinimize, iguess, method = 'TNC', args = (median/np.average(median), R_to_min, method), bounds = [(1e-7, np.inf), (1e-5, np.inf), (-np.inf, np.inf)]) # Only hand over rescaled median!
            best_fit = res.x
        elif method == 'alpha_beta_gamma':
            iguess = np.array([0.1, 1.0, 1.0, 1.0, r200/5])
            res = optimize.minimize(toMinimize, iguess, method = 'TNC', args = (median/np.average(median), R_to_min, method), bounds = [(1e-7, np.inf), (1e-5, np.inf), (1e-5, np.inf), (1e-5, np.inf), (1e-5, np.inf)]) # Only hand over rescaled median!
            best_fit = res.x
        elif method == 'hernquist':
            iguess = np.array([0.1, r200/5])
            res = optimize.minimize(toMinimize, iguess, method = 'TNC', args = (median/np.average(median), R_to_min, method), bounds = [(1e-7, np.inf), (1e-5, np.inf)]) # Only hand over rescaled median!
            best_fit = res.x
        else:
            iguess = np.array([0.1, r200/5])
            res = optimize.minimize(toMinimize, iguess, method = 'TNC', args = (median/np.average(median), R_to_min, method), bounds = [(1e-7, np.inf), (1e-5, np.inf)]) # Only hand over rescaled median!
            best_fit = res.x
        best_fit[0] *= np.average(median)
    except ValueError: # For poor density profiles one might encounter "ValueError: `x0` violates bound constraints."
        best_fit = iguess*np.nan
    return best_fit, obj_nb

@np_cache_factory(3,0)
def fitDensProfHelper(dens_profs, ROverR200, r200s, method):
    """ Helper function to carry out density profile fitting
    
    :param dens_profs: array containing density profiles in units 
        of UnitMass/(Mpc/h)**3)
    :type dens_profs: (N,r_res) floats
    :param ROverR200: normalized radii where density profile is defined
        and fitting should be carried out
    :type ROverR200: (r_res,) floats
    :param r200s: R200 values of objects
    :type r200s: (N,) floats
    :param method: string describing density profile model assumed for fitting
    :type method: string, either `einasto`, `alpha_beta_gamma`, `hernquist`, `nfw`
    :return best_fits: best-fit results
    :rtype: (N,n) floats"""
    
    if rank == 0:
        if method == 'einasto':
            best_fits = np.zeros((dens_profs.shape[0], 3))
        elif method == 'alpha_beta_gamma':
            best_fits = np.zeros((dens_profs.shape[0], 5))
        else:
            best_fits = np.zeros((dens_profs.shape[0], 2))
        with Pool(processes=len(os.sched_getaffinity(0))) as pool:
            results = pool.map(partial(fitDensProf, ROverR200, method), [(dens_profs[obj_nb], r200s[obj_nb], obj_nb) for obj_nb in range(dens_profs.shape[0])])
        for result in results:
            x, obj_nb = tuple(result)
            best_fits[obj_nb] = x
        return best_fits
    else:
        return None