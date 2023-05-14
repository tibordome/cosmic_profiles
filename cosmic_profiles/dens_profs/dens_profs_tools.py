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
from cosmic_profiles.common import config
import inspect
import subprocess
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
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
    rho_s = model_pars['rho_s']
    alpha = model_pars['alpha']
    r_s = model_pars['r_s']
    return rho_s*np.exp(-2/alpha*((r/r_s)**alpha-1))

def getAlphaBetaGammaProf(r, model_pars):
    """
    Get alpha-beta-gamma density profile at radius ``r``
    
    :param r: radius of interest
    :type r: float
    :param model_pars: model parameters
    :type model_pars: (n,) float array
    :return: profile value at ``r``
    :rtype: float"""
    rho_s = model_pars['rho_s']
    alpha = model_pars['alpha']
    beta = model_pars['beta']
    gamma = model_pars['gamma']
    r_s = model_pars['r_s']
    return rho_s/((r/r_s)**gamma*(1+(r/r_s)**alpha)**((beta-gamma)/alpha))

def getNFWProf(r, model_pars):
    """
    Get NFW density profile at radius ``r``
    
    :param r: radius of interest
    :type r: float
    :param model_pars: model parameters
    :type model_pars: (n,) float array
    :return: profile value at ``r``
    :rtype: float"""
    rho_s = model_pars['rho_s']
    r_s = model_pars['r_s']
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
    rho_s = model_pars['rho_s']
    r_s = model_pars['r_s']
    return rho_s/((r/r_s)*(1+r/r_s)**3)

def drawDensProfs(VIZ_DEST, SNAP, r200s, dens_profs_fit, ROverR200_fit, dens_profs, ROverR200, obj_masses, obj_centers, method, nb_bins, start_time, MASS_UNIT, suffix = '_'):
    """
    Create a series of plots to analyze object shapes
    
    Plot intertial tensor axis ratios, triaxialities and ellipticity histograms.
    
    :param VIZ_DEST: visualisation folder destination
    :type VIZ_DEST: string
    :param SNAP: e.g. '024'
    :type SNAP: string
    :param r200s: catalogue of virial radii (of parent halos) in Mpc/h
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
    :param obj_masses: masses of objects in 10^10*M_sun/h
    :type obj_masses: (N,) floats
    :param obj_centers: centers of objects, each coordinate in Mpc/h
    :type obj_centers: (N,3) floats
    :param method: describes density profile model assumed for fitting, if parameter should be kept fixed during fitting then it needs to be provided, e.g. method['alpha'] = 0.18
    :type method: dictionary, method['profile'] is either `einasto`, `alpha_beta_gamma`, `hernquist`, `nfw`, minimum requirement
    :param nb_bins: Number of mass bins to plot density profiles for
    :type nb_bins: int
    :param start_time: time of start of shape analysis
    :type start_time: float
    :param MASS_UNIT: conversion factor from previous mass unit to M_sun/h, usually 10^10
    :type MASS_UNIT: float
    :param suffix: either '_dm_' or '_gx_' or '_' (latter for CosmicProfsDirect)
    :type suffix: string"""
    
    if rank == 0:
        
        def getModelParsDict(model_pars_arr, method):
            if method == 'einasto':
                rho_s, alpha, r_s = model_pars_arr
                model_pars_dict = {'rho_s': rho_s, 'alpha': alpha, 'r_s': r_s}
            elif method == 'alpha_beta_gamma':
                rho_s, alpha, beta, gamma, r_s = model_pars_arr
                model_pars_dict = {'rho_s': rho_s, 'alpha': alpha, 'beta': beta, 'gamma': gamma, 'r_s': r_s}
            else:
                rho_s, r_s = model_pars_arr
                model_pars_dict = {'rho_s': rho_s, 'r_s': r_s}
            return model_pars_dict
        
        # Mass splitting
        max_min_m, obj_m_groups, obj_center_groups, idx_groups = M_split(MASS_UNIT*obj_masses, obj_centers, start_time, NB_BINS = nb_bins)
        del obj_centers; del obj_center_groups; del idx_groups
        print_status(rank, start_time, "The number of objects considered is {0}".format(r200s.shape[0]))
        print_status(rank, start_time, "The mass bins (except maybe last) have size {0}".format(np.sum(np.int32([1 if (obj_masses[i]*MASS_UNIT > max_min_m[0] and obj_masses[i]*MASS_UNIT < max_min_m[1]) else 0 for i in range(r200s.shape[0])]))))
        print_status(rank, start_time, "The number of mass bins is {0}".format(len(obj_m_groups)))
        prof_models = {'einasto': getEinastoProf, 'alpha_beta_gamma': getAlphaBetaGammaProf, 'nfw': getNFWProf, 'hernquist': getHernquistProf}
        model_name = {'einasto': 'Einasto', 'alpha_beta_gamma': r'\alpha \beta \gamma', 'nfw': 'NFW', 'hernquist': 'Hernquist'}
        
        # Average over all objects' density profiles
        if r200s.shape[0] == 0:
            dens_profs = np.zeros((0,ROverR200.shape[0]), dtype = np.float32)
        y = [list(dens_profs[:,i]) for i in range(ROverR200.shape[0])]
        prof_median = np.array([np.median(z) if z != [] else np.nan for z in y])
        err_low = np.array([np.quantile(np.array(z), 0.25)/(np.sqrt(len(z))) if z != [] else np.nan for z in y])
        err_high = np.array([np.quantile(np.array(z), 0.75)/(np.sqrt(len(z))) if z != [] else np.nan for z in y])
        r200 = np.average(r200s[np.arange(r200s.shape[0])])
        # Prepare median for fitting
        if r200s.shape[0] == 0:
            dens_profs_fit = np.zeros((0,ROverR200_fit.shape[0]), dtype = np.float32)
        y = [list(dens_profs_fit[:,i]) for i in range(ROverR200_fit.shape[0])]
        prof_median_fit = np.array([np.median(z) if z != [] else np.nan for z in y])
        best_fit, obj_nb = fitDensProf(ROverR200_fit, method, (prof_median_fit, r200, 0)) # Fit median
        best_fit_dict = getModelParsDict(best_fit, method['profile'])
        del r200
        # Create VIZ_DEST if not available
        subprocess.call(['mkdir', '-p', '{}'.format(VIZ_DEST)], cwd=os.path.join(currentdir))
        
        # Go to outgoing unit system
        l_internal, m_internal, vel_internal = config.getLMVInternal()
        m_current = m_internal/MASS_UNIT
        dens_current = m_current/l_internal**3
        dens_target = config.OutUnitMass_in_g/config.OutUnitLength_in_cm**3
        dens_current_over_target = dens_current/dens_target
        l_label, m_label, vel_lable = config.LMVLabel()
        dens_label = r"{}/{}^3".format("({})".format(m_label) if "/h" in m_label else m_label, "({})".format(l_label) if "/h" in l_label else l_label).replace("sun", "_{\odot}")
        length_in_cm_d, mass_in_g_d, velocity_in_cm_per_s_d = config.getAdmissibleLMV()
        for key in list(length_in_cm_d.keys()):
            dens_label_new = dens_label.replace("{}".format(key), "\mathrm{{ {} }}".format(key))
            if dens_label_new != dens_label:
                dens_label = dens_label_new
                break
        dens_label = dens_label.replace("E+10", "10^{10}")
        # Best-fit parameters have dimensions too
        best_fit_dict['rho_s'] = best_fit_dict['rho_s']*dens_current_over_target
        
        # Plotting
        plt.figure()
        plt.loglog(ROverR200_fit, prof_models[method['profile']](ROverR200_fit*np.average(r200s[np.arange(r200s.shape[0])]), best_fit_dict), 'o--', color = 'r', linewidth=2, markersize=4, label=r'${}$-profile fit'.format(model_name[method['profile']]))
        plt.loglog(ROverR200, prof_median*dens_current_over_target, color = 'blue')
        plt.fill_between(ROverR200, (prof_median-err_low)*dens_current_over_target, (prof_median+err_high)*dens_current_over_target, facecolor = 'blue', edgecolor='g', alpha = 0.5, label = r"All objects")
        plt.xlabel(r"$r/R_{200}$")
        plt.ylabel(r"$\rho$ [${}$]".format(dens_label))
        plt.legend(loc="upper right", fontsize="x-small")
        plt.savefig("{}/RhoProf_{}.pdf".format(VIZ_DEST, SNAP), bbox_inches="tight")
        
        for group in range(len(obj_m_groups)):
            obj_pass_m = np.int32([1 if (obj_masses[i]*MASS_UNIT > max_min_m[group] and obj_masses[i]*MASS_UNIT < max_min_m[group+1]) else 0 for i in range(r200s.shape[0])])
            # Find profile median
            y = [list(dens_profs[np.nonzero(obj_pass_m > 0)[0],i]) for i in range(ROverR200.shape[0])]
            prof_median = np.array([np.median(z) if z != [] else np.nan for z in y])
            err_low = np.array([np.quantile(np.array(z), 0.25)/(np.sqrt(len(z))) if z != [] else np.nan for z in y])
            err_high = np.array([np.quantile(np.array(z), 0.75)/(np.sqrt(len(z))) if z != [] else np.nan for z in y])
            r200_m = np.average(r200s[np.arange(r200s.shape[0])[obj_pass_m.nonzero()[0]]])
            # Prepare median for fitting
            y = [list(dens_profs_fit[np.nonzero(obj_pass_m > 0)[0],i]) for i in range(ROverR200_fit.shape[0])]
            prof_median_fit = np.array([np.median(z) if z != [] else np.nan for z in y])
            best_fit_m, obj_nb = fitDensProf(ROverR200_fit, method, (prof_median_fit, r200_m, 0))
            best_fit_m_dict = getModelParsDict(best_fit_m, method['profile'])
            # Best-fit parameters have dimensions too
            best_fit_m_dict['rho_s'] = best_fit_m_dict['rho_s']*dens_current_over_target
            # Plotting
            plt.figure()
            plt.loglog(ROverR200_fit, prof_models[method['profile']](ROverR200_fit*np.average(r200s[np.arange(r200s.shape[0])[obj_pass_m.nonzero()[0]]]), best_fit_m_dict), 'o--', color = 'r', linewidth=2, markersize=4, label=r'${}$-profile fit'.format(model_name[method['profile']]))
            plt.loglog(ROverR200, prof_median*dens_current_over_target, color = 'blue')
            plt.fill_between(ROverR200, (prof_median-err_low)*dens_current_over_target, (prof_median+err_high)*dens_current_over_target, facecolor = 'blue', edgecolor='g', alpha = 0.5, label = r"$M: {0} - {1} \ M_{{\odot}}/h$".format(eTo10("{:.2E}".format(max_min_m[group])), eTo10("{:.2E}".format(max_min_m[group+1]))))
            plt.xlabel(r"$r/R_{200}$")
            plt.ylabel(r"$\rho$ [${}$]".format(dens_label))
            plt.legend(loc="upper right", fontsize="x-small")
            plt.savefig("{}/RhoProfM{:.2f}_{}.pdf".format(VIZ_DEST, np.float32(np.log10(max_min_m[group])), SNAP), bbox_inches="tight")
        del y; del err_low; del err_high
        return

def getModelParsDict(model_pars_arr, profile):
    if profile == 'einasto':
        rho_s, alpha, r_s = model_pars_arr
        model_pars_dict = {'rho_s': rho_s, 'alpha': alpha, 'r_s': r_s}
    elif profile == 'alpha_beta_gamma':
        rho_s, alpha, beta, gamma, r_s = model_pars_arr
        model_pars_dict = {'rho_s': rho_s, 'alpha': alpha, 'beta': beta, 'gamma': gamma, 'r_s': r_s}
    else:
        rho_s, r_s = model_pars_arr
        model_pars_dict = {'rho_s': rho_s, 'r_s': r_s}
    return model_pars_dict

def toMinimize(model_pars_arr, median, rbin_centers, profile):
    prof_models = {'einasto': getEinastoProf, 'alpha_beta_gamma': getAlphaBetaGammaProf, 'nfw': getNFWProf, 'hernquist': getHernquistProf}
    psi_2 = np.sum(np.array([(np.log(median[i])-np.log(prof_models[profile](rbin, getModelParsDict(model_pars_arr, profile))))**2/rbin_centers.shape[0] for i, rbin in enumerate(rbin_centers)]))
    return psi_2

def fitDensProf(ROverR200, method, median_r200_obj_nb):
    """
    Fit density profile according to model provided
    
    Note that ``median`` which is defined at ``ROverR200``
    must be in units of M_sun*h^2/(Mpc)**3. ``r200`` must be
    in units of Mpc/h.
    
    :param ROverR200: normalized radii where ``median`` is defined and 
        fitting should be carried out
    :type ROverR200: (N,) floats
    :param method: describes density profile model assumed for fitting, if parameter should be kept fixed during fitting then it needs to be provided, e.g. method['alpha'] = 0.18
    :type method: dictionary, method['profile'] is either `einasto`, `alpha_beta_gamma`, `hernquist`, `nfw`, minimum requirement
    :param median_r200_obj_nb: density profile (often a median of many profiles combined,
        in units of M_sun*h^2/(Mpc)**3), virial radius (of parent halo in units of Mpc/h) 
        and object number
    :type median_r200_obj_nb: tuple of (N,) floats, float, int
    :return best_fit, obj_nb: best-fit results, object number
    :rtype: (n,) floats, int"""
    
    median, r200, obj_nb = median_r200_obj_nb
    
    # Choose user's minimization method if available
    if 'min_method' in method:
        min_method = method['min_method']
    else:
        min_method = 'Powell'
    
    R_to_min = ROverR200*r200 # Mpc/h
    # Discard nan values in median
    R_to_min = R_to_min[~np.isnan(median)] # Note: np.isnan returns a boolean
    median = median[~np.isnan(median)]
    # Set initial guess and minimize scalar function
    try:
        if method['profile'] == 'einasto':
            if 'alpha' in method:
                alpha_lbound = method['alpha']
                alpha_rbound = alpha_lbound
            else:
                alpha_lbound = 1e-10 # Negative would mean density profile bends upwards in log-log plot
                alpha_rbound = np.inf
            if 'r_s' in method:
                r_s_lbound = method['r_s']
                r_s_rbound = r_s_lbound
            else:
                r_s_lbound = 1e-5
                r_s_rbound = np.inf
            iguess = np.array([0.1, 0.18, r200/5]) # Note: alpha = 0.18 gives ~ NFW
            res = optimize.minimize(toMinimize, iguess, method = min_method, args = (median/np.average(median), R_to_min, method['profile']), bounds = [(1e-7, np.inf), (alpha_lbound, alpha_rbound), (r_s_lbound, r_s_rbound)]) # Only hand over rescaled median!
            best_fit = res.x
        elif method['profile'] == 'alpha_beta_gamma':
            if 'alpha' in method:
                alpha_lbound = method['alpha']
                alpha_rbound = alpha_lbound
            else:
                alpha_lbound = 1e-5
                alpha_rbound = np.inf
            if 'beta' in method:
                beta_lbound = method['beta']
                beta_rbound = beta_lbound
            else:
                beta_lbound = 1e-5
                beta_rbound = np.inf
            if 'gamma' in method:
                gamma_lbound = method['gamma']
                gamma_rbound = gamma_lbound
            else:
                gamma_lbound = 1e-5
                gamma_rbound = np.inf
            if 'r_s' in method:
                r_s_lbound = method['r_s']
                r_s_rbound = r_s_lbound
            else:
                r_s_lbound = 1e-5
                r_s_rbound = np.inf
            iguess = np.array([0.1, 1.0, 1.0, 1.0, r200/5])
            res = optimize.minimize(toMinimize, iguess, method = min_method, args = (median/np.average(median), R_to_min, method['profile']), bounds = [(1e-7, np.inf), (alpha_lbound, alpha_rbound), (beta_lbound, beta_rbound), (gamma_lbound, gamma_rbound), (r_s_lbound, r_s_rbound)]) # Only hand over rescaled median!
            best_fit = res.x
        else:
            if 'r_s' in method:
                r_s_lbound = method['r_s']
                r_s_rbound = r_s_lbound
            else:
                r_s_lbound = 1e-5
                r_s_rbound = np.inf
            iguess = np.array([0.1, r200/5])
            res = optimize.minimize(toMinimize, iguess, method = min_method, args = (median/np.average(median), R_to_min, method['profile']), bounds = [(1e-7, np.inf), (r_s_lbound, r_s_rbound)]) # Only hand over rescaled median!
            best_fit = res.x
        best_fit[0] *= np.average(median)
    except ValueError: # For poor density profiles one might encounter "ValueError: `x0` violates bound constraints."
        best_fit = iguess*np.nan
    return best_fit, obj_nb

def fitDensProfHelper(dens_profs, ROverR200, r200s, method): # method is unhashable so @np_cache_factory(3,0) would fail
    """ Helper function to carry out density profile fitting
    
    :param dens_profs: array containing density profiles in units 
        of M_sun*h^2/(Mpc)**3
    :type dens_profs: (N,r_res) floats
    :param ROverR200: normalized radii where density profile is defined
        and fitting should be carried out
    :type ROverR200: (r_res,) floats
    :param r200s: R200 values of parent halos in units of Mpc/h
    :type r200s: (N,) floats
    :param method: describes density profile model assumed for fitting, if parameter should be kept fixed during fitting then it needs to be provided, e.g. method['alpha'] = 0.18
    :type method: dictionary, method['profile'] is either `einasto`, `alpha_beta_gamma`, `hernquist`, `nfw`, minimum requirement
    :return best_fits: best-fit results
    :rtype: (N,n) floats"""
    if rank == 0:
        if method['profile'] == 'einasto':
            best_fits = np.zeros((dens_profs.shape[0], 3))
        elif method['profile'] == 'alpha_beta_gamma':
            best_fits = np.zeros((dens_profs.shape[0], 5))
        else:
            best_fits = np.zeros((dens_profs.shape[0], 2))
        n_jobs = None
        try:
            n_jobs = len(os.sched_getaffinity(0))
        except AttributeError:
            # In this case (Windows and OSX), n_jobs stays None such that Pool will try to
            # automatically set the number of processes appropriately.
            pass
        with Pool(processes=n_jobs) as pool:
            results = pool.map(partial(fitDensProf, ROverR200, method), [(dens_profs[obj_nb], r200s[obj_nb], obj_nb) for obj_nb in range(dens_profs.shape[0])])
        for result in results:
            x, obj_nb = tuple(result)
            best_fits[obj_nb] = x
        return best_fits
    else:
        return None