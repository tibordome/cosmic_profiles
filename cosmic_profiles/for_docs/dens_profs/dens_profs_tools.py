#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

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
    return

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
    return

def fitDensProfHelper(dens_profs, ROverR200, r200s, method):
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
    return