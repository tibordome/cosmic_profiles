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
    
    return

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
    
    return