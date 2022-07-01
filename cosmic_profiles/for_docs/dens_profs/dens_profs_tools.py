#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
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

def getDensityProfiles(VIZ_DEST, SNAP, cat, r200s, fits_ROverR200, dens_profs, ROverR200, obj_masses, obj_centers, method, start_time, MASS_UNIT=1e10, suffix = '_'):
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
    :param fits_ROverR200: normalized radii at which the mass-decomposed density
        profile fits shall be calculated
    :type fits_ROverR200: (N3,) floats
    :param dens_profs: density profiles, defined at ``ROverR200``, in M_sun*h^2/(Mpc)**3
    :type dens_profs: (N, n) floats
    :param ROverR200: normalized radii at which ``dens_profs`` are defined
    :type ROverR200: (N4,) floats
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
    :param suffix: either '_dm_' or '_gx_' or '' (latter for CosmicProfilesDirect)
    :type suffix: string"""
    
    return

def fitDensProf(median, ROverR200, r200, method):
    """
    Fit density profile according to model provided
    
    Note that ``median`` which is defined at ``ROverR200``
    must be in units of UnitMass/(Mpc/h)**3.
    
    :param median: density profile (often a median of many profiles combined)
    :type median: (N,) floats
    :param ROverR200: normalized radii where ``median`` is defined and 
        fitting should be carried out
    :type ROverR200: (N,) floats
    :param r200: virial raiuds (of parent halo in case of gx)
    :type r200: float
    :param method: string describing density profile model assumed for fitting
    :type method: string, either `einasto`, `alpha_beta_gamma`, `hernquist`, `nfw`
    :return res.x: best-fit results
    :rtype: (n,) floats"""
    
    return
        
def fitDensProfHelper(ROverR200, method, dens_prof_plus_r200_plus_obj_number):
    """ Helper function to carry out density profile fitting
    
    :param ROverR200: normalized radii where density profile is defined
        and fitting should be carried out
    :type ROverR200: (N2,) floats
    :param method: string describing density profile model assumed for fitting
    :type method: string, either `einasto`, `alpha_beta_gamma`, `hernquist`, `nfw`
    :param dens_prof_plus_r200_plus_obj_number: array containing density profile,
        (in units of UnitMass/(Mpc/h)**3), r200-value and object number
    :type dens_prof_plus_r200_plus_obj_number: (N,) float
    :return res, object_number: best-fit results and object number
    :rtype: (n,) floats, int"""
        
    return
