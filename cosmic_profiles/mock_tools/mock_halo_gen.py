#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.integrate import quad
import os
from functools import partial
from pathos.multiprocessing import ProcessingPool as Pool
from cosmic_profiles.common.python_routines import drawUniformFromShell
from cosmic_profiles.dens_profs.dens_profs_tools import getEinastoProf, getAlphaBetaGammaProf, getHernquistProf, getNFWProf
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def genHalo(tot_mass, res, model_pars, method, a, b, c):
    """ Mock halo generator
    
    Create mock halo of mass ``tot_mass`` consisting of approximately ``res`` particles. The ``model_pars``
    array contains the parameters for the profile model given in ``method``. 
    
    :param tot_mass: total target mass of halo, in units of M_sun*h^2/Mpc^3
    :type tot_mass: float
    :param res: halo resolution
    :type res: int
    :param model_pars: parameters (except for ``rho_0`` which will be deduced from ``tot_mass``)
        in density profile model
    :type model_pars: float array (of length 4, 2 or 1)
    :param a: major axis array
    :type a: float array, units are Mpc/h
    :param b: intermediate axis array
    :type b: float array, units are Mpc/h
    :param c: minor axis array
    :type c: float array, units are Mpc/h
    :return: halo_x, halo_y, halo_z: arrays containing positions of halo particles, 
        mass_ptc: mass of each DM ptc in units of M_sun/h, rho_0: ``rho_0`` parameter in profile model
    :rtype: 3 (N,) float arrays, 2 floats
    """
    print('Starting genHalo()')
        
    if rank == 0:
        # Determine rho_0 in units of M_sun*h^2/Mpc^3
        def getMassIntegrand0(r, method, model_pars):
            if method == 'einasto':
                r_2, alpha = model_pars
                return 4*np.pi*r**2*np.exp(-2/alpha*((r/r_2)**alpha-1))
            if method == 'alpha_beta_gamma':
                alpha, beta, gamma, r_s = model_pars
                return 4*np.pi*r**2/((r/r_s)**gamma*(1+(r/r_s)**alpha)**((beta-gamma)/alpha))
            if method == 'hernquist':
                r_s = model_pars
                return 4*np.pi*r**2/((r/r_s)*(1+r/r_s)**3)
            else:
                r_s = model_pars
                return 4*np.pi*r**2/((r/r_s)*(1+r/r_s)**2)
        rho_0 = tot_mass/quad(getMassIntegrand0, 1e-8, a[-1], args=(method, model_pars))[0]
        model_pars = np.hstack((np.array([rho_0]),model_pars))
        
        # Determine number of particles in second shell (first proper shell)
        def getMassIntegrand(r, model_pars):
            if method == 'einasto':
                return 4*np.pi*r**2*getEinastoProf(r, model_pars)
            if method == 'alpha_beta_gamma':
                return 4*np.pi*r**2*getAlphaBetaGammaProf(r, model_pars)
            if method == 'hernquist':
                return 4*np.pi*r**2*getHernquistProf(r, model_pars)
            else:
                return 4*np.pi*r**2*getNFWProf(r, model_pars)
        
        # Determine number of particles in second shell
        mass_1 = quad(getMassIntegrand, a[0], a[1], args=(model_pars))[0] # Mass in shell 1 in units of M_sun/h
        N1 = int(round(res*mass_1/tot_mass)) # Number of particles in shell 1. Rounding error is unavoidable.
        count_it = 0
        while N1 == 0: # Need to have coarser radial binning in this case
            a = a[::2] if a.shape[0] % 2 == 1 else np.hstack((a[::2], a[-1])) # Reduce number of shells 
            b = b[::2] if b.shape[0] % 2 == 1 else np.hstack((b[::2], b[-1]))
            c = c[::2] if c.shape[0] % 2 == 1 else np.hstack((c[::2], c[-1]))
            mass_1 = quad(getMassIntegrand, a[0], a[1], args=(model_pars))[0] # Mass in shell 1 in units of M_sun/h
            N1 = int(round(res*mass_1/tot_mass)) # Number of particles in shell 1. Rounding error is unavoidable.
            if count_it > 1000:
                raise ValueError("Please provide a higher halo resolution target!")
            count_it += 1
        # Build up halo successively in onion-like fashion
        halo_x = np.empty(0)
        halo_y = np.empty(0)
        halo_z = np.empty(0)
        Nptc = np.zeros((a.shape[0],), dtype = np.int32) # Note: Number of ptcs in shell Shell(a[idx-1],b[idx-1],c[idx-1],a[idx],b[idx],c[idx])
        for shell in range(a.shape[0]):
            if shell != 0:
                Nptc[shell] = int(round(N1*quad(getMassIntegrand, a[shell-1], a[shell], args=(model_pars))[0]/quad(getMassIntegrand, a[0], a[1], args=(model_pars))[0]))
            else:
                Nptc[shell] = int(round(N1*quad(getMassIntegrand, 1e-8, a[0], args=(model_pars))[0]/quad(getMassIntegrand, a[0], a[1], args=(model_pars))[0]))
        with Pool(processes=os.sched_getaffinity(0)) as pool:
            results = pool.map(partial(drawUniformFromShell, 3, a, b, c, Nptc), [idx for idx in range(Nptc.shape[0])]) # Draws from Shell(a[idx-1],b[idx-1],c[idx-1],a[idx],b[idx],c[idx])
        for result in results:
            halo_x = np.hstack((halo_x, result[:,0]))
            halo_y = np.hstack((halo_y, result[:,1]))
            halo_z = np.hstack((halo_z, result[:,2]))
        mass_dm = quad(getMassIntegrand, 1e-8, a[-1], args=(model_pars))[0]/halo_z.shape[0]    
        return halo_x, halo_y, halo_z, mass_dm, rho_0
    else:
        return None, None, None, None, None
