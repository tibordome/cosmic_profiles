#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 16:21:27 2022
@author: tibor
"""

import numpy as np
import os
import subprocess
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
subprocess.call(['python3', 'setup.py', 'build_ext', '--inplace'], cwd=os.path.join(currentdir, '..'))
subprocess.call(['mkdir', 'viz'], cwd=os.path.join(currentdir))
subprocess.call(['mkdir', 'cat'], cwd=os.path.join(currentdir))
sys.path.append(os.path.join(currentdir, '..', '..')) # Only needed if cosmic_profiles is not installed
from cosmic_profiles import genHalo, CosmicProfilesDirect
import time
start_time = time.time()

def test_densities():
    #################################### Parameters ################################################
    L_BOX = np.float32(10) # Mpc/h
    CAT_DEST = "./cat"
    VIZ_DEST = "./viz"
    D_LOGSTART = -2
    D_LOGEND = 0
    D_BINS = 30 # If D_LOGSTART == -2 D_LOGEND == 1, 60 corresponds to shell width of 0.05 dex
    M_TOL = np.float32(1e-2)
    N_WALL = 100
    N_MIN = 10
    SNAP = '015'
    MASS_UNIT = 1e+10
    MIN_NUMBER_DM_PTCS = 1000
    CENTER = 'mode'
    r_over_rvir = np.logspace(-2,0,50)
    method = 'einasto'
    nb_model_pars = {'einasto': 3, 'nfw': 2, 'hernquist': 2, 'alpha_beta_gamma': 5}
    N = 10 # Number of halos used for test
    
    #################################### Generate N mock halos ####################################
    r_s = 0.5 # Units are Mpc/h
    alpha = 0.18
    
    model_pars = np.array([r_s, alpha])
    dm_x = np.empty(0, dtype = np.float32)
    dm_y = np.empty(0, dtype = np.float32)
    dm_z = np.empty(0, dtype = np.float32)
    nb_ptcs = np.empty(0, dtype = np.int32)
    r_vir = np.empty(0, dtype = np.float32)
    for n in range(N):
        tot_mass = 10**(np.random.uniform(10,12,1)[0]) # M_sun/h
        halo_res = np.random.uniform(600,20000,1)[0]
        r_vir = np.hstack((r_vir, np.float32(np.random.uniform(0.5,1.5,1)[0]))) # Units are Mpc/h
        a = np.logspace(-1.5,0.2,100)*r_vir[-1] # Units are Mpc/h
        b = a*0.6 # Units are Mpc/h
        c = a*0.2 # Units are Mpc/h
        halo_x, halo_y, halo_z, mass_dm, rho_0 = genHalo(tot_mass, halo_res, model_pars, 'einasto', a, b, c)
        print("Number of particles in the halo is {}.".format(halo_x.shape[0]))
        halo_x += np.random.uniform(0,L_BOX/2,1)[0] # Move mock halo into the middle of the simulation box
        halo_y += np.random.uniform(0,L_BOX/2,1)[0]
        halo_z += np.random.uniform(0,L_BOX/2,1)[0]
        dm_x = np.hstack((dm_x, halo_x))
        dm_y = np.hstack((dm_y, halo_y))
        dm_z = np.hstack((dm_z, halo_z))
        nb_ptcs = np.hstack((nb_ptcs, halo_x.shape[0]))
    dm_xyz = np.float32(np.hstack((np.reshape(dm_x, (dm_x.shape[0],1)), np.reshape(dm_y, (dm_y.shape[0],1)), np.reshape(dm_z, (dm_z.shape[0],1)))))
    
    ######################### Extract R_vir, halo indices and halo sizes ##########################
    mass_array = np.ones((dm_xyz.shape[0],), dtype = np.float32)*mass_dm/MASS_UNIT # Has to be in unit mass (= 10^10 M_sun/h)
    h_indices = [np.arange(0+np.sum(nb_ptcs[:idx]),nb_ptc+np.sum(nb_ptcs[:idx]), dtype = np.int32).tolist() for idx, nb_ptc in enumerate(nb_ptcs)]
    
    ########################### Define CosmicProfilesDirect object ###################################
    cprofiles = CosmicProfilesDirect(dm_xyz, mass_array, h_indices, r_vir, CAT_DEST, VIZ_DEST, SNAP, L_BOX, MIN_NUMBER_DM_PTCS, D_LOGSTART, D_LOGEND, D_BINS, M_TOL, N_WALL, N_MIN, CENTER, start_time)
    
    ############################## Estimate Density Profile ########################################
    cprofiles.calcDensProfsDirectBinning(r_over_rvir)
    cprofiles.calcDensProfsKernelBased(r_over_rvir)
    dens_profs_db, r_over_rvir = cprofiles.fetchDensProfsDirectBinning() # dens_profs_db is in M_sun*h^2/Mpc^3
    dens_profs_kb, r_over_rvir = cprofiles.fetchDensProfsKernelBased()
    assert dens_profs_db.shape[0] == N    
    assert dens_profs_db.shape[1] == r_over_rvir.shape[0]
    assert dens_profs_kb.shape[0] == N    
    assert dens_profs_kb.shape[1] == r_over_rvir.shape[0]
    
    ############################## Fit Density Profile #############################################
    r_over_rvir = r_over_rvir[10:] # Do not fit innermost region since not reliable in practice. Use gravitational softening scale and / or relaxation timescale to estimate inner convergence radius.
    dens_profs_db = dens_profs_db[:,10:]
    cprofiles.fitDensProfs(dens_profs_db, r_over_rvir, cprofiles.fetchCat(), r_vir, method = method)
    best_fits, r_over_rvir = cprofiles.fetchDensProfsBestFits(method)
    assert best_fits.shape[0] == N
    assert best_fits.shape[1] == nb_model_pars[method]
    
    # Clean-up
    subprocess.call(['bash', 'decythonize.sh'], cwd=os.path.join(currentdir, '..'))
