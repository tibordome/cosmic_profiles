#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 16:21:27 2022
@author: tibor
"""

import numpy as np
import os
import itertools
import pytest
import subprocess
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
subprocess.call(['python3', 'setup_compile.py', 'build_ext', '--inplace'], cwd=os.path.join(currentdir, '..', '..'))
subprocess.call(['mkdir', 'viz'], cwd=os.path.join(currentdir))
subprocess.call(['mkdir', 'cat'], cwd=os.path.join(currentdir))
sys.path.append(os.path.join(currentdir, '..', '..')) # Only needed if cosmic_profiles is not installed
from cosmic_profiles import genHalo, DensProfs, updateInUnitSystem, updateOutUnitSystem
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

@pytest.mark.parametrize('method, direct_binning', [p for p in itertools.product(*[['einasto', 'nfw', 'hernquist', 'alpha_beta_gamma'], [False, True]])])
def test_densities(method, direct_binning):
    #################################### Parameters #################################################
    updateInUnitSystem(in_unit_length_in_cm = 3.085678e24, in_unit_mass_in_g = 1.989e33, in_unit_velocity_in_cm_per_s = 1e5)
    updateOutUnitSystem(out_unit_length_in_cm = 3.085678e24, out_unit_mass_in_g = 1.989e33, out_unit_velocity_in_cm_per_s = 1e5)
    L_BOX = np.float32(10) # Mpc/h
    SNAP = '018'
    MIN_NUMBER_DM_PTCS = 1000
    CENTER = 'mode'
    r_over_rvir = np.logspace(-2,0,50)
    VIZ_DEST = "./cosmic_profiles/tests/viz"
    CAT_DEST = "./cosmic_profiles/tests/cat"
    nb_model_pars = {'einasto': 3, 'nfw': 2, 'hernquist': 2, 'alpha_beta_gamma': 5}
    N = 10 # Number of halos used for test
    
    #################################### Generate N mock halos ######################################
    r_s = 0.5 # Units are Mpc/h
    alpha = 0.18
    beta = 3.0
    gamma = 1.0
    
    model_pars = {'einasto': {'alpha': alpha, 'r_s': r_s}, 'nfw': {'r_s': r_s}, 'hernquist': {'r_s': r_s}, 'alpha_beta_gamma': {'alpha': alpha, 'beta': beta, 'gamma': gamma, 'r_s': r_s}}
    dm_x = np.empty(0, dtype = np.float32)
    dm_y = np.empty(0, dtype = np.float32)
    dm_z = np.empty(0, dtype = np.float32)
    nb_ptcs = np.empty(0, dtype = np.int32)
    r_vir = np.empty(0, dtype = np.float32)
    for n in range(N):
        tot_mass = 10**(np.random.uniform(10,12,1)[0]) # M_sun/h
        halo_res = np.random.uniform(3000,20000,1)[0]
        r_vir = np.hstack((r_vir, np.float32(np.random.uniform(0.5,1.5,1)[0]))) # Units are Mpc/h
        a = np.logspace(-1.5,0.2,100)*r_vir[-1] # Units are Mpc/h
        b = a*0.6 # Units are Mpc/h
        c = a*0.2 # Units are Mpc/h
        halo_x, halo_y, halo_z, mass_dm, rho_s = genHalo(tot_mass, halo_res, model_pars[method], method, a, b, c)
        print("Number of particles in the halo is {}.".format(halo_x.shape[0]))
        halo_x += np.random.uniform(0,L_BOX/2,1)[0] # Move mock halo into the middle of the simulation box
        halo_y += np.random.uniform(0,L_BOX/2,1)[0]
        halo_z += np.random.uniform(0,L_BOX/2,1)[0]
        dm_x = np.hstack((dm_x, halo_x))
        dm_y = np.hstack((dm_y, halo_y))
        dm_z = np.hstack((dm_z, halo_z))
        nb_ptcs = np.hstack((nb_ptcs, halo_x.shape[0]))
    dm_xyz = np.float32(np.hstack((np.reshape(dm_x, (dm_x.shape[0],1)), np.reshape(dm_y, (dm_y.shape[0],1)), np.reshape(dm_z, (dm_z.shape[0],1)))))
    
    ######################### Extract R_vir, halo indices and halo sizes #############################
    mass_array = np.ones((dm_xyz.shape[0],), dtype = np.float32)*mass_dm # In M_sun/h
    idx_cat_in = [np.arange(0+np.sum(nb_ptcs[:idx]),nb_ptc+np.sum(nb_ptcs[:idx]), dtype = np.int32).tolist() for idx, nb_ptc in enumerate(nb_ptcs)]
    
    ########################### Define DensProfs object ##############################################
    cprofiles = DensProfs(dm_xyz, mass_array, idx_cat_in, r_vir, SNAP, L_BOX, MIN_NUMBER_DM_PTCS, CENTER, VIZ_DEST, CAT_DEST)
    
    ############################## Estimate Density Profiles #########################################
    obj_numbers = np.arange(5)
    dens_profs = cprofiles.estDensProfs(r_over_rvir, obj_numbers = obj_numbers, direct_binning = direct_binning) # dens_profs is in M_sun*h^2/Mpc^3
    if rank == 0:
        nb_suff_res = len(obj_numbers)
        assert dens_profs.shape[0] == nb_suff_res
        assert dens_profs.shape[1] == r_over_rvir.shape[0]
    else:
        dens_profs = np.zeros((nb_suff_res, r_over_rvir.shape[0]), dtype = np.float32)
    comm.Bcast(dens_profs, root = 0)
    
    ############################## Fit Density Profile ###############################################
    r_over_rvir_fit = r_over_rvir[10:] # Do not fit innermost region since not reliable in practice. Use gravitational softening scale and / or relaxation timescale to estimate inner convergence radius.
    dens_profs_fit = dens_profs[:,10:]
    best_fits = cprofiles.fitDensProfs(dens_profs_fit, r_over_rvir_fit, method = method, obj_numbers = obj_numbers)
    if rank == 0:
        assert best_fits.shape[0] == nb_suff_res
        assert best_fits.shape[1] == nb_model_pars[method]