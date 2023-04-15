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
import itertools
import pytest
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
subprocess.call(['python3', 'setup_compile.py', 'build_ext', '--inplace'], cwd=os.path.join(currentdir, '..', '..'))
subprocess.call(['mkdir', 'viz'], cwd=os.path.join(currentdir))
subprocess.call(['mkdir', 'cat'], cwd=os.path.join(currentdir))
sys.path.append(os.path.join(currentdir, '..', '..')) # Only needed if cosmic_profiles is not installed
from cosmic_profiles import genHalo, DensShapeProfs, updateInUnitSystem, updateOutUnitSystem
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

@pytest.mark.parametrize('method, reduced, shell_based', [p for p in itertools.product(*[['einasto', 'nfw', 'hernquist', 'alpha_beta_gamma'], [False, True], [False, True]])])
def test_shapes(method, reduced, shell_based):
    #################################### Parameters ################################################
    updateInUnitSystem(in_unit_length_in_cm = 3.085678e24, in_unit_mass_in_g = 1.989e33, in_unit_velocity_in_cm_per_s = 1e5)
    updateOutUnitSystem(out_unit_length_in_cm = 3.085678e24, out_unit_mass_in_g = 1.989e33, out_unit_velocity_in_cm_per_s = 1e5)
    L_BOX = np.float32(10) # Mpc/h
    VIZ_DEST = "./cosmic_profiles/tests/viz"
    CAT_DEST = "./cosmic_profiles/tests/cat"
    D_LOGSTART = -2
    D_LOGEND = 0
    D_BINS = 30 # If D_LOGSTART == -2 D_LOGEND == 1, 60 corresponds to shell width of 0.05 dex
    IT_TOL = np.float32(1e-2)
    IT_WALL = 100
    IT_MIN = 10
    SNAP = '016'
    MIN_NUMBER_DM_PTCS = 1000
    CENTER = 'mode'
    HIST_NB_BINS = 11 # Number of bins used for e.g. ellipticity histogram
    frac_r200 = 0.5 # At what depth to calculate e.g. histogram of triaxialities (cf. plotLocalTHist())
    N = 10 # Number of halos used for test
    # For (ellipsoidal shell-based) density profiles
    r_over_rvir = np.logspace(-2,0,50)
    nb_model_pars = {'einasto': 3, 'nfw': 2, 'hernquist': 2, 'alpha_beta_gamma': 5}
    
    #################################### Generate N mock halos ####################################
    r_s = 0.5 # Units are Mpc/h
    alpha = 0.18
    
    model_pars = {'alpha': alpha, 'r_s': r_s}
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
        halo_x, halo_y, halo_z, mass_dm, rho_s = genHalo(tot_mass, halo_res, model_pars, 'einasto', a, b, c)
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
    mass_array = np.ones((dm_xyz.shape[0],), dtype = np.float32)*mass_dm # In M_sun/h
    idx_cat_in = [np.arange(0+np.sum(nb_ptcs[:idx]),nb_ptc+np.sum(nb_ptcs[:idx]), dtype = np.int32).tolist() for idx, nb_ptc in enumerate(nb_ptcs)]
    
    ########################### Define CosmicProfilesDirect object ###################################
    cprofiles = DensShapeProfs(dm_xyz, mass_array, idx_cat_in, r_vir, SNAP, L_BOX, MIN_NUMBER_DM_PTCS, D_LOGSTART, D_LOGEND, D_BINS, IT_TOL, IT_WALL, IT_MIN, CENTER, VIZ_DEST, CAT_DEST)
    
    idx_cat, obj_size = cprofiles.getIdxCat()
    obj_numbers = [0, 1, 2, 3, 4, 5]
    centers, ms = cprofiles.getMassesCenters(obj_numbers)
    if rank == 0:
        assert len(obj_size) <= len(idx_cat_in) # Note: idx_cat only contains objects with sufficient resolution
        assert centers.shape == (len(obj_numbers),3)
        assert ms.shape == (len(obj_numbers),)
    
    ######################### Calculating Local Morphological Properties #############################
    # Create halo shape catalogue
    d, q, s, minor, inter, major, obj_centers, obj_masses = cprofiles.getShapeCatLocal(obj_numbers, reduced = reduced, shell_based = shell_based)
    
    if rank == 0:
        nb_suff_res = len(obj_numbers)
        assert obj_masses.shape[0] == nb_suff_res
        assert obj_centers.shape[0] == nb_suff_res
        assert d.shape[0] == nb_suff_res
        assert q.shape[0] == nb_suff_res
        assert s.shape[0] == nb_suff_res
        assert minor.shape[0] == nb_suff_res
        assert inter.shape[0] == nb_suff_res
        assert major.shape[0] == nb_suff_res
        assert d.shape[1] == D_BINS+1
        assert q.shape[1] == D_BINS+1
        assert s.shape[1] == D_BINS+1
        assert minor.shape[1] == D_BINS + 1
        assert minor.shape[2] == 3
        assert inter.shape[1] == D_BINS + 1
        assert inter.shape[2] == 3
        assert major.shape[1] == D_BINS + 1
        assert major.shape[2] == 3
    
    # Draw halo shape profiles (overall and mass-decomposed ones)
    cprofiles.plotShapeProfs(nb_bins = 2, obj_numbers = obj_numbers, reduced = reduced, shell_based = shell_based)
    
    # Viz first few halos' shapes
    cprofiles.vizLocalShapes(obj_numbers = obj_numbers, reduced = reduced, shell_based = shell_based)
    
    # Plot halo triaxiality histogram
    cprofiles.plotLocalTHist(HIST_NB_BINS, frac_r200, obj_numbers = obj_numbers, reduced = reduced, shell_based = shell_based)
    
    ######################### Calculating Global Morphological Properties ############################
    obj_numbers = np.arange(N)
    d, q, s, minor, inter, major, obj_centers, obj_masses = cprofiles.getShapeCatGlobal(obj_numbers = obj_numbers, reduced = reduced)
    
    if rank == 0:
        nb_suff_res = len(obj_numbers)
        assert obj_masses.shape[0] == nb_suff_res
        assert obj_centers.shape[0] == nb_suff_res
        assert d.shape[0] == nb_suff_res
        assert q.shape[0] == nb_suff_res
        assert s.shape[0] == nb_suff_res
        assert minor.shape[0] == nb_suff_res
        assert inter.shape[0] == nb_suff_res
        assert major.shape[0] == nb_suff_res
        assert d.shape[1] == 1
        assert q.shape[1] == 1
        assert s.shape[1] == 1
        assert minor.shape[1] == 1
        assert minor.shape[2] == 3
        assert inter.shape[1] == 1
        assert inter.shape[2] == 3
        assert major.shape[1] == 1
        assert major.shape[2] == 3
    
    # Plot halo ellipticity histogram
    cprofiles.plotGlobalEpsHist(HIST_NB_BINS, obj_numbers = obj_numbers)
    
    # Viz first few halos' shapes
    cprofiles.vizGlobalShapes(obj_numbers = obj_numbers, reduced = reduced)
    
    ######################### Calculating Ellipsoidal Density Profiles ######################################################
    dens_profs_db = cprofiles.estDensProfs(r_over_rvir, obj_numbers = obj_numbers, direct_binning = True, reduced = reduced, shell_based = shell_based) # dens_profs_db is in M_sun*h^2/Mpc^3
    dens_profs_kb = cprofiles.estDensProfs(r_over_rvir, obj_numbers = obj_numbers, direct_binning = False) # These estimates will be kernel-based
    if rank == 0:
        nb_suff_res = len(obj_numbers)
        assert dens_profs_db.shape[0] == nb_suff_res
        assert dens_profs_db.shape[1] == r_over_rvir.shape[0]
        assert dens_profs_kb.shape[0] == nb_suff_res
        assert dens_profs_kb.shape[1] == r_over_rvir.shape[0]
    else:
        dens_profs_db = np.zeros((nb_suff_res, r_over_rvir.shape[0]), dtype = np.float32)
    comm.Bcast(dens_profs_db, root = 0)
    
    ############################## Fit Ellipsoidal Density Profile ##########################################################
    r_over_rvir_fit = r_over_rvir[10:] # Do not fit innermost region since not reliable in practice. Use gravitational softening scale and / or relaxation timescale to estimate inner convergence radius.
    dens_profs_db_fit = dens_profs_db[:,10:]
    best_fits = cprofiles.fitDensProfs(dens_profs_db_fit, r_over_rvir_fit, method = method, obj_numbers = obj_numbers)
    if rank == 0:
        assert best_fits.shape[0] == nb_suff_res
        assert best_fits.shape[1] == nb_model_pars[method]
        
    # Draw ellipsoidal halo density profiles (overall and mass-decomposed ones). The results from fitDensProfs() got cached.
    cprofiles.plotDensProfs(dens_profs_db, r_over_rvir, dens_profs_db[:,25:], r_over_rvir[25:], method = 'nfw', nb_bins = 2, obj_numbers = obj_numbers)