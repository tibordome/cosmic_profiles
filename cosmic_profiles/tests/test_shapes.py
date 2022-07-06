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
from cosmic_profiles import genHalo, DensShapeProfs
import time
start_time = time.time()

def test_shapes():
    #################################### Parameters ################################################
    L_BOX = np.float32(10) # Mpc/h
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
    HIST_NB_BINS = 11 # Number of bins used for e.g. ellipticity histogram
    frac_r200 = 0.5 # At what depth to calculate e.g. histogram of triaxialities (cf. plotLocalTHist())
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
    idx_cat = [np.arange(0+np.sum(nb_ptcs[:idx]),nb_ptc+np.sum(nb_ptcs[:idx]), dtype = np.int32).tolist() for idx, nb_ptc in enumerate(nb_ptcs)]
    
    ########################### Define CosmicProfilesDirect object ###################################
    cprofiles = DensShapeProfs(dm_xyz, mass_array, idx_cat, r_vir, SNAP, L_BOX, MIN_NUMBER_DM_PTCS, D_LOGSTART, D_LOGEND, D_BINS, M_TOL, N_WALL, N_MIN, CENTER, start_time)
    
    assert idx_cat == cprofiles.getIdxCat()
    centers, ms = cprofiles.getMassesCenters()
    nb_pass = len([1 if nb_ptc > MIN_NUMBER_DM_PTCS else 0 for nb_ptc in nb_ptcs])
    assert centers.shape == (nb_pass,3)
    assert ms.shape == (nb_pass,)
    
    ######################### Calculating Local Morphological Properties #############################
    # Create halo shape catalogue
    d, q, s, minor, inter, major, obj_centers, obj_masses, succeeded = cprofiles.getShapeCatLocal()
    
    idx_cat_local = cprofiles.getIdxCatLocal()
    assert np.sum([1 if idx_cat[obj] != [] else 0 for obj in range(len(idx_cat))]) >= np.sum([1 if idx_cat_local[obj] != [] else 0 for obj in range(len(idx_cat_local))])
    
    assert obj_masses.shape[0] <= nb_pass
    assert obj_centers.shape[0] <= nb_pass
    assert d.shape[0] <= nb_pass
    assert q.shape[0] <= nb_pass
    assert s.shape[0] <= nb_pass
    assert minor.shape[0] <= nb_pass
    assert inter.shape[0] <= nb_pass
    assert major.shape[0] <= nb_pass
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
    cprofiles.plotShapeProfs(VIZ_DEST)
    
    # Viz first few halos' shapes
    cprofiles.vizLocalShapes([0,1,2], VIZ_DEST)
    
    # Plot halo triaxiality histogram
    cprofiles.plotLocalTHist(HIST_NB_BINS, VIZ_DEST, frac_r200)
    
    ######################### Calculating Global Morphological Properties ############################
    d, q, s, minor, inter, major, obj_centers, obj_masses = cprofiles.getShapeCatGlobal()
    
    idx_cat_global = cprofiles.getIdxCatGlobal()
    assert np.sum([1 if idx_cat[obj] != [] else 0 for obj in range(len(idx_cat))]) >= np.sum([1 if idx_cat_global[obj] != [] else 0 for obj in range(len(idx_cat_global))])
    
    assert obj_masses.shape[0] == nb_pass
    assert obj_centers.shape[0] == nb_pass
    assert d.shape[0] == nb_pass
    assert q.shape[0] == nb_pass
    assert s.shape[0] == nb_pass
    assert minor.shape[0] == nb_pass
    assert inter.shape[0] == nb_pass
    assert major.shape[0] == nb_pass
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
    cprofiles.plotGlobalEpsHist(HIST_NB_BINS, VIZ_DEST)
    
    # Viz first few halos' shapes
    cprofiles.vizGlobalShapes([0,1,2], VIZ_DEST)
    
    # Clean-up
    subprocess.call(['bash', 'decythonize.sh'], cwd=os.path.join(currentdir, '..'))
