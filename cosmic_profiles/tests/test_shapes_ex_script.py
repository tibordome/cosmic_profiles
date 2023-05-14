#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 18:23:43 2021
"""

import numpy as np
import os
import subprocess
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
subprocess.call(['python3', 'setup_compile.py', 'build_ext', '--inplace'], cwd=os.path.join(currentdir, '..', '..'))
subprocess.call(['mkdir', 'viz'], cwd=os.path.join(currentdir))
subprocess.call(['mkdir', 'cat'], cwd=os.path.join(currentdir))
sys.path.append(os.path.join(currentdir, '..', '..')) # Only needed if cosmic_profiles is not installed
from cosmic_profiles import genHalo, DensShapeProfs, updateInUnitSystem, updateOutUnitSystem

def test_shapes_ex_script():
    
    #################################### Parameters ################################################
    updateInUnitSystem(length_in_cm = 'Mpc/h', mass_in_g = 'Msun/h', velocity_in_cm_per_s = 1e5, little_h = 0.6774)
    updateOutUnitSystem(length_in_cm = 'kpc/h', mass_in_g = 'Msun/h', velocity_in_cm_per_s = 1e5, little_h = 0.6774)
    L_BOX = np.float32(10) # Mpc/h
    SNAP = '015'
    CAT_DEST = "./cosmic_profiles/tests/cat"
    VIZ_DEST = "./cosmic_profiles/tests/viz"
    MIN_NUMBER_PTCS = 200
    CENTER = 'mode'   
    katz_config = {
        'ROverR200': np.logspace(-1.5,0,70),
        'IT_TOL': 1e-2,
        'IT_WALL': 100,
        'IT_MIN': 10,
        'REDUCED': False, 
        'SHELL_BASED': False
    }
    
    #################################### Generate 1 mock halo ######################################
    tot_mass = 10**(12) # M_sun/h
    halo_res = 100000
    r_s = 0.5 # Units are Mpc/h
    alpha = 0.18
    N_bin = 100
    r_vir = np.array([1.0], dtype = np.float32) # Units are Mpc/h
    a = np.logspace(-1.5,0.2,N_bin)*r_vir[0] # Units are Mpc/h
    b = a*0.6 # Units are Mpc/h
    c = a*0.2 # Units are Mpc/h
    
    model_pars = {'alpha': alpha, 'r_s': r_s}
    halo_x, halo_y, halo_z, mass_dm, rho_s = genHalo(tot_mass, halo_res, model_pars, 'einasto', a, b, c)
    print("Number of particles in the halo is {}.".format(halo_x.shape[0]))
    halo_x += L_BOX/2 # Move mock halo into the middle of the simulation box
    halo_y += L_BOX/2
    halo_z += L_BOX/2
    dm_xyz = np.float32(np.hstack((np.reshape(halo_x, (halo_x.shape[0],1)), np.reshape(halo_y, (halo_y.shape[0],1)), np.reshape(halo_z, (halo_z.shape[0],1)))))
    
    ######################### Extract R_vir, halo indices and halo sizes ###########################
    mass_array = np.ones((dm_xyz.shape[0],), dtype = np.float32)*mass_dm # In M_sun/h
    idx_cat = [np.arange(len(halo_x), dtype = np.int32).tolist()]
    
    ########################### Define DensShapeProfs object #######################################
    cprofiles = DensShapeProfs(dm_xyz, mass_array, idx_cat, r_vir, L_BOX, SNAP, VIZ_DEST, CAT_DEST, MIN_NUMBER_PTCS = MIN_NUMBER_PTCS, CENTER = CENTER)
    
    ######################### Calculating Morphological Properties #################################
    # Create halo shape catalogue
    obj_numbers = [0]
    cprofiles.dumpShapeCatLocal(obj_numbers = obj_numbers, katz_config = katz_config)
    
    ######################################## Visualizations ########################################
    # Visualize halo: A sample output is shown above!
    cprofiles.vizLocalShapes(obj_numbers = obj_numbers, katz_config = katz_config)