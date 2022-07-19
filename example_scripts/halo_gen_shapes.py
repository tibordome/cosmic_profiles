#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 18:23:43 2021
"""

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
import numpy as np
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.append(os.path.join(currentdir, '..')) # Only needed if cosmic_profiles is not installed
from cosmic_profiles import DensShapeProfs, genHalo

def calcShapeEx():
    
    #################################### Parameters ################################################
    L_BOX = np.float32(10) # Mpc/h
    CAT_DEST = "./cat"
    VIZ_DEST = "./viz"
    D_LOGSTART = -2
    D_LOGEND = 0
    D_BINS = 20 # If D_LOGSTART == -2 D_LOGEND == 1, 60 corresponds to shell width of 0.05 dex
    M_TOL = np.float32(1e-2)
    N_WALL = 100
    N_MIN = 10
    SNAP = '015'
    MASS_UNIT = 1e+10
    MIN_NUMBER_DM_PTCS = 200
    CENTER = 'mode'
    
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
    mass_array = np.ones((dm_xyz.shape[0],), dtype = np.float32)*mass_dm/MASS_UNIT # Has to be in unit mass (= 10^10 M_sun/h)
    idx_cat = [np.arange(len(halo_x), dtype = np.int32).tolist()]
    
    ########################### Define DensShapeProfs object #######################################
    cprofiles = DensShapeProfs(dm_xyz, mass_array, idx_cat, r_vir, SNAP, L_BOX, MIN_NUMBER_DM_PTCS, D_LOGSTART, D_LOGEND, D_BINS, M_TOL, N_WALL, N_MIN, CENTER)
    
    ######################### Calculating Morphological Properties #################################
    # Create halo shape catalogue
    cprofiles.dumpShapeCatLocal(CAT_DEST, reduced = False, shell_based = False)
    
    ######################################## Visualizations ########################################
    # Visualize halo: A sample output is shown above!
    cprofiles.vizLocalShapes(obj_numbers = [0], VIZ_DEST = VIZ_DEST, reduced = False, shell_based = False)

calcShapeEx()
