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
subprocess.call(['python3', 'setup_compile.py', 'build_ext', '--inplace'], cwd=os.path.join(currentdir, '..', '..'))
subprocess.call(['mkdir', 'viz'], cwd=os.path.join(currentdir))
subprocess.call(['mkdir', 'cat'], cwd=os.path.join(currentdir))
sys.path.append(os.path.join(currentdir, '..', '..')) # Only needed if cosmic_profiles is not installed
from cosmic_profiles import genHalo, DensShapeProfs, updateInUnitSystem, updateOutUnitSystem
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def test_shapes():
    #################################### Parameters ################################################
    updateInUnitSystem(in_unit_length_in_cm = 3.085678e24, in_unit_mass_in_g = 1.989e33, in_unit_velocity_in_cm_per_s = 1e5)
    updateOutUnitSystem(out_unit_length_in_cm = 3.085678e24, out_unit_mass_in_g = 1.989e33, out_unit_velocity_in_cm_per_s = 1e5)
    L_BOX = np.float32(10) # Mpc/h
    VIZ_DEST = "./cosmic_profiles/tests/viz"
    D_LOGSTART = -2
    D_LOGEND = 0
    D_BINS = 30 # If D_LOGSTART == -2 D_LOGEND == 1, 60 corresponds to shell width of 0.05 dex
    IT_TOL = np.float32(1e-2)
    IT_WALL = 100
    IT_MIN = 10
    SNAP = '015'
    MIN_NUMBER_DM_PTCS = 1000
    CENTER = 'mode'
    HIST_NB_BINS = 11 # Number of bins used for e.g. ellipticity histogram
    frac_r200 = 0.5 # At what depth to calculate e.g. histogram of triaxialities (cf. plotLocalTHist())
    N = 10 # Number of halos used for test
    
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
    cprofiles = DensShapeProfs(dm_xyz, mass_array, idx_cat_in, r_vir, SNAP, L_BOX, MIN_NUMBER_DM_PTCS, D_LOGSTART, D_LOGEND, D_BINS, IT_TOL, IT_WALL, IT_MIN, CENTER)
    
    idx_cat = cprofiles.getIdxCat()[0]
    halos_select = [0, 5]
    centers, ms = cprofiles.getMassesCenters(halos_select)
    if rank == 0:
        assert len(idx_cat) <= len(idx_cat_in) # Note: idx_cat only contains objects with sufficient resolution
        assert centers.shape == (halos_select[1] - halos_select[0] + 1,3)
        assert ms.shape == (halos_select[1] - halos_select[0] + 1,)
    
    ######################### Calculating Local Morphological Properties #############################
    # Create halo shape catalogue
    d, q, s, minor, inter, major, obj_centers, obj_masses = cprofiles.getShapeCatLocal(select = halos_select, reduced = False, shell_based = False)
    
    if rank == 0:
        nb_suff_res = halos_select[1]-halos_select[0]+1
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
    cprofiles.plotShapeProfs(nb_bins = 2, VIZ_DEST = VIZ_DEST, select = halos_select, reduced = True, shell_based = True)
    
    # Viz first few halos' shapes
    cprofiles.vizLocalShapes(obj_numbers = [0,1,2], VIZ_DEST = VIZ_DEST, reduced = False, shell_based = False)
    
    # Plot halo triaxiality histogram
    cprofiles.plotLocalTHist(HIST_NB_BINS, VIZ_DEST, frac_r200, select = halos_select, reduced = False, shell_based = False)
    
    ######################### Calculating Global Morphological Properties ############################
    halos_select = [0, N-1]
    d, q, s, minor, inter, major, obj_centers, obj_masses = cprofiles.getShapeCatGlobal(select = halos_select, reduced = False)
    
    if rank == 0:
        nb_suff_res = halos_select[1]-halos_select[0]+1
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
    cprofiles.plotGlobalEpsHist(HIST_NB_BINS, VIZ_DEST, select = halos_select)
    
    # Viz first few halos' shapes
    cprofiles.vizGlobalShapes(obj_numbers = [0,1,2], VIZ_DEST = VIZ_DEST, reduced = False)