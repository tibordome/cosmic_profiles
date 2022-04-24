#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 18:23:43 2021
"""

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
import time
import numpy as np
start_time = time.time()
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.append(os.path.join(currentdir, '..', 'cosmic_shapes')) # Only needed if cosmic_shapes is not installed
from cosmic_shapes import CosmicShapesDirect, genAlphaBetaGammaHalo
import time
start_time = time.time()

#################################### Parameters ################################################
L_BOX = np.float32(10) # cMpc/h
CAT_DEST = "./cat"
VIZ_DEST = "./viz"
D_LOGSTART = -2
D_LOGEND = 0
D_BINS = 30 # If D_LOGSTART == -2 D_LOGEND == 1, 60 corresponds to shell width of 0.05 dex
M_TOL = np.float32(1e-2)
N_WALL = 100
N_MIN = 10
SNAP = '015'
MIN_NUMBER_DM_PTCS = 200

#################################### Generate 1 mock halo ######################################
rho_0 = 1
r_s = 1
alpha = 1
beta = 3
gamma = 1
N_bin = 200
a_max = 2
delta_a = 0.01
a = np.linspace(delta_a,a_max,N_bin)
assert np.allclose(a[2]-a[1], delta_a)
b = a*0.7
c = a*0.2

halo_x, halo_y, halo_z = genAlphaBetaGammaHalo(N_MIN, alpha, beta, gamma, rho_0, r_s, a, b, c)
dm_xyz = np.float32(np.hstack((np.reshape(halo_x, (halo_x.shape[0],1)), np.reshape(halo_y, (halo_y.shape[0],1)), np.reshape(halo_z, (halo_z.shape[0],1)))))

######################### Extract R_vir, halo indices and halo sizes ##########################
r_vir = np.array([a_max], dtype = np.float32)
mass_array = np.ones((dm_xyz.shape[0],), dtype = np.float32)
h_indices = [np.arange(len(halo_x), dtype = np.int32).tolist()]

########################### Define CosmicShapesDirect object ###################################
cshapes = CosmicShapesDirect(dm_xyz, mass_array, h_indices, r_vir, CAT_DEST, VIZ_DEST, SNAP, L_BOX, MIN_NUMBER_DM_PTCS, D_LOGSTART, D_LOGEND, D_BINS, M_TOL, N_WALL, N_MIN, start_time)

######################### Calculating Morphological Properties #################################
# Create halo shape catalogue
cshapes.calcLocalShapes()

######################################## Visualizations ########################################
# Visualize halo: The output is shown above!
cshapes.vizLocalShapes([0])