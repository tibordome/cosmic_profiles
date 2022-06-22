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
start_time = time.time()
import os
import sys
import inspect
import numpy as np
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.append(os.path.join(currentdir, '..')) # Only needed if cosmic_profiles is not installed
from cosmic_profiles import CosmicProfilesGadgetHDF5
import time
start_time = time.time()

# Parameters
L_BOX = np.float32(10) # Mpc/h
CAT_DEST = "./cat"
HDF5_GROUP_DEST = "../../code_git/example_snapshot/LLGas256b20/groups_035"
HDF5_SNAP_DEST = "../../code_git/example_snapshot/LLGas256b20/snapdir_035"
VIZ_DEST = "./viz"
SNAP_MAX = 16
D_LOGSTART = -2
D_LOGEND = 0.5
D_BINS = 30 # If D_LOGSTART == -2 D_LOGEND == 1, 60 corresponds to shell width of 0.05 dex
M_TOL = np.float32(1e-2)
N_WALL = 100
N_MIN = 10
SNAP = '035'
CENTER = 'mode'
MIN_NUMBER_DM_PTCS = 200
MIN_NUMBER_STAR_PTCS = 100
ROverR200 = np.logspace(-1.5,0,70)

# Define CosmicProfiles object
cprofiles = CosmicProfilesGadgetHDF5(HDF5_SNAP_DEST, HDF5_GROUP_DEST, CAT_DEST, VIZ_DEST, SNAP, SNAP_MAX, L_BOX, MIN_NUMBER_DM_PTCS, MIN_NUMBER_STAR_PTCS, D_LOGSTART, D_LOGEND, D_BINS, M_TOL, N_WALL, N_MIN, CENTER, start_time)

########################## Calculating Morphological Properties ###############################
# Retrieve HDF5 halo catalogue, extract CSHs
cprofiles.loadDMCat()

# Create local halo shape catalogue
cprofiles.calcLocalShapesDM()

# Create global halo shape catalogue
cprofiles.calcGlobalShapesDM()

# Retrieve/Construct gx catalogue
cprofiles.loadGxCat()

# Create local gx shape catalogue
cprofiles.calcLocalShapesGx()

######################################## Visualizations #######################################
# Viz first few halos' shapes
cprofiles.vizLocalShapes([0,1,2], obj_type = 'dm')

# Viz first few galaxies' shapes
cprofiles.vizLocalShapes([0,1,2], obj_type = 'gx')

# Plot halo ellipticity histogram
cprofiles.plotGlobalEpsHisto(obj_type = 'dm')

# Plot halo triaxiality histogram
cprofiles.plotLocalTHisto(obj_type = 'dm')

# Draw halo shape profiles (overall and mass-decomposed ones)
cprofiles.drawShapeProfiles(obj_type = 'dm')

########################## Calculating and Visualizing Density Profiles #######################
# Create local halo density catalogue
cprofiles.calcDensProfsDirectBinning(ROverR200, obj_type = 'dm')

# Fit density profiles
cprofiles.fitDensProfs(cprofiles.fetchDensProfsDirectBinning()[0][:,25:], cprofiles.fetchDensProfsDirectBinning()[1][25:], cprofiles.fetchHaloCat(), cprofiles.fetchR200s(), 'nfw')

# Draw halo density profiles (overall and mass-decomposed ones)
cprofiles.drawDensityProfiles(cprofiles.fetchDensProfsDirectBinning()[0][:,25:], cprofiles.fetchDensProfsDirectBinning()[1][25:], cprofiles.fetchHaloCat(), cprofiles.fetchR200s(), 'nfw', obj_type = 'dm')
