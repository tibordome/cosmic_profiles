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
sys.path.append(os.path.join(currentdir, '..', 'cosmic_shapes')) # Only needed if cosmic_shapes is not installed
from cosmic_shapes import CosmicShapesGadgetHDF5
import time
start_time = time.time()

# Parameters
L_BOX = np.float32(10) # cMpc/h
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
MIN_NUMBER_DM_PTCS = 200
MIN_NUMBER_STAR_PTCS = 100

# Define CosmicShapes object
cshapes = CosmicShapesGadgetHDF5(HDF5_SNAP_DEST, HDF5_GROUP_DEST, CAT_DEST, VIZ_DEST, SNAP, SNAP_MAX, L_BOX, MIN_NUMBER_DM_PTCS, MIN_NUMBER_STAR_PTCS, D_LOGSTART, D_LOGEND, D_BINS, M_TOL, N_WALL, N_MIN, start_time)

########################## Calculating Morphological Properties ###############################
# Retrieve HDF5 halo catalogue, extract CSHs
cshapes.loadDMCat()

# Create local halo shape catalogue
cshapes.calcLocalShapesDM()

# Create global halo shape catalogue
cshapes.calcGlobalShapesDM()

# Retrieve/Construct gx catalogue
cshapes.loadGxCat()

# Create local gx shape catalogue
cshapes.calcLocalShapesGx()

######################################## Visualizations #######################################
# Viz first few halos' shapes
cshapes.vizLocalShapes([0,1,2], obj_type = 'dm')

# Viz first few galaxies' shapes
cshapes.vizLocalShapes([0,1,2], obj_type = 'gx')

# Plot halo ellipticity histogram
cshapes.plotGlobalEpsHisto(obj_type = 'dm')

# Plot halo triaxiality histogram
cshapes.plotLocalTHisto(obj_type = 'dm')

# Draw halo shape curves (averaging over halos' shape curves)
cshapes.drawShapeCurves(obj_type = 'dm')