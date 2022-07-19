#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 18:23:43 2021
"""

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
import os
import sys
import inspect
import numpy as np
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.append(os.path.join(currentdir, '..')) # Only needed if cosmic_profiles is not installed
from cosmic_profiles import DensShapeProfsHDF5

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
WANT_RVIR = False # Whether or not we want quantities (e.g. D_LOGSTART) expressed with respect to the virial radius R_vir or the overdensity radius R_200
ROverR200 = np.logspace(-1.5,0,70)
HIST_NB_BINS = 11 # Number of bins used for e.g. ellipticity histogram
frac_r200 = 0.5 # At what depth to calculate e.g. histogram of triaxialities (cf. plotLocalTHist())

def HDF5Ex():
    
    # Define DensShapeProfsHDF5 object
    cprofiles = DensShapeProfsHDF5(HDF5_SNAP_DEST, HDF5_GROUP_DEST, SNAP, SNAP_MAX, L_BOX, MIN_NUMBER_DM_PTCS, MIN_NUMBER_STAR_PTCS, D_LOGSTART, D_LOGEND, D_BINS, M_TOL, N_WALL, N_MIN, CENTER, WANT_RVIR)
    
    ########################## Calculating Morphological Properties ###############################
    # Create halo shape catalogue
    cprofiles.dumpShapeCatLocal(CAT_DEST, reduced = False, shell_based = False, obj_type = 'dm')
    
    # Create global halo shape catalogue
    cprofiles.dumpShapeCatGlobal(CAT_DEST, reduced = False, obj_type = 'dm')
    
    # Create local gx shape catalogue
    cprofiles.dumpShapeCatLocal(CAT_DEST, reduced = False, shell_based = False, obj_type = 'gx')
    
    ######################################## Visualizations #######################################
    # Viz first few halos' shapes
    cprofiles.vizLocalShapes(obj_numbers = [0,1,2], VIZ_DEST = VIZ_DEST, reduced = False, shell_based = False, obj_type = 'dm')
    
    # Viz first few galaxies' shapes
    cprofiles.vizLocalShapes(obj_numbers = [0,1,2], VIZ_DEST = VIZ_DEST, reduced = False, shell_based = False, obj_type = 'gx')
    
    # Plot halo ellipticity histogram
    cprofiles.plotGlobalEpsHist(HIST_NB_BINS, VIZ_DEST, obj_type = 'dm')
    
    # Plot halo triaxiality histogram
    cprofiles.plotLocalTHist(HIST_NB_BINS, VIZ_DEST, frac_r200, reduced = False, shell_based = False, obj_type = 'dm')
    
    # Draw halo shape profiles (overall and mass-decomposed ones)
    cprofiles.plotShapeProfs(VIZ_DEST, reduced = False, shell_based = False, obj_type = 'dm')
    
    ########################## Calculating and Visualizing Density Profs ##########################
    # Create local halo density catalogue
    dens_profs = cprofiles.getDensProfsDirectBinning(ROverR200, obj_type = 'dm')
    
    # Fit density profiles
    best_fits = cprofiles.getDensProfsBestFits(dens_profs[:,25:], ROverR200[25:], 'nfw', obj_type = 'dm')
    
    # Draw halo density profiles (overall and mass-decomposed ones). The results from getDensProfsBestFits() got cached.
    cprofiles.plotDensProfs(dens_profs, ROverR200, dens_profs[:,25:], ROverR200[25:], 'nfw', VIZ_DEST, obj_type = 'dm')
    
HDF5Ex()
