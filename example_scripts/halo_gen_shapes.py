#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 18:23:43 2021
"""

from mpi4py import MPI
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 13})
comm = MPI.COMM_WORLD
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot
rank = comm.Get_rank()
size = comm.Get_size()
import time
import numpy as np
start_time = time.time()
import matplotlib
matplotlib.rcParams.update({'font.size': 13})
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.append(os.path.join(currentdir, '..')) # Only needed if cosmic_shapes is not installed
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
N0 = 1
N_MIN = 10
SNAP = '015'
MIN_NUMBER_DM_PTCS = 200
CENTER = 'mode'

#################################### Generate 1 mock halo ######################################
rho_0 = 10**(13) # Units are M_sun*h^2/Mpc^3
r_s = 0.5 # Units are Mpc/h
alpha = 5
beta = 3
gamma = 1.1
N_bin = 100
a_max = 1 # Units are Mpc/h
a = np.logspace(-1.5,0,N_bin)*a_max # Units are Mpc/h
b = a*0.2 # Units are Mpc/h
c = a*0.2 # Units are Mpc/h

halo_x, halo_y, halo_z, mass_dm = genAlphaBetaGammaHalo(N0, alpha, beta, gamma, rho_0, r_s, a, b, c)
print("Number of particles in the halo is {}.".format(halo_x.shape[0]))
halo_x += L_BOX/2 # Move mock halo into the middle of the simulation box
halo_y += L_BOX/2
halo_z += L_BOX/2
dm_xyz = np.float32(np.hstack((np.reshape(halo_x, (halo_x.shape[0],1)), np.reshape(halo_y, (halo_y.shape[0],1)), np.reshape(halo_z, (halo_z.shape[0],1)))))

fig = pyplot.figure()
ax = Axes3D(fig, auto_add_to_figure = False)
fig.add_axes(ax)
ax.scatter(dm_xyz[:,0],dm_xyz[:,1],dm_xyz[:,2],s=0.3, label = "Particles")
ax.scatter(L_BOX/2, L_BOX/2, L_BOX/2,s=50,c="r", label = "COM")
plt.legend(bbox_to_anchor=(0.95, 1), loc='upper right')
plt.savefig('scatter_plot_test.pdf')

######################### Extract R_vir, halo indices and halo sizes ##########################
r_vir = np.array([a_max], dtype = np.float32)
mass_array = np.ones((dm_xyz.shape[0],), dtype = np.float32)*mass_dm
h_indices = [np.arange(len(halo_x), dtype = np.int32).tolist()]

########################### Define CosmicShapesDirect object ###################################
cshapes = CosmicShapesDirect(dm_xyz, mass_array, h_indices, r_vir, CAT_DEST, VIZ_DEST, SNAP, L_BOX, MIN_NUMBER_DM_PTCS, D_LOGSTART, D_LOGEND, D_BINS, M_TOL, N_WALL, N_MIN, CENTER, start_time)

######################### Calculating Morphological Properties #################################
# Create halo shape catalogue
cshapes.calcLocalShapes()

######################################## Visualizations ########################################
# Visualize halo: A sample output is shown above!
cshapes.vizLocalShapes([0])