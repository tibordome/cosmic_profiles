#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
import time
import numpy as np
start_time = time.time()
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
sys.path.append(os.path.join(currentdir, '..', '..', 'example_scripts'))
from halo_gen_shapes import calcShapeEx
from halo_gen_dens_prof import calcDensEx
import time
start_time = time.time()

#################################### Parameters ################################################
L_BOX = np.float32(10) # Mpc/h
CAT_DEST = "./cat"
VIZ_DEST = "./viz"
D_LOGSTART = -2
D_LOGEND = 0
D_BINS = 30 # If D_LOGSTART == -2 D_LOGEND == 1, 60 corresponds to shell width of 0.05 dex
M_TOL = np.float32(1e-2)
N_WALL = 100
N_MIN = 10
SNAP = '015'
MASS_UNIT = 1e+10
MIN_NUMBER_DM_PTCS = 200
CENTER = 'mode'

def test_ex_scripts():
    
    calcShapeEx()
    calcDensEx()
    
    subprocess.call(['bash', 'decythonize.sh'], cwd=os.path.join(currentdir, '..'))
