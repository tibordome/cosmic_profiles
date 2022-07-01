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
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 13})
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.append(os.path.join(currentdir, '..')) # Only needed if cosmic_profiles is not installed
from cosmic_profiles import CosmicProfilesDirect, genHalo, getEinastoProf
import time
start_time = time.time()

def calcDensEx():
    
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
    MASS_UNIT = 1e+10
    MIN_NUMBER_DM_PTCS = 200
    CENTER = 'com'
    
    #################################### Generate 1 mock halo ######################################
    tot_mass = 10**(12) # M_sun/h
    halo_res = 600000
    r_s = 0.5 # Units are Mpc/h
    alpha = 0.18
    N_bin = 100
    r_vir = np.array([1.0], dtype = np.float32) # Units are Mpc/h
    a = np.logspace(-2,0.2,N_bin)*r_vir[0] # Units are Mpc/h
    b = a # Units are Mpc/h. The more b differs from a, the more biased the spherically averaged density profile.
    c = a # Units are Mpc/h
    rho_res = 50
    r_over_rvir = np.logspace(-2,0,rho_res) # Estimate density profile out to the virial radius.
    
    model_pars = np.array([r_s, alpha])
    halo_x, halo_y, halo_z, mass_dm, rho_0 = genHalo(tot_mass, halo_res, model_pars, 'einasto', a, b, c)
    halo_x += L_BOX/2 # Move mock halo into the middle of the simulation box
    halo_y += L_BOX/2
    halo_z += L_BOX/2
    print("Number of particles in the halo is {}. Mass of each DM ptc is {:.2e} M_sun/h, the average halo mass density is {:.2e} M_sun*h^2/(Mpc^3).".format(halo_x.shape[0], mass_dm, mass_dm*halo_x.shape[0]/(4/3*np.pi*r_vir[0]**3)))
    dm_xyz = np.float32(np.hstack((np.reshape(halo_x, (halo_x.shape[0],1)), np.reshape(halo_y, (halo_y.shape[0],1)), np.reshape(halo_z, (halo_z.shape[0],1)))))
    
    ######################### Extract halo indices and halo sizes ##########################
    mass_array = np.ones((dm_xyz.shape[0],), dtype = np.float32)*mass_dm/MASS_UNIT # Has to be in unit mass (= 10^10 M_sun/h)
    h_indices = [np.arange(len(halo_x), dtype = np.int32).tolist()]
    
    ########################### Define CosmicProfilesDirect object ###################################
    cprofiles = CosmicProfilesDirect(dm_xyz, mass_array, h_indices, r_vir, CAT_DEST, VIZ_DEST, SNAP, L_BOX, MIN_NUMBER_DM_PTCS, D_LOGSTART, D_LOGEND, D_BINS, M_TOL, N_WALL, N_MIN, CENTER, start_time)
    
    ############################## Estimate Density Profile ########################################
    # Visualize density profile: A sample output is shown above!
    cprofiles.calcDensProfsDirectBinning(r_over_rvir)
    cprofiles.calcDensProfsKernelBased(r_over_rvir)
    dens_profs_db, r_over_rvir = cprofiles.fetchDensProfsDirectBinning() # dens_profs_db is in M_sun*h^2/Mpc^3
    dens_profs_kb, r_over_rvir = cprofiles.fetchDensProfsKernelBased()
    plt.figure()
    plt.loglog(r_over_rvir, dens_profs_db, 'o--', label='direct binning', markersize = 3)
    plt.loglog(r_over_rvir, dens_profs_kb, 'o--', label='kernel-based', markersize = 3)
    plt.loglog(r_over_rvir, getEinastoProf(r_over_rvir*r_vir[0], np.array([rho_0, r_s, alpha])), lw = 1.0, label=r'Einasto-target: $\alpha$ = {:.2f}, $r_s$ = {:.2f} cMpc/h'.format(alpha, r_s))
    plt.xlabel(r'r/$R_{\mathrm{vir}}$')
    plt.ylabel(r"$\rho$ [$h^2M_{{\odot}}$ / Mpc${{}}^3$]")
    plt.legend(fontsize="small", loc='lower left')
    plt.savefig('{}/RhoProfObj0_{}.pdf'.format(VIZ_DEST, SNAP), bbox_inches='tight')
    
    ############################## Fit Density Profile #############################################
    r_over_rvir = r_over_rvir[10:] # Do not fit innermost region since not reliable in practice. Use gravitational softening scale and / or relaxation timescale to estimate inner convergence radius.
    dens_profs_db = dens_profs_db[10:]
    cprofiles.fitDensProfs(dens_profs_db.reshape((1,dens_profs_db.shape[0])), r_over_rvir, cprofiles.fetchCat(), r_vir, method = 'einasto')
    best_fits = cprofiles.fetchDensProfsBestFits('einasto')
    best_fit = best_fits[0]
    plt.figure()
    plt.loglog(r_over_rvir, dens_profs_db, 'o--', label='density profile', markersize = 4)
    plt.loglog(r_over_rvir, getEinastoProf(r_over_rvir*r_vir[0], np.array([best_fit[0], best_fit[1], best_fit[2]])), '--', color = 'r', label=r'Einasto-fit')
    plt.xlabel(r'r/$R_{\mathrm{vir}}$')
    plt.ylabel(r"$\rho$ [$h^2M_{{\odot}}$ / Mpc${{}}^3$]")
    plt.legend(fontsize="small", bbox_to_anchor=(0.95, 1), loc='upper right')
    plt.savefig('{}/RhoProfFitObj0_{}.pdf'.format(VIZ_DEST, SNAP), bbox_inches='tight')