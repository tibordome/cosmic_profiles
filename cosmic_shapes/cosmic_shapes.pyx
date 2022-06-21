#cython: language_level=3
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 13:17:51 2021
"""

import numpy as np
from libc.math cimport sqrt, pi
from copy import deepcopy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot
from numpy.random import default_rng
from matplotlib.font_manager import FontProperties
import matplotlib
matplotlib.rcParams.update({'font.size': 13})
cimport openmp
from libc.math cimport isnan
from cython.parallel import prange
from libc.stdio cimport printf
import json
import h5py
from pathos.multiprocessing import ProcessingPool as Pool
from functools import partial
cimport cython
import os
from mpl_toolkits.mplot3d import Axes3D
from cosmic_shapes.python_helpers import print_status, set_axes_equal, fibonacci_ellipsoid, drawUniformFromShell, getMassDMParticle, respectPBCNoRef, getCoM, findMode
from cosmic_shapes.shape_analysis import getGlobalEpsHisto, getLocalTHisto, getShapeProfiles, getDensityProfiles, fitDensProfHelper
from cosmic_shapes.get_hdf5 import getHDF5Data, getHDF5GxData, getHDF5SHDMData, getHDF5SHGxData, getHDF5DMData
from cosmic_shapes.gen_csh_gx_cat import getCSHCat, getGxCat
from cosmic_shapes.cython_helpers cimport CythonHelpers
from nbodykit.lab import cosmology, LogNormalCatalog
from pynverse import inversefunc
from scipy.integrate import quad
import math
import time
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

@cython.embedsignature(True)
def createLogNormUni(BoxSize, nbar, redshift, Nmesh, UNIT_MASS):
    
    # Generating LogNormal Catalog
    redshift = redshift
    cosmo = cosmology.Planck15
    Plin = cosmology.LinearPower(cosmo, redshift, transfer='EisensteinHu')
    
    cat = LogNormalCatalog(Plin=Plin, nbar=nbar, BoxSize=BoxSize, Nmesh=Nmesh, bias=2.0, seed=42)
    x_vec = np.float32(np.array(cat['Position'][:,0])) # Mpc/h
    y_vec = np.float32(np.array(cat['Position'][:,1]))
    z_vec = np.float32(np.array(cat['Position'][:,2]))
    
    x_vel = np.float32(np.array(cat['Velocity'][:,0]))
    y_vel = np.float32(np.array(cat['Velocity'][:,1]))
    z_vel = np.float32(np.array(cat['Velocity'][:,2]))
    
    N = int(round(len(x_vec)**(1/3)))
    N_tot = len(x_vec)
    dm_mass = getMassDMParticle(N, BoxSize)/UNIT_MASS
    return N_tot, x_vec, y_vec, z_vec, x_vel, y_vel, z_vel, np.ones((len(x_vec),),dtype = np.float32)*dm_mass

@cython.embedsignature(True)
def getAlphaBetaGammaProf(r, rho_0, alpha, beta, gamma, r_s):
    
    return rho_0/((r/r_s)**gamma*(1+(r/r_s)**alpha)**((beta-gamma)/alpha))

@cython.embedsignature(True)
def getEinastoProf(r, rho_2, r_2, alpha):
    
    return rho_2*np.exp(-2/alpha*((r/r_2)**alpha-1))

@cython.embedsignature(True)
def getNFWProf(r, rho_s, r_s):
    
    return rho_s/((r/r_s)*(1+r/r_s)**2)

@cython.embedsignature(True)
def getHernquistProf(r, rho_s, r_s):
    
    return rho_s/((r/r_s)*(1+r/r_s)**3)
    
@cython.embedsignature(True)
def genHalo(tot_mass, res, model_pars, method, a, b, c):
    
    # Determine rho_0 in units of M_sun*h^2/Mpc^3
    def getMassIntegrand0(r, method, model_pars):
        if method == 'einasto':
            r_2, alpha = model_pars
            return 4*np.pi*r**2*np.exp(-2/alpha*((r/r_2)**alpha-1))
        if method == 'alpha_beta_gamma':
            alpha, beta, gamma, r_s = model_pars
            return 4*np.pi*r**2/((r/r_s)**gamma*(1+(r/r_s)**alpha)**((beta-gamma)/alpha))
        if method == 'hernquist':
            r_s = model_pars
            return 4*np.pi*r**2/((r/r_s)*(1+r/r_s)**3)
        else:
            r_s = model_pars
            return 4*np.pi*r**2/((r/r_s)*(1+r/r_s)**2)
    rho_0 = tot_mass/quad(getMassIntegrand0, 1e-8, a[-1], args=(method, model_pars))[0]
    model_pars = np.hstack((np.array([rho_0]),model_pars))
    
    # Determine number of particles in second shell (first proper shell)
    def getMassIntegrand(r, model_pars):
        if method == 'einasto':
            rho_2, r_2, alpha = model_pars
            return 4*np.pi*r**2*getEinastoProf(r, rho_2, r_2, alpha)
        if method == 'alpha_beta_gamma':
            rho_0, alpha, beta, gamma, r_s = model_pars
            return 4*np.pi*r**2*getAlphaBetaGammaProf(r, rho_0, alpha, beta, gamma, r_s)
        if method == 'hernquist':
            rho_s, r_s = model_pars
            return 4*np.pi*r**2*getHernquistProf(r, rho_s, r_s)
        else:
            rho_s, r_s = model_pars
            return 4*np.pi*r**2*getNFWProf(r, rho_s, r_s)
    
    # Determine number of particles in second shell
    mass_1 = quad(getMassIntegrand, a[0], a[1], args=(model_pars))[0] # Mass in shell 1 in units of M_sun/h
    N1 = int(round(res*mass_1/tot_mass)) # Number of particles in shell 1. Rounding error is unavoidable.
    
    # Build up halo successively in onion-like fashion
    halo_x = np.empty(0)
    halo_y = np.empty(0)
    halo_z = np.empty(0)
    Nptc = np.zeros((a.shape[0],), dtype = np.int32) # Note: Number of ptcs in shell Shell(0,a[idx-1],b[idx-1],c[idx-1],a[idx],b[idx],c[idx])
    for shell in range(a.shape[0]):
        if shell != 0:
            Nptc[shell] = int(round(N1*quad(getMassIntegrand, a[shell-1], a[shell], args=(model_pars))[0]/quad(getMassIntegrand, a[0], a[1], args=(model_pars))[0]))
        else:
            Nptc[shell] = int(round(N1*quad(getMassIntegrand, 1e-8, a[0], args=(model_pars))[0]/quad(getMassIntegrand, a[0], a[1], args=(model_pars))[0]))
    with Pool(processes=openmp.omp_get_max_threads()) as pool:
        results = pool.map(partial(drawUniformFromShell, 3, a, b, c, Nptc), [idx for idx in range(Nptc.shape[0])]) # Draws from Shell(0,a[idx-1],b[idx-1],c[idx-1],a[idx],b[idx],c[idx])
    for result in results:
        halo_x = np.hstack((halo_x, result[:,0]))
        halo_y = np.hstack((halo_y, result[:,1]))
        halo_z = np.hstack((halo_z, result[:,2]))
    mass_dm = quad(getMassIntegrand, 1e-8, a[-1], args=(model_pars))[0]/halo_z.shape[0]    
    return halo_x, halo_y, halo_z, mass_dm, rho_0

@cython.embedsignature(True)
cdef class CosmicShapes:
    cdef str CAT_DEST
    cdef str VIZ_DEST
    cdef str SNAP
    cdef float L_BOX
    cdef int MIN_NUMBER_PTCS
    cdef int D_LOGSTART
    cdef int D_LOGEND
    cdef int D_BINS
    cdef float M_TOL
    cdef int N_WALL
    cdef int N_MIN
    cdef str CENTER
    cdef float SAFE # Units: cMpc/h. Ellipsoidal radius will be maxdist(COM,point)+SAFE where point is any point in the point cloud. The larger the better.
    cdef double start_time
    
    def __init__(self, str CAT_DEST, str VIZ_DEST, str SNAP, float L_BOX, int MIN_NUMBER_PTCS, int D_LOGSTART, int D_LOGEND, int D_BINS, float M_TOL, int N_WALL, int N_MIN, str CENTER, double start_time):
        
        self.CAT_DEST = CAT_DEST
        self.VIZ_DEST = VIZ_DEST
        self.SNAP = SNAP
        self.L_BOX = L_BOX
        self.MIN_NUMBER_PTCS = MIN_NUMBER_PTCS
        self.D_LOGSTART = D_LOGSTART
        self.D_LOGEND = D_LOGEND
        self.D_BINS = D_BINS
        self.M_TOL = M_TOL
        self.N_WALL = N_WALL
        self.N_MIN = N_MIN
        self.CENTER = CENTER
        self.SAFE = 6
        self.start_time = start_time
    
    def getDensProfsDirectBinning(self, cat, float[:,:] xyz, int[:] obj_keep, float[:] masses, float[:] r200s, float[:] ROverR200, float L_BOX, str CENTER):
        
        cdef int nb_objs = len(cat)
        # Determine endpoints of radial bins
        cdef float[:] rad_bins = np.hstack(([np.float32(1e-8), (ROverR200.base[:-1] + ROverR200.base[1:])/2., ROverR200.base[-1]])) # Length = ROverR200.shape[0]+1
        cdef int r_res = ROverR200.shape[0]
        cdef int p
        cdef int n
        cdef int[:] obj_size = np.zeros((nb_objs,), dtype = np.int32)
        cdef int[:] obj_pass = np.zeros((nb_objs,), dtype = np.int32)
        for p in range(nb_objs):
            if cat[p] != []: # Only add objects that have sufficient resolution
                obj_size[p] = len(cat[p])
                obj_pass[p] = 1
            else:
                if obj_keep[p] == 1:
                    raise ValueError("Having a 1 in obj_keep for an object for which cat is empty is not allowed.")
        cdef int nb_pass = np.sum(obj_pass.base)
        cdef int nb_keep = np.sum(obj_keep.base)
        if nb_objs == 0:
            return ROverR200.base, np.zeros((0,r_res), dtype = np.float32), np.zeros((0,r_res), dtype = np.float32), np.zeros((0,r_res), dtype = np.float32), np.zeros((0,r_res), dtype = np.float32)
        # Transform cat to int[:,:]
        cdef int[:,:] cat_arr = np.zeros((nb_pass,np.max([len(cat[p]) for p in range(nb_objs)])), dtype = np.int32)
        cdef int[:] idxs_compr = np.zeros((nb_objs,), dtype = np.int32)
        idxs_compr.base[obj_pass.base.nonzero()[0]] = np.arange(np.sum(obj_pass.base))
        for p in range(nb_objs):
            if obj_pass[p] == 1:
                cat_arr.base[idxs_compr[p],:obj_size[p]] = np.array(cat[p])
        # Define memoryviews
        cdef int[:] idxs_keep = np.zeros((nb_objs,), dtype = np.int32)
        idxs_keep.base[obj_keep.base.nonzero()[0]] = np.arange(np.sum(obj_keep.base))
        cdef float[:,:] dens_profs = np.zeros((nb_keep, r_res), dtype = np.float32)
        cdef float[:,:] Mencls = np.zeros((nb_keep, r_res), dtype = np.float32)
        cdef float[:,:] rad_bins_tiled = np.reshape(np.tile(rad_bins.base, reps = openmp.omp_get_max_threads()), (openmp.omp_get_max_threads(), r_res+1))
        cdef int[:,:] shell = np.zeros((openmp.omp_get_max_threads(), cat_arr.shape[1]), dtype = np.int32)
        cdef float[:,:,:] xyz_obj = np.zeros((openmp.omp_get_max_threads(), cat_arr.shape[1], 3), dtype = np.float32)
        cdef float[:,:] m_obj = np.zeros((openmp.omp_get_max_threads(), cat_arr.shape[1]), dtype = np.float32)
        cdef float[:,:] centers = np.zeros((nb_pass,3), dtype = np.float32)
        for p in range(nb_objs): # Calculate centers of objects
            if obj_pass[p] == 1:
                xyz_ = respectPBCNoRef(xyz.base[cat_arr.base[idxs_compr[p],:obj_size[p]]], L_BOX)
                if CENTER == 'mode':
                    centers.base[p] = findMode(xyz_, masses.base[cat_arr.base[idxs_compr[p],:obj_size[p]]], 1000)
                else:
                    centers.base[p] = getCoM(xyz_, masses.base[cat_arr.base[idxs_compr[p],:obj_size[p]]])
        for p in prange(nb_objs, schedule = 'dynamic', nogil = True):
            if obj_keep[p] == 1:
                for n in range(obj_size[p]):
                    xyz_obj[openmp.omp_get_thread_num(),n] = xyz[cat_arr[idxs_compr[p],n]]
                    m_obj[openmp.omp_get_thread_num(),n] = masses[cat_arr[idxs_compr[p],n]]
                xyz_obj[openmp.omp_get_thread_num(),:obj_size[p]] = CythonHelpers.respectPBCNoRef(xyz_obj[openmp.omp_get_thread_num(),:obj_size[p]], L_BOX)
                dens_profs[idxs_keep[p]] = CythonHelpers.getDensProfBruteForce(xyz_obj[openmp.omp_get_thread_num(),:obj_size[p]], m_obj[openmp.omp_get_thread_num(),:obj_size[p]], centers[idxs_compr[p]], r200s[p], rad_bins_tiled[openmp.omp_get_thread_num()], dens_profs[idxs_keep[p]], shell[openmp.omp_get_thread_num()])
        
        return ROverR200.base, dens_profs.base
    
    def getDensProfsKernelBased(self, cat, float[:,:] xyz, int[:] obj_keep, float[:] masses, float[:] r200s, float[:] ROverR200, float L_BOX, str CENTER):
        
        cdef int nb_objs = len(cat)
        cdef int r_res = ROverR200.shape[0]
        cdef int p
        cdef int r_idx
        cdef int n
        cdef int[:] obj_size = np.zeros((nb_objs,), dtype = np.int32)
        cdef int[:] obj_pass = np.zeros((nb_objs,), dtype = np.int32)
        for p in range(nb_objs):
            if cat[p] != []: # Only add objects that have sufficient resolution
                obj_size[p] = len(cat[p])
                obj_pass[p] = 1
            else:
                if obj_keep[p] == 1:
                    raise ValueError("Having a 1 in obj_keep for an object for which cat is empty is not allowed.")
        cdef int nb_pass = np.sum(obj_pass.base)
        cdef int nb_keep = np.sum(obj_keep.base)
        if nb_objs == 0:
            return ROverR200.base, np.zeros((0,r_res), dtype = np.float32), np.zeros((0,r_res), dtype = np.float32), np.zeros((0,r_res), dtype = np.float32), np.zeros((0,r_res), dtype = np.float32)
        # Transform cat to int[:,:]
        cdef int[:,:] cat_arr = np.zeros((nb_pass,np.max([len(cat[p]) for p in range(nb_objs)])), dtype = np.int32)
        cdef int[:] idxs_compr = np.zeros((nb_objs,), dtype = np.int32)
        idxs_compr.base[obj_pass.base.nonzero()[0]] = np.arange(np.sum(obj_pass.base))
        for p in range(nb_objs):
            if obj_pass[p] == 1:
                cat_arr.base[idxs_compr[p],:obj_size[p]] = np.array(cat[p])
        # Define memoryviews
        cdef int[:] idxs_keep = np.zeros((nb_objs,), dtype = np.int32)
        idxs_keep.base[obj_keep.base.nonzero()[0]] = np.arange(np.sum(obj_keep.base))
        cdef float[:,:] dens_profs = np.zeros((nb_keep, r_res), dtype = np.float32)
        cdef float[:,:] Mencls = np.zeros((nb_keep, r_res), dtype = np.float32)
        cdef int[:] shell = np.zeros((cat_arr.shape[1]), dtype = np.int32)
        cdef float[:,:] xyz_obj = np.zeros((cat_arr.shape[1], 3), dtype = np.float32)
        cdef float[:] m_obj = np.zeros((cat_arr.shape[1]), dtype = np.float32)
        cdef float[:,:] centers = np.zeros((nb_pass,3), dtype = np.float32)
        cdef float[:] dists = np.zeros((cat_arr.shape[1],), dtype = np.float32) # Distances from center of halo
        cdef float[:] hs = np.zeros((cat_arr.shape[1],), dtype = np.float32) # Kernel widths
        for p in range(nb_objs): # Calculate centers of objects
            if obj_pass[p] == 1:
                xyz_ = respectPBCNoRef(xyz.base[cat_arr.base[idxs_compr[p],:obj_size[p]]], L_BOX)
                if CENTER == 'mode':
                    centers.base[p] = findMode(xyz_, masses.base[cat_arr.base[idxs_compr[p],:obj_size[p]]], 1000)
                else:
                    centers.base[p] = getCoM(xyz_, masses.base[cat_arr.base[idxs_compr[p],:obj_size[p]]])
            if obj_keep[p] == 1:
                for n in range(obj_size[p]):
                    xyz_obj[n] = xyz[cat_arr[idxs_compr[p],n]]
                    m_obj[n] = masses[cat_arr[idxs_compr[p],n]]
                xyz_obj[:obj_size[p]] = CythonHelpers.respectPBCNoRef(xyz_obj[:obj_size[p]], L_BOX)
                dists = np.linalg.norm(xyz_obj.base[:obj_size[p]]-centers.base[idxs_compr[p]], axis = 1)
                hs = 0.005*ROverR200[-1]*r200s[p]*(dists.base/(ROverR200[-1]*r200s[p]))**(0.5)
                for r_idx in prange(r_res, schedule = 'dynamic', nogil = True):
                    for n in range(obj_size[p]):
                        dens_profs[idxs_keep[p]][r_idx] += m_obj[n]*CythonHelpers.getKTilde(ROverR200[r_idx]*r200s[p], dists[n], hs[n])/(hs[n]**3)
        
        return ROverR200.base, dens_profs.base
    
    cdef float[:] runS1(self, float[:] morph_info, float[:,:] xyz, float[:,:] xyz_princ, float[:] masses, int[:] shell, float[:] center, complex[::1,:] shape_tensor, double[::1] eigval, complex[::1,:] eigvec, float d, float delta_d, float M_TOL, int N_WALL, int N_MIN) nogil:
        
        shell[:] = 0
        cdef int pts_in_shell = 0
        cdef int corr = 0
        cdef float err = 1.0
        cdef float q_new = 1.0
        cdef float s_new = 1.0
        cdef float q_old = 1.0
        cdef float s_old = 1.0
        cdef int iteration = 1
        cdef float vec2_norm = 1.0
        cdef float vec1_norm = 1.0
        cdef float vec0_norm = 1.0
        cdef int i
        # Start with spherical shell
        for i in range(xyz.shape[0]):
            if (center[0]-xyz[i,0])**2+(center[1]-xyz[i,1])**2+(center[2]-xyz[i,2])**2 < d**2 and (center[0]-xyz[i,0])**2+(center[1]-xyz[i,1])**2+(center[2]-xyz[i,2])**2 >= (d-delta_d)**2:
                shell[i-corr] = i
                pts_in_shell += 1
            else:
                corr += 1
        while (err > M_TOL):
            if iteration > N_WALL:
                morph_info[:] = 0.0
                return morph_info
            if pts_in_shell < N_MIN:
                morph_info[:] = 0.0
                return morph_info
            # Get shape tensor
            shape_tensor = CythonHelpers.getShapeTensor(xyz, shell, shape_tensor, masses, center, pts_in_shell)
            # Diagonalize shape_tensor
            eigvec[:,:] = 0.0
            eigval[:] = 0.0
            CythonHelpers.ZHEEVR(shape_tensor[:,:], &eigval[0], eigvec, 3)
            q_old = q_new; s_old = s_new
            q_new = sqrt(eigval[1]/eigval[2])
            s_new = sqrt(eigval[0]/eigval[2]) # It is assumed that eigenvalues are approximately proportional to a^2 etc. (true for uniform ellipsoid or uniform shell), though I have never seen any proof..
            err = max(CythonHelpers.cython_abs(q_new - q_old)/q_old, CythonHelpers.cython_abs(s_new - s_old)/s_old) # Fractional differences
            vec2_norm = sqrt(eigvec[0,2].real**2+eigvec[1,2].real**2+eigvec[2,2].real**2)
            vec1_norm = sqrt(eigvec[0,1].real**2+eigvec[1,1].real**2+eigvec[2,1].real**2)
            vec0_norm = sqrt(eigvec[0,0].real**2+eigvec[1,0].real**2+eigvec[2,0].real**2)
            # Update morph_info
            morph_info[0] = d
            morph_info[1] = q_new
            morph_info[2] = s_new
            morph_info[3] = eigvec[0,2].real/vec2_norm
            morph_info[4] = eigvec[1,2].real/vec2_norm
            morph_info[5] = eigvec[2,2].real/vec2_norm
            morph_info[6] = eigvec[0,1].real/vec1_norm
            morph_info[7] = eigvec[1,1].real/vec1_norm
            morph_info[8] = eigvec[2,1].real/vec1_norm
            morph_info[9] = eigvec[0,0].real/vec0_norm
            morph_info[10] = eigvec[1,0].real/vec0_norm
            morph_info[11] = eigvec[2,0].real/vec0_norm
            # Transformation into the principal frame
            for i in range(xyz.shape[0]):
                xyz_princ[i,0] = eigvec[0,2].real/vec2_norm*(xyz[i,0]-center[0])+eigvec[1,2].real/vec2_norm*(xyz[i,1]-center[1])+eigvec[2,2].real/vec2_norm*(xyz[i,2]-center[2])
                xyz_princ[i,1] = eigvec[0,1].real/vec1_norm*(xyz[i,0]-center[0])+eigvec[1,1].real/vec1_norm*(xyz[i,1]-center[1])+eigvec[2,1].real/vec1_norm*(xyz[i,2]-center[2])
                xyz_princ[i,2] = eigvec[0,0].real/vec0_norm*(xyz[i,0]-center[0])+eigvec[1,0].real/vec0_norm*(xyz[i,1]-center[1])+eigvec[2,0].real/vec0_norm*(xyz[i,2]-center[2])
            shell[:] = 0
            pts_in_shell = 0
            corr = 0
            if q_new*d <= delta_d or s_new*d <= delta_d:
                for i in range(xyz_princ.shape[0]):
                    if xyz_princ[i,0]**2+xyz_princ[i,1]**2/q_new**2+xyz_princ[i,2]**2/s_new**2 < d**2:
                        shell[i-corr] = i
                        pts_in_shell += 1
                    else:
                        corr += 1
            else:
                for i in range(xyz_princ.shape[0]):
                    if xyz_princ[i,0]**2+xyz_princ[i,1]**2/q_new**2+xyz_princ[i,2]**2/s_new**2 < d**2 and xyz_princ[i,0]**2/(d-delta_d)**2+xyz_princ[i,1]**2/(q_new*d-delta_d)**2+xyz_princ[i,2]**2/(s_new*d-delta_d)**2 >= 1:
                        shell[i-corr] = i
                        pts_in_shell += 1
                    else:
                        corr += 1
            iteration += 1
        return morph_info
    
    
    cdef float[:] runE1(self, float[:] morph_info, float[:,:] xyz, float[:,:] xyz_princ, float[:] masses, int[:] ellipsoid, float[:] center, complex[::1,:] shape_tensor, double[::1] eigval, complex[::1,:] eigvec, float d, float M_TOL, int N_WALL, int N_MIN) nogil:
        
        ellipsoid[:] = 0
        cdef int pts_in_ell = 0
        cdef int corr = 0
        cdef float err = 1.0
        cdef float q_new = 1.0
        cdef float s_new = 1.0
        cdef float q_old = 1.0
        cdef float s_old = 1.0
        cdef int iteration = 1
        cdef float vec2_norm = 1.0
        cdef float vec1_norm = 1.0
        cdef float vec0_norm = 1.0
        cdef int i
        # Start with sphere
        for i in range(xyz.shape[0]):
            if (center[0]-xyz[i,0])**2+(center[1]-xyz[i,1])**2+(center[2]-xyz[i,2])**2 < d**2:
                ellipsoid[i-corr] = i
                pts_in_ell += 1
            else:
                corr += 1
        while (err > M_TOL):
            if iteration > N_WALL:
                morph_info[:] = 0.0
                return morph_info
            if pts_in_ell < N_MIN:
                morph_info[:] = 0.0
                return morph_info
            # Get shape tensor
            shape_tensor = CythonHelpers.getShapeTensor(xyz, ellipsoid, shape_tensor, masses, center, pts_in_ell)
            # Diagonalize shape_tensor
            eigvec[:,:] = 0.0
            eigval[:] = 0.0
            CythonHelpers.ZHEEVR(shape_tensor[:,:], &eigval[0], eigvec, 3)
            q_old = q_new; s_old = s_new
            q_new = sqrt(eigval[1]/eigval[2])
            s_new = sqrt(eigval[0]/eigval[2]) # It is assumed that eigenvalues are approximately proportional to a^2 etc. (true for uniform ellipsoid or uniform shell), though I have never seen any proof..
            err = max(CythonHelpers.cython_abs(q_new - q_old)/q_old, CythonHelpers.cython_abs(s_new - s_old)/s_old) # Fractional differences
            vec2_norm = sqrt(eigvec[0,2].real**2+eigvec[1,2].real**2+eigvec[2,2].real**2)
            vec1_norm = sqrt(eigvec[0,1].real**2+eigvec[1,1].real**2+eigvec[2,1].real**2)
            vec0_norm = sqrt(eigvec[0,0].real**2+eigvec[1,0].real**2+eigvec[2,0].real**2)
            # Update morph_info
            morph_info[0] = d
            morph_info[1] = q_new
            morph_info[2] = s_new
            morph_info[3] = eigvec[0,2].real/vec2_norm
            morph_info[4] = eigvec[1,2].real/vec2_norm
            morph_info[5] = eigvec[2,2].real/vec2_norm
            morph_info[6] = eigvec[0,1].real/vec1_norm
            morph_info[7] = eigvec[1,1].real/vec1_norm
            morph_info[8] = eigvec[2,1].real/vec1_norm
            morph_info[9] = eigvec[0,0].real/vec0_norm
            morph_info[10] = eigvec[1,0].real/vec0_norm
            morph_info[11] = eigvec[2,0].real/vec0_norm
            # Transformation into the principal frame
            for i in range(xyz.shape[0]):
                xyz_princ[i,0] = eigvec[0,2].real/vec2_norm*(xyz[i,0]-center[0])+eigvec[1,2].real/vec2_norm*(xyz[i,1]-center[1])+eigvec[2,2].real/vec2_norm*(xyz[i,2]-center[2])
                xyz_princ[i,1] = eigvec[0,1].real/vec1_norm*(xyz[i,0]-center[0])+eigvec[1,1].real/vec1_norm*(xyz[i,1]-center[1])+eigvec[2,1].real/vec1_norm*(xyz[i,2]-center[2])
                xyz_princ[i,2] = eigvec[0,0].real/vec0_norm*(xyz[i,0]-center[0])+eigvec[1,0].real/vec0_norm*(xyz[i,1]-center[1])+eigvec[2,0].real/vec0_norm*(xyz[i,2]-center[2])
            ellipsoid[:] = 0
            pts_in_ell = 0
            corr = 0
            for i in range(xyz_princ.shape[0]):
                if xyz_princ[i,0]**2+xyz_princ[i,1]**2/q_new**2+xyz_princ[i,2]**2/s_new**2 < d**2:
                    ellipsoid[i-corr] = i
                    pts_in_ell += 1
                else:
                    corr += 1
            iteration += 1
        return morph_info
    
    cdef float[:] runE1VelDisp(self, float[:] morph_info, float[:,:] xyz, float[:,:] vxyz, float[:,:] xyz_princ, float[:] masses, int[:] ellipsoid, float[:] center, float[:] vcenter, complex[::1,:] shape_tensor, double[::1] eigval, complex[::1,:] eigvec, float d, float M_TOL, int N_WALL, int N_MIN) nogil:
        
        ellipsoid[:] = 0
        cdef int pts_in_ell = 0
        cdef int corr = 0
        cdef float err = 1.0
        cdef float q_new = 1.0
        cdef float s_new = 1.0
        cdef float q_old = 1.0
        cdef float s_old = 1.0
        cdef int iteration = 1
        cdef float vec2_norm = 1.0
        cdef float vec1_norm = 1.0
        cdef float vec0_norm = 1.0
        cdef int i
        # Start with sphere
        for i in range(xyz.shape[0]):
            if (center[0]-xyz[i,0])**2+(center[1]-xyz[i,1])**2+(center[2]-xyz[i,2])**2 < d**2:
                ellipsoid[i-corr] = i
                pts_in_ell += 1
            else:
                corr += 1
        while (err > M_TOL):
            if iteration > N_WALL:
                morph_info[:] = 0.0
                return morph_info
            if pts_in_ell < N_MIN:
                morph_info[:] = 0.0
                return morph_info
            # Get shape tensor
            shape_tensor = CythonHelpers.getShapeTensor(vxyz, ellipsoid, shape_tensor, masses, vcenter, pts_in_ell)
            # Diagonalize shape_tensor
            eigvec[:,:] = 0.0
            eigval[:] = 0.0
            CythonHelpers.ZHEEVR(shape_tensor[:,:], &eigval[0], eigvec, 3)
            q_old = q_new; s_old = s_new
            q_new = sqrt(eigval[1]/eigval[2])
            s_new = sqrt(eigval[0]/eigval[2]) # It is assumed that eigenvalues are approximately proportional to a^2 etc. (true for uniform ellipsoid or uniform shell), though I have never seen any proof..
            err = max(CythonHelpers.cython_abs(q_new - q_old)/q_old, CythonHelpers.cython_abs(s_new - s_old)/s_old) # Fractional differences
            vec2_norm = sqrt(eigvec[0,2].real**2+eigvec[1,2].real**2+eigvec[2,2].real**2)
            vec1_norm = sqrt(eigvec[0,1].real**2+eigvec[1,1].real**2+eigvec[2,1].real**2)
            vec0_norm = sqrt(eigvec[0,0].real**2+eigvec[1,0].real**2+eigvec[2,0].real**2)
            # Update morph_info
            morph_info[0] = d
            morph_info[1] = q_new
            morph_info[2] = s_new
            morph_info[3] = eigvec[0,2].real/vec2_norm
            morph_info[4] = eigvec[1,2].real/vec2_norm
            morph_info[5] = eigvec[2,2].real/vec2_norm
            morph_info[6] = eigvec[0,1].real/vec1_norm
            morph_info[7] = eigvec[1,1].real/vec1_norm
            morph_info[8] = eigvec[2,1].real/vec1_norm
            morph_info[9] = eigvec[0,0].real/vec0_norm
            morph_info[10] = eigvec[1,0].real/vec0_norm
            morph_info[11] = eigvec[2,0].real/vec0_norm
            # Transformation into the principal frame
            for i in range(xyz.shape[0]):
                xyz_princ[i,0] = eigvec[0,2].real/vec2_norm*(xyz[i,0]-center[0])+eigvec[1,2].real/vec2_norm*(xyz[i,1]-center[1])+eigvec[2,2].real/vec2_norm*(xyz[i,2]-center[2])
                xyz_princ[i,1] = eigvec[0,1].real/vec1_norm*(xyz[i,0]-center[0])+eigvec[1,1].real/vec1_norm*(xyz[i,1]-center[1])+eigvec[2,1].real/vec1_norm*(xyz[i,2]-center[2])
                xyz_princ[i,2] = eigvec[0,0].real/vec0_norm*(xyz[i,0]-center[0])+eigvec[1,0].real/vec0_norm*(xyz[i,1]-center[1])+eigvec[2,0].real/vec0_norm*(xyz[i,2]-center[2])
            ellipsoid[:] = 0
            pts_in_ell = 0
            corr = 0
            for i in range(xyz_princ.shape[0]):
                if xyz_princ[i,0]**2+xyz_princ[i,1]**2/q_new**2+xyz_princ[i,2]**2/s_new**2 < d**2:
                    ellipsoid[i-corr] = i
                    pts_in_ell += 1
                else:
                    corr += 1
            iteration += 1
        return morph_info
    
    cdef float[:,:] getObjMorphLocal(self, float[:,:] morph_info, float r200, float[:] log_d, float[:,:] xyz, float[:,:] xyz_princ, float[:] masses, int[:] shell, float[:] center, complex[::1,:] shape_tensor, double[::1] eigval, complex[::1,:] eigvec, float M_TOL, int N_WALL, int N_MIN) nogil:
        
        # Return if problematic
        morph_info[:,:] = 0.0
        if CythonHelpers.getLocalSpread(xyz) == 0.0: # Too low resolution = no points in this object
            morph_info[:,:] = 0.0
            return morph_info
        if r200 == 0.0: # We are dealing with a halo which does not have any SHs, so R_200 = 0.0 according to AREPO
            morph_info[:,:] = 0.0
            return morph_info
        
        # Retrieve morphologies for all shells
        cdef int nb_shells = 0
        cdef int i
        for i in range(log_d.shape[0]):
            morph_info[0,i] = r200*log_d[i]
        nb_shells = log_d.shape[0]
        for i in range(nb_shells):
            morph_info[:,i] = self.runE1(morph_info[:,i], xyz, xyz_princ, masses, shell, center, shape_tensor, eigval, eigvec, morph_info[0,i], M_TOL, N_WALL, N_MIN)
        
        # Discard if r200 ellipsoid did not converge
        closest_idx = 0
        for i in range(nb_shells):
            if (r200*log_d[i] - r200)**2 < (r200*log_d[closest_idx] - r200)**2:
                closest_idx = i
        if morph_info[1,closest_idx] == 0: # Return empty morph_info if R200 ellipsoid did not converge
            morph_info[:,:] = 0.0
        return morph_info
    
    cdef float[:] getObjMorphGlobal(self, float[:] morph_info, float r200, float[:,:] xyz, float[:,:] xyz_princ, float[:] masses, int[:] ellipsoid, float[:] center, complex[::1,:] shape_tensor, double[::1] eigval, complex[::1,:] eigvec, float M_TOL, int N_WALL, int N_MIN) nogil:
    
        # Return if problematic
        morph_info[:] = 0.0
        if CythonHelpers.getLocalSpread(xyz) == 0.0: # Too low resolution = no points in this object
            morph_info[:] = 0.0
            return morph_info
        morph_info[0] = r200+self.SAFE
        
        # Retrieve morphology
        morph_info[:] = self.runE1(morph_info[:], xyz, xyz_princ, masses, ellipsoid, center, shape_tensor, eigval, eigvec, morph_info[0], M_TOL, N_WALL, N_MIN)
        return morph_info
    
    cdef float[:,:] getObjMorphLocalVelDisp(self, float[:,:] morph_info, float r200, float[:] log_d, float[:,:] xyz, float[:,:] vxyz, float[:,:] xyz_princ, float[:] masses, int[:] shell, float[:] center, float[:] vcenter, complex[::1,:] shape_tensor, double[::1] eigval, complex[::1,:] eigvec, float M_TOL, int N_WALL, int N_MIN) nogil:
        
        # Return if problematic
        morph_info[:,:] = 0.0
        if CythonHelpers.getLocalSpread(xyz) == 0.0: # Too low resolution = no points in this object
            morph_info[:,:] = 0.0
            return morph_info
        if r200 == 0.0: # We are dealing with a halo which does not have any SHs, so R_200 = 0.0 according to AREPO
            morph_info[:,:] = 0.0
            return morph_info
        
        # Retrieve morphologies for all shells
        cdef int nb_shells = 0
        cdef int i
        for i in range(log_d.shape[0]):
            morph_info[0,i] = r200*log_d[i]
        nb_shells = log_d.shape[0]
        for i in range(nb_shells):
            morph_info[:,i] = self.runE1VelDisp(morph_info[:,i], xyz, vxyz, xyz_princ, masses, shell, center, vcenter, shape_tensor, eigval, eigvec, morph_info[0,i], M_TOL, N_WALL, N_MIN)
        
        # Discard if r200 ellipsoid did not converge
        closest_idx = 0
        for i in range(nb_shells):
            if (r200*log_d[i] - r200)**2 < (r200*log_d[closest_idx] - r200)**2:
                closest_idx = i
        if morph_info[1,closest_idx] == 0: # Return empty morph_info if R200 ellipsoid did not converge
            morph_info[:,:] = 0.0
        return morph_info
    
    cdef float[:] getObjMorphGlobalVelDisp(self, float[:] morph_info, float r200, float[:,:] xyz, float[:,:] vxyz, float[:,:] xyz_princ, float[:] masses, int[:] ellipsoid, float[:] center, float[:] vcenter, complex[::1,:] shape_tensor, double[::1] eigval, complex[::1,:] eigvec, float M_TOL, int N_WALL, int N_MIN) nogil:
        
        # Return if problematic
        morph_info[:] = 0.0
        if CythonHelpers.getLocalSpread(xyz) == 0.0: # Too low resolution = no points in this object
            morph_info[:] = 0.0
            return morph_info
        morph_info[0] = r200+self.SAFE
        
        # Retrieve morphology
        morph_info[:] = self.runE1VelDisp(morph_info[:], xyz, vxyz, xyz_princ, masses, ellipsoid, center, vcenter, shape_tensor, eigval, eigvec, morph_info[0], M_TOL, N_WALL, N_MIN)
        return morph_info
    
    def getMorphLocal(self, float[:,:] xyz, cat, float[:] masses, float[:] r200, float L_BOX, int MIN_NUMBER_PTCS, int D_LOGSTART, int D_LOGEND, int D_BINS, int M_TOL, int N_WALL, int N_MIN, str CENTER):
                        
        # Transform cat to int[:,:]
        cdef int nb_objs = len(cat)
        cdef int p
        cdef int[:] obj_pass = np.zeros((nb_objs,), dtype = np.int32)
        cdef int[:] obj_size = np.zeros((nb_objs,), dtype = np.int32)
        for p in range(nb_objs):
            if len(cat[p]) >= MIN_NUMBER_PTCS: # Only add objects that have sufficient resolution
                obj_pass[p] = 1      
                obj_size[p] = len(cat[p]) 
        cdef int nb_pass = np.sum(obj_pass.base)
        cdef int[:,:] cat_arr = np.zeros((nb_pass,np.max([len(cat[p]) for p in range(nb_objs)])), dtype = np.int32)
        cdef int[:] idxs_compr = np.zeros((nb_objs,), dtype = np.int32)
        idxs_compr.base[obj_pass.base.nonzero()[0]] = np.arange(np.sum(obj_pass.base))
        for p in range(nb_objs):
            if obj_pass[p] == 1:
                cat_arr.base[idxs_compr[p],:obj_size[p]] = np.array(cat[p])
    
        cdef float[:] m = np.zeros((nb_objs,), dtype = np.float32)
        cdef float[:,:] d = np.zeros((nb_objs, D_BINS+1), dtype = np.float32)
        cdef float[:,:] q = np.zeros((nb_objs, D_BINS+1), dtype = np.float32)
        cdef float[:,:] s = np.zeros((nb_objs, D_BINS+1), dtype = np.float32)
        cdef float[:,:] major_x = np.zeros((nb_objs, D_BINS+1), dtype = np.float32)
        cdef float[:,:] major_y = np.zeros((nb_objs, D_BINS+1), dtype = np.float32)
        cdef float[:,:] major_z = np.zeros((nb_objs, D_BINS+1), dtype = np.float32)
        cdef float[:,:] inter_x = np.zeros((nb_objs, D_BINS+1), dtype = np.float32)
        cdef float[:,:] inter_y = np.zeros((nb_objs, D_BINS+1), dtype = np.float32)
        cdef float[:,:] inter_z = np.zeros((nb_objs, D_BINS+1), dtype = np.float32)
        cdef float[:,:] minor_x = np.zeros((nb_objs, D_BINS+1), dtype = np.float32)
        cdef float[:,:] minor_y = np.zeros((nb_objs, D_BINS+1), dtype = np.float32)
        cdef float[:,:] minor_z = np.zeros((nb_objs, D_BINS+1), dtype = np.float32)
        cdef float[:,:,:] morph_info = np.zeros((openmp.omp_get_max_threads(), 12, D_BINS+1), dtype = np.float32)
        cdef float[:,:,:] xyz_obj = np.zeros((openmp.omp_get_max_threads(), cat_arr.shape[1],3), dtype = np.float32)
        cdef float[:,:,:] xyz_princ = np.zeros((openmp.omp_get_max_threads(), cat_arr.shape[1],3), dtype = np.float32)
        cdef int[:,:] shell = np.zeros((openmp.omp_get_max_threads(), cat_arr.shape[1]), dtype = np.int32)
        cdef complex[::1,:,:] shape_tensor = np.zeros((3, 3, openmp.omp_get_max_threads()), dtype = np.complex128, order='F')
        cdef double[::1,:] eigval = np.zeros((3, openmp.omp_get_max_threads()), dtype=np.float64, order='F')
        cdef complex[::1,:,:] eigvec = np.zeros((3,3, openmp.omp_get_max_threads()), dtype=np.complex128, order='F')
        cdef float[:,:] m_obj = np.zeros((openmp.omp_get_max_threads(), cat_arr.shape[1]), dtype = np.float32)
        cdef float[:] log_d = np.logspace(D_LOGSTART,D_LOGEND,D_BINS+1, dtype = np.float32)
        cdef bint success
        cdef int n
        cdef int r
        cdef float[:,:] centers = np.zeros((nb_objs,3), dtype = np.float32)
        for p in range(nb_objs): # Calculate centers of objects
            if obj_pass[p] == 1:
                xyz_ = respectPBCNoRef(xyz.base[cat_arr.base[idxs_compr[p],:obj_size[p]]], L_BOX)
                if CENTER == 'mode':
                    centers.base[p] = findMode(xyz_, masses.base[cat_arr.base[idxs_compr[p],:obj_size[p]]], 1000)
                else:
                    centers.base[p] = getCoM(xyz_, masses.base[cat_arr.base[idxs_compr[p],:obj_size[p]]])
        for p in prange(nb_objs, schedule = 'dynamic', nogil = True):
            if obj_pass[p] == 1:
                for n in range(obj_size[p]):
                    xyz_obj[openmp.omp_get_thread_num(),n,0] = xyz[cat_arr[idxs_compr[p],n],0]
                    xyz_obj[openmp.omp_get_thread_num(),n,1] = xyz[cat_arr[idxs_compr[p],n],1]
                    xyz_obj[openmp.omp_get_thread_num(),n,2] = xyz[cat_arr[idxs_compr[p],n],2]
                    m_obj[openmp.omp_get_thread_num(),n] = masses[cat_arr[idxs_compr[p],n]]
                    m[p] = m[p] + masses[cat_arr[idxs_compr[p],n]]
                xyz_obj[openmp.omp_get_thread_num(),:obj_size[p]] = CythonHelpers.respectPBCNoRef(xyz_obj[openmp.omp_get_thread_num(),:obj_size[p]], L_BOX)
                morph_info[openmp.omp_get_thread_num(),:,:] = self.getObjMorphLocal(morph_info[openmp.omp_get_thread_num(),:,:], r200[p], log_d, xyz_obj[openmp.omp_get_thread_num(),:obj_size[p]], xyz_princ[openmp.omp_get_thread_num(),:obj_size[p]], m_obj[openmp.omp_get_thread_num(),:obj_size[p]], shell[openmp.omp_get_thread_num()], centers[p], shape_tensor[:,:,openmp.omp_get_thread_num()], eigval[:,openmp.omp_get_thread_num()], eigvec[:,:,openmp.omp_get_thread_num()], M_TOL, N_WALL, N_MIN)
                d[p] = morph_info[openmp.omp_get_thread_num(),0]
                q[p] = morph_info[openmp.omp_get_thread_num(),1]
                s[p] = morph_info[openmp.omp_get_thread_num(),2]
                major_x[p] = morph_info[openmp.omp_get_thread_num(),3]
                major_y[p] = morph_info[openmp.omp_get_thread_num(),4]
                major_z[p] = morph_info[openmp.omp_get_thread_num(),5]
                inter_x[p] = morph_info[openmp.omp_get_thread_num(),6]
                inter_y[p] = morph_info[openmp.omp_get_thread_num(),7]
                inter_z[p] = morph_info[openmp.omp_get_thread_num(),8]
                minor_x[p] = morph_info[openmp.omp_get_thread_num(),9]
                minor_y[p] = morph_info[openmp.omp_get_thread_num(),10]
                minor_z[p] = morph_info[openmp.omp_get_thread_num(),11]
                if obj_size[p] != 0:
                    success = False
                    for n in range(morph_info.shape[1]):
                        for r in range(morph_info.shape[2]):
                            if morph_info[openmp.omp_get_thread_num(),n,r] != 0.0:
                                success = True
                                break
                    printf("Purpose: local. Dealing with object number %d. The number of ptcs is %d. Shape determination at R200 successful: %d\n", p, obj_size[p], success)
            
        l_succeed = []
        for p in range(nb_objs):
            if not (d.base[p] == d.base[p,0]).all():
                l_succeed += [p]
        succeed = np.array(l_succeed)
        if succeed.shape[0] != 0:
            minor = np.transpose(np.stack((minor_x.base[succeed],minor_y.base[succeed],minor_z.base[succeed])),(1,2,0))
            inter = np.transpose(np.stack((inter_x.base[succeed],inter_y.base[succeed],inter_z.base[succeed])),(1,2,0))
            major = np.transpose(np.stack((major_x.base[succeed],major_y.base[succeed],major_z.base[succeed])),(1,2,0))
            d.base[succeed][d.base[succeed]==0.0] = np.nan
            s.base[succeed][s.base[succeed]==0.0] = np.nan
            q.base[succeed][q.base[succeed]==0.0] = np.nan
            minor[minor==0.0] = np.nan
            inter[inter==0.0] = np.nan
            major[major==0.0] = np.nan
            centers.base[succeed][centers.base[succeed]==0.0] = np.nan
            m.base[succeed][m.base[succeed]==0.0] = np.nan
            return d.base[succeed], q.base[succeed], s.base[succeed], minor, inter, major, centers.base[succeed], m.base[succeed], l_succeed # Only rank = 0 content matters
        else:
            return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), l_succeed
    
    def getMorphGlobal(self, float[:,:] xyz, cat, float[:] masses, float[:] r200, float L_BOX, int MIN_NUMBER_PTCS, int M_TOL, int N_WALL, int N_MIN, str CENTER):
        
        # Transform cat to int[:,:]
        cdef int nb_objs = len(cat)
        cdef int p
        cdef int[:] obj_pass = np.zeros((nb_objs,), dtype = np.int32)
        cdef int[:] obj_size = np.zeros((nb_objs,), dtype = np.int32)
        for p in range(nb_objs):
            if len(cat[p]) >= MIN_NUMBER_PTCS: # Only add objects that have sufficient resolution
                obj_pass[p] = 1      
                obj_size[p] = len(cat[p]) 
        cdef int nb_pass = np.sum(obj_pass.base)
        cdef int[:,:] cat_arr = np.zeros((nb_pass,np.max([len(cat[p]) for p in range(nb_objs)])), dtype = np.int32)
        cdef int[:] idxs_compr = np.zeros((nb_objs,), dtype = np.int32)
        idxs_compr.base[obj_pass.base.nonzero()[0]] = np.arange(np.sum(obj_pass.base))
        for p in range(nb_objs):
            if obj_pass[p] == 1:
                cat_arr.base[idxs_compr[p],:obj_size[p]] = np.array(cat[p])
    
        cdef float[:] m = np.zeros((nb_objs,), dtype = np.float32)
        cdef float[:] d = np.zeros((nb_objs,), dtype = np.float32)
        cdef float[:] q = np.zeros((nb_objs,), dtype = np.float32)
        cdef float[:] s = np.zeros((nb_objs,), dtype = np.float32)
        cdef float[:] major_x = np.zeros((nb_objs,), dtype = np.float32)
        cdef float[:] major_y = np.zeros((nb_objs,), dtype = np.float32)
        cdef float[:] major_z = np.zeros((nb_objs,), dtype = np.float32)
        cdef float[:] inter_x = np.zeros((nb_objs,), dtype = np.float32)
        cdef float[:] inter_y = np.zeros((nb_objs,), dtype = np.float32)
        cdef float[:] inter_z = np.zeros((nb_objs,), dtype = np.float32)
        cdef float[:] minor_x = np.zeros((nb_objs,), dtype = np.float32)
        cdef float[:] minor_y = np.zeros((nb_objs,), dtype = np.float32)
        cdef float[:] minor_z = np.zeros((nb_objs,), dtype = np.float32)
        cdef float[:,:] centers = np.zeros((nb_objs,3), dtype = np.float32)
        cdef float[:,:] morph_info = np.zeros((openmp.omp_get_max_threads(), 12), dtype = np.float32)
        cdef float[:,:,:] xyz_obj = np.zeros((openmp.omp_get_max_threads(), cat_arr.shape[1],3), dtype = np.float32)
        cdef float[:,:,:] xyz_princ = np.zeros((openmp.omp_get_max_threads(), cat_arr.shape[1],3), dtype = np.float32)
        cdef int[:,:] ellipsoid = np.zeros((openmp.omp_get_max_threads(), cat_arr.shape[1]), dtype = np.int32)
        cdef complex[::1,:,:] shape_tensor = np.zeros((3, 3, openmp.omp_get_max_threads()), dtype = np.complex128, order='F')
        cdef double[::1,:] eigval = np.zeros((3, openmp.omp_get_max_threads()), dtype=np.float64, order='F')
        cdef complex[::1,:,:] eigvec = np.zeros((3,3, openmp.omp_get_max_threads()), dtype=np.complex128, order='F')
        cdef float[:,:] m_obj = np.zeros((openmp.omp_get_max_threads(), cat_arr.shape[1]), dtype = np.float32)
        cdef bint success
        cdef int n
        for p in range(nb_objs): # Calculate centers of objects
            if obj_pass[p] == 1:
                xyz_ = respectPBCNoRef(xyz.base[cat_arr.base[idxs_compr[p],:obj_size[p]]], L_BOX)
                if CENTER == 'mode':
                    centers.base[p] = findMode(xyz_, masses.base[cat_arr.base[idxs_compr[p],:obj_size[p]]], 1000)
                else:
                    centers.base[p] = getCoM(xyz_, masses.base[cat_arr.base[idxs_compr[p],:obj_size[p]]])
        for p in prange(nb_objs, schedule = 'dynamic', nogil = True):
            if obj_pass[p] == 1:
                for n in range(obj_size[p]):
                    xyz_obj[openmp.omp_get_thread_num(),n] = xyz[cat_arr[idxs_compr[p],n]]
                    m_obj[openmp.omp_get_thread_num(),n] = masses[cat_arr[idxs_compr[p],n]]
                    m[p] = m[p] + masses[cat_arr[idxs_compr[p],n]]
                xyz_obj[openmp.omp_get_thread_num(),:obj_size[p]] = CythonHelpers.respectPBCNoRef(xyz_obj[openmp.omp_get_thread_num(),:obj_size[p]], L_BOX)
                morph_info[openmp.omp_get_thread_num(),:] = self.getObjMorphGlobal(morph_info[openmp.omp_get_thread_num(),:], r200[p], xyz_obj[openmp.omp_get_thread_num(),:obj_size[p]], xyz_princ[openmp.omp_get_thread_num(),:obj_size[p]], m_obj[openmp.omp_get_thread_num(),:obj_size[p]], ellipsoid[openmp.omp_get_thread_num()], centers[p], shape_tensor[:,:,openmp.omp_get_thread_num()], eigval[:,openmp.omp_get_thread_num()], eigvec[:,:,openmp.omp_get_thread_num()], M_TOL, N_WALL, N_MIN)
                d[p] = morph_info[openmp.omp_get_thread_num(),0]
                q[p] = morph_info[openmp.omp_get_thread_num(),1]
                s[p] = morph_info[openmp.omp_get_thread_num(),2]
                major_x[p] = morph_info[openmp.omp_get_thread_num(),3]
                major_y[p] = morph_info[openmp.omp_get_thread_num(),4]
                major_z[p] = morph_info[openmp.omp_get_thread_num(),5]
                inter_x[p] = morph_info[openmp.omp_get_thread_num(),6]
                inter_y[p] = morph_info[openmp.omp_get_thread_num(),7]
                inter_z[p] = morph_info[openmp.omp_get_thread_num(),8]
                minor_x[p] = morph_info[openmp.omp_get_thread_num(),9]
                minor_y[p] = morph_info[openmp.omp_get_thread_num(),10]
                minor_z[p] = morph_info[openmp.omp_get_thread_num(),11]
                if obj_size[p] != 0:
                    success = False
                    for n in range(12):
                        if morph_info[openmp.omp_get_thread_num(),n] != 0.0:
                            success = True
                            break
                    printf("Purpose: global. Dealing with object number %d. The number of ptcs is %d. Global shape determination successful: %d\n", p, obj_size[p], success)
            
        l_succeed = []
        for p in range(nb_objs):
            if obj_pass[p] == 1:
                l_succeed += [p]
        succeed = np.array(l_succeed)
        if succeed.shape[0] == 0:
            return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
        minor = np.hstack((np.reshape(minor_x.base[succeed], (minor_x.base[succeed].shape[0],1)), np.reshape(minor_y.base[succeed], (minor_y.base[succeed].shape[0],1)), np.reshape(minor_z.base[succeed], (minor_z.base[succeed].shape[0],1))))
        inter = np.hstack((np.reshape(inter_x.base[succeed], (inter_x.base[succeed].shape[0],1)), np.reshape(inter_y.base[succeed], (inter_y.base[succeed].shape[0],1)), np.reshape(inter_z.base[succeed], (inter_z.base[succeed].shape[0],1))))
        major = np.hstack((np.reshape(major_x.base[succeed], (major_x.base[succeed].shape[0],1)), np.reshape(major_y.base[succeed], (major_y.base[succeed].shape[0],1)), np.reshape(major_z.base[succeed], (major_z.base[succeed].shape[0],1))))
        d.base[succeed][d.base[succeed]==0.0] = np.nan
        s.base[succeed][s.base[succeed]==0.0] = np.nan
        q.base[succeed][q.base[succeed]==0.0] = np.nan
        minor[minor==0.0] = np.nan
        inter[inter==0.0] = np.nan
        major[major==0.0] = np.nan
        centers.base[succeed][centers.base[succeed]==0.0] = np.nan
        m.base[succeed][m.base[succeed]==0.0] = np.nan
        return d.base[succeed], q.base[succeed], s.base[succeed], minor, inter, major, centers.base[succeed], m.base[succeed] # Only rank = 0 content matters
    
    def getMorphLocalVelDisp(self, float[:,:] xyz, float[:,:] vxyz, cat, float[:] masses, float[:] r200, float L_BOX, int MIN_NUMBER_PTCS, int D_LOGSTART, int D_LOGEND, int D_BINS, int M_TOL, int N_WALL, int N_MIN, str CENTER):
        
        # Transform cat to int[:,:]
        cdef int nb_objs = len(cat)
        cdef int p
        cdef int[:] obj_pass = np.zeros((nb_objs,), dtype = np.int32)
        cdef int[:] obj_size = np.zeros((nb_objs,), dtype = np.int32)
        for p in range(nb_objs):
            if len(cat[p]) >= MIN_NUMBER_PTCS: # Only add objects that have sufficient resolution
                obj_pass[p] = 1      
                obj_size[p] = len(cat[p]) 
        cdef int nb_pass = np.sum(obj_pass.base)
        cdef int[:,:] cat_arr = np.zeros((nb_pass,np.max([len(cat[p]) for p in range(nb_objs)])), dtype = np.int32)
        cdef int[:] idxs_compr = np.zeros((nb_objs,), dtype = np.int32)
        idxs_compr.base[obj_pass.base.nonzero()[0]] = np.arange(np.sum(obj_pass.base))
        for p in range(nb_objs):
            if obj_pass[p] == 1:
                cat_arr.base[idxs_compr[p],:obj_size[p]] = np.array(cat[p])
    
        cdef float[:] m = np.zeros((nb_objs,), dtype = np.float32)
        cdef float[:,:] d = np.zeros((nb_objs, D_BINS+1), dtype = np.float32)
        cdef float[:,:] q = np.zeros((nb_objs, D_BINS+1), dtype = np.float32)
        cdef float[:,:] s = np.zeros((nb_objs, D_BINS+1), dtype = np.float32)
        cdef float[:,:] major_x = np.zeros((nb_objs, D_BINS+1), dtype = np.float32)
        cdef float[:,:] major_y = np.zeros((nb_objs, D_BINS+1), dtype = np.float32)
        cdef float[:,:] major_z = np.zeros((nb_objs, D_BINS+1), dtype = np.float32)
        cdef float[:,:] inter_x = np.zeros((nb_objs, D_BINS+1), dtype = np.float32)
        cdef float[:,:] inter_y = np.zeros((nb_objs, D_BINS+1), dtype = np.float32)
        cdef float[:,:] inter_z = np.zeros((nb_objs, D_BINS+1), dtype = np.float32)
        cdef float[:,:] minor_x = np.zeros((nb_objs, D_BINS+1), dtype = np.float32)
        cdef float[:,:] minor_y = np.zeros((nb_objs, D_BINS+1), dtype = np.float32)
        cdef float[:,:] minor_z = np.zeros((nb_objs, D_BINS+1), dtype = np.float32)
        cdef float[:,:] centers = np.zeros((nb_objs,3), dtype = np.float32)
        cdef float[:,:] vcenters = np.zeros((nb_objs,3), dtype = np.float32)
        cdef float[:,:,:] morph_info = np.zeros((openmp.omp_get_max_threads(), 12, D_BINS+1), dtype = np.float32)
        cdef float[:,:,:] xyz_obj = np.zeros((openmp.omp_get_max_threads(), cat_arr.shape[1],3), dtype = np.float32)
        cdef float[:,:,:] vxyz_obj = np.zeros((openmp.omp_get_max_threads(), cat_arr.shape[1],3), dtype = np.float32)
        cdef float[:,:,:] xyz_princ = np.zeros((openmp.omp_get_max_threads(), cat_arr.shape[1],3), dtype = np.float32)
        cdef int[:,:] shell = np.zeros((openmp.omp_get_max_threads(), cat_arr.shape[1]), dtype = np.int32)
        cdef complex[::1,:,:] shape_tensor = np.zeros((3, 3, openmp.omp_get_max_threads()), dtype = np.complex128, order='F')
        cdef double[::1,:] eigval = np.zeros((3, openmp.omp_get_max_threads()), dtype=np.float64, order='F')
        cdef complex[::1,:,:] eigvec = np.zeros((3,3, openmp.omp_get_max_threads()), dtype=np.complex128, order='F')
        cdef float[:,:] m_obj = np.zeros((openmp.omp_get_max_threads(), cat_arr.shape[1]), dtype = np.float32)
        cdef float[:] log_d = np.logspace(D_LOGSTART,D_LOGEND,D_BINS+1, dtype = np.float32)
        cdef bint success
        cdef int n
        cdef int r
        for p in range(nb_objs): # Calculate centers of objects
            if obj_pass[p] == 1:
                xyz_ = respectPBCNoRef(xyz.base[cat_arr.base[idxs_compr[p],:obj_size[p]]], L_BOX)
                if CENTER == 'mode':
                    centers.base[p] = findMode(xyz_, masses.base[cat_arr.base[idxs_compr[p],:obj_size[p]]], 1000)
                else:
                    centers.base[p] = getCoM(xyz_, masses.base[cat_arr.base[idxs_compr[p],:obj_size[p]]])
        for p in prange(nb_objs, schedule = 'dynamic', nogil = True):
            if obj_pass[p] == 1:
                for n in range(obj_size[p]):
                    xyz_obj[openmp.omp_get_thread_num(),n,0] = xyz[cat_arr[idxs_compr[p],n],0]
                    xyz_obj[openmp.omp_get_thread_num(),n,1] = xyz[cat_arr[idxs_compr[p],n],1]
                    xyz_obj[openmp.omp_get_thread_num(),n,2] = xyz[cat_arr[idxs_compr[p],n],2]
                    m_obj[openmp.omp_get_thread_num(),n] = masses[cat_arr[idxs_compr[p],n]]
                    m[p] = m[p] + masses[cat_arr[idxs_compr[p],n]]
                xyz_obj[openmp.omp_get_thread_num(),:obj_size[p]] = CythonHelpers.respectPBCNoRef(xyz_obj[openmp.omp_get_thread_num(),:obj_size[p]], L_BOX)
                vcenters[p] = CythonHelpers.getCoM(vxyz_obj[openmp.omp_get_thread_num(),:obj_size[p]], m_obj[openmp.omp_get_thread_num(),:obj_size[p]], vcenters[p])
                morph_info[openmp.omp_get_thread_num(),:,:] = self.getObjMorphLocalVelDisp(morph_info[openmp.omp_get_thread_num(),:,:], r200[p], log_d, xyz_obj[openmp.omp_get_thread_num(),:obj_size[p]], vxyz_obj[openmp.omp_get_thread_num(),:obj_size[p]], xyz_princ[openmp.omp_get_thread_num(),:obj_size[p]], m_obj[openmp.omp_get_thread_num(),:obj_size[p]], shell[openmp.omp_get_thread_num()], centers[p], vcenters[p], shape_tensor[:,:,openmp.omp_get_thread_num()], eigval[:,openmp.omp_get_thread_num()], eigvec[:,:,openmp.omp_get_thread_num()], M_TOL, N_WALL, N_MIN)
                d[p] = morph_info[openmp.omp_get_thread_num(),0]
                q[p] = morph_info[openmp.omp_get_thread_num(),1]
                s[p] = morph_info[openmp.omp_get_thread_num(),2]
                major_x[p] = morph_info[openmp.omp_get_thread_num(),3]
                major_y[p] = morph_info[openmp.omp_get_thread_num(),4]
                major_z[p] = morph_info[openmp.omp_get_thread_num(),5]
                inter_x[p] = morph_info[openmp.omp_get_thread_num(),6]
                inter_y[p] = morph_info[openmp.omp_get_thread_num(),7]
                inter_z[p] = morph_info[openmp.omp_get_thread_num(),8]
                minor_x[p] = morph_info[openmp.omp_get_thread_num(),9]
                minor_y[p] = morph_info[openmp.omp_get_thread_num(),10]
                minor_z[p] = morph_info[openmp.omp_get_thread_num(),11]
                if obj_size[p] != 0:
                    success = False
                    for n in range(morph_info.shape[1]):
                        for r in range(morph_info.shape[2]):
                            if morph_info[openmp.omp_get_thread_num(),n,r] != 0.0:
                                success = True
                                break
                    printf("Purpose: local. Dealing with object number %d. The number of ptcs is %d. Shape determination at R200 successful: %d\n", p, obj_size[p], success)
        
        l_succeed = []
        for p in range(nb_objs):
            if not (d.base[p] == d.base[p,0]).all():
                l_succeed += [p]
        succeed = np.array(l_succeed)
        if succeed.shape[0] != 0:
            minor = np.transpose(np.stack((minor_x.base[succeed],minor_y.base[succeed],minor_z.base[succeed])),(1,2,0))
            inter = np.transpose(np.stack((inter_x.base[succeed],inter_y.base[succeed],inter_z.base[succeed])),(1,2,0))
            major = np.transpose(np.stack((major_x.base[succeed],major_y.base[succeed],major_z.base[succeed])),(1,2,0))
            d.base[succeed][d.base[succeed]==0.0] = np.nan
            s.base[succeed][s.base[succeed]==0.0] = np.nan
            q.base[succeed][q.base[succeed]==0.0] = np.nan
            minor[minor==0.0] = np.nan
            inter[inter==0.0] = np.nan
            major[major==0.0] = np.nan
            centers.base[succeed][centers.base[succeed]==0.0] = np.nan
            m.base[succeed][m.base[succeed]==0.0] = np.nan
            return d.base[succeed], q.base[succeed], s.base[succeed], minor, inter, major, centers.base[succeed], m.base[succeed], l_succeed # Only rank = 0 content matters
        else:
            return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), l_succeed # Only rank = 0 content matters
    
    def getMorphGlobalVelDisp(self, float[:,:] xyz, float[:,:] vxyz, cat, float[:] masses, float[:] r200, float L_BOX, int MIN_NUMBER_PTCS, int M_TOL, int N_WALL, int N_MIN, str CENTER):
        
        # Transform cat to int[:,:]
        cdef int nb_objs = len(cat)
        cdef int p
        cdef int[:] obj_pass = np.zeros((nb_objs,), dtype = np.int32)
        cdef int[:] obj_size = np.zeros((nb_objs,), dtype = np.int32)
        for p in range(nb_objs):
            if len(cat[p]) >= MIN_NUMBER_PTCS: # Only add objects that have sufficient resolution
                obj_pass[p] = 1      
                obj_size[p] = len(cat[p]) 
        cdef int nb_pass = np.sum(obj_pass.base)
        cdef int[:,:] cat_arr = np.zeros((nb_pass,np.max([len(cat[p]) for p in range(nb_objs)])), dtype = np.int32)
        cdef int[:] idxs_compr = np.zeros((nb_objs,), dtype = np.int32)
        idxs_compr.base[obj_pass.base.nonzero()[0]] = np.arange(np.sum(obj_pass.base))
        for p in range(nb_objs):
            if obj_pass[p] == 1:
                cat_arr.base[idxs_compr[p],:obj_size[p]] = np.array(cat[p])
    
        cdef float[:] m = np.zeros((nb_objs,), dtype = np.float32)
        cdef float[:] d = np.zeros((nb_objs,), dtype = np.float32)
        cdef float[:] q = np.zeros((nb_objs,), dtype = np.float32)
        cdef float[:] s = np.zeros((nb_objs,), dtype = np.float32)
        cdef float[:] major_x = np.zeros((nb_objs,), dtype = np.float32)
        cdef float[:] major_y = np.zeros((nb_objs,), dtype = np.float32)
        cdef float[:] major_z = np.zeros((nb_objs,), dtype = np.float32)
        cdef float[:] inter_x = np.zeros((nb_objs,), dtype = np.float32)
        cdef float[:] inter_y = np.zeros((nb_objs,), dtype = np.float32)
        cdef float[:] inter_z = np.zeros((nb_objs,), dtype = np.float32)
        cdef float[:] minor_x = np.zeros((nb_objs,), dtype = np.float32)
        cdef float[:] minor_y = np.zeros((nb_objs,), dtype = np.float32)
        cdef float[:] minor_z = np.zeros((nb_objs,), dtype = np.float32)
        cdef float[:,:] centers = np.zeros((nb_objs,3), dtype = np.float32)
        cdef float[:,:] vcenters = np.zeros((nb_objs,3), dtype = np.float32)
        cdef float[:,:] morph_info = np.zeros((openmp.omp_get_max_threads(), 12), dtype = np.float32)
        cdef float[:,:,:] xyz_obj = np.zeros((openmp.omp_get_max_threads(), cat_arr.shape[1],3), dtype = np.float32)
        cdef float[:,:,:] vxyz_obj = np.zeros((openmp.omp_get_max_threads(), cat_arr.shape[1],3), dtype = np.float32)
        cdef float[:,:,:] xyz_princ = np.zeros((openmp.omp_get_max_threads(), cat_arr.shape[1],3), dtype = np.float32)
        cdef int[:,:] ellipsoid = np.zeros((openmp.omp_get_max_threads(), cat_arr.shape[1]), dtype = np.int32)
        cdef complex[::1,:,:] shape_tensor = np.zeros((3, 3, openmp.omp_get_max_threads()), dtype = np.complex128, order='F')
        cdef double[::1,:] eigval = np.zeros((3, openmp.omp_get_max_threads()), dtype=np.float64, order='F')
        cdef complex[::1,:,:] eigvec = np.zeros((3,3, openmp.omp_get_max_threads()), dtype=np.complex128, order='F')
        cdef float[:,:] m_obj = np.zeros((openmp.omp_get_max_threads(), cat_arr.shape[1]), dtype = np.float32)
        cdef bint success
        cdef int n
        for p in range(nb_objs): # Calculate centers of objects
            if obj_pass[p] == 1:
                xyz_ = respectPBCNoRef(xyz.base[cat_arr.base[idxs_compr[p],:obj_size[p]]], L_BOX)
                if CENTER == 'mode':
                    centers.base[p] = findMode(xyz_, masses.base[cat_arr.base[idxs_compr[p],:obj_size[p]]], 1000)
                else:
                    centers.base[p] = getCoM(xyz_, masses.base[cat_arr.base[idxs_compr[p],:obj_size[p]]])
        for p in prange(nb_objs, schedule = 'dynamic', nogil = True):
            if obj_pass[p] == 1:
                for n in range(obj_size[p]):
                    xyz_obj[openmp.omp_get_thread_num(),n] = xyz[cat_arr[idxs_compr[p],n]]
                    vxyz_obj[openmp.omp_get_thread_num(),n] = vxyz[cat_arr[idxs_compr[p],n]]
                    m_obj[openmp.omp_get_thread_num(),n] = masses[cat_arr[idxs_compr[p],n]]
                    m[p] = m[p] + masses[cat_arr[idxs_compr[p],n]]
                xyz_obj[openmp.omp_get_thread_num(),:obj_size[p]] = CythonHelpers.respectPBCNoRef(xyz_obj[openmp.omp_get_thread_num(),:obj_size[p]], L_BOX)
                vcenters[p] = CythonHelpers.getCoM(vxyz_obj[openmp.omp_get_thread_num(),:obj_size[p]], m_obj[openmp.omp_get_thread_num(),:obj_size[p]], vcenters[p])
                morph_info[openmp.omp_get_thread_num(),:] = self.getObjMorphGlobalVelDisp(morph_info[openmp.omp_get_thread_num(),:], r200[p], xyz_obj[openmp.omp_get_thread_num(),:obj_size[p]], vxyz_obj[openmp.omp_get_thread_num(),:obj_size[p]], xyz_princ[openmp.omp_get_thread_num(),:obj_size[p]], m_obj[openmp.omp_get_thread_num(),:obj_size[p]], ellipsoid[openmp.omp_get_thread_num()], centers[p], vcenters[p], shape_tensor[:,:,openmp.omp_get_thread_num()], eigval[:,openmp.omp_get_thread_num()], eigvec[:,:,openmp.omp_get_thread_num()], M_TOL, N_WALL, N_MIN)
                d[p] = morph_info[openmp.omp_get_thread_num(),0]
                q[p] = morph_info[openmp.omp_get_thread_num(),1]
                s[p] = morph_info[openmp.omp_get_thread_num(),2]
                major_x[p] = morph_info[openmp.omp_get_thread_num(),3]
                major_y[p] = morph_info[openmp.omp_get_thread_num(),4]
                major_z[p] = morph_info[openmp.omp_get_thread_num(),5]
                inter_x[p] = morph_info[openmp.omp_get_thread_num(),6]
                inter_y[p] = morph_info[openmp.omp_get_thread_num(),7]
                inter_z[p] = morph_info[openmp.omp_get_thread_num(),8]
                minor_x[p] = morph_info[openmp.omp_get_thread_num(),9]
                minor_y[p] = morph_info[openmp.omp_get_thread_num(),10]
                minor_z[p] = morph_info[openmp.omp_get_thread_num(),11]
                if obj_size[p] != 0:
                    success = False
                    for n in range(12):
                        if morph_info[openmp.omp_get_thread_num(),n] != 0.0:
                            success = True
                            break
                    printf("Purpose: vdisp. Dealing with object number %d. The number of ptcs is %d. VelDisp shape determination at R200 successful: %d\n", p, obj_size[p], success)
            
        l_succeed = []
        for p in range(nb_objs):
            if obj_pass[p] == 1:
                l_succeed += [p]
        succeed = np.array(l_succeed)
        if succeed.shape[0] == 0:
            return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
        minor = np.hstack((np.reshape(minor_x.base[succeed], (minor_x.base[succeed].shape[0],1)), np.reshape(minor_y.base[succeed], (minor_y.base[succeed].shape[0],1)), np.reshape(minor_z.base[succeed], (minor_z.base[succeed].shape[0],1))))
        inter = np.hstack((np.reshape(inter_x.base[succeed], (inter_x.base[succeed].shape[0],1)), np.reshape(inter_y.base[succeed], (inter_y.base[succeed].shape[0],1)), np.reshape(inter_z.base[succeed], (inter_z.base[succeed].shape[0],1))))
        major = np.hstack((np.reshape(major_x.base[succeed], (major_x.base[succeed].shape[0],1)), np.reshape(major_y.base[succeed], (major_y.base[succeed].shape[0],1)), np.reshape(major_z.base[succeed], (major_z.base[succeed].shape[0],1))))
        d.base[succeed][d.base[succeed]==0.0] = np.nan
        s.base[succeed][s.base[succeed]==0.0] = np.nan
        q.base[succeed][q.base[succeed]==0.0] = np.nan
        minor[minor==0.0] = np.nan
        inter[inter==0.0] = np.nan
        major[major==0.0] = np.nan
        centers.base[succeed][centers.base[succeed]==0.0] = np.nan
        m.base[succeed][m.base[succeed]==0.0] = np.nan
        return d.base[succeed], q.base[succeed], s.base[succeed], minor, inter, major, centers.base[succeed], m.base[succeed] # Only rank = 0 content matters
    
    def drawShapeProfiles(self, obj_type = ''):
        if obj_type == 'dm':
            suffix = '_dm_'
        elif obj_type == 'gx':
            suffix = '_gx_'
        else:
            assert obj_type == ''
            suffix = '_'
        getShapeProfiles(self.CAT_DEST, self.VIZ_DEST, self.SNAP, self.D_LOGSTART, self.D_LOGEND, self.D_BINS, self.start_time, MASS_UNIT=1e10, suffix = suffix)
        
    def drawDensityProfiles(self, obj_type = ''):
        if obj_type == 'dm':
            suffix = '_dm_'
        elif obj_type == 'gx':
            suffix = '_gx_'
        else:
            assert obj_type == ''
            suffix = '_'
        getDensityProfiles(self.CAT_DEST, self.VIZ_DEST, self.SNAP, self.D_LOGSTART, self.D_LOGEND, self.D_BINS, self.start_time, MASS_UNIT=1e10, suffix = suffix)
    
    def plotLocalTHisto(self, obj_type = ''):
        if obj_type == 'dm':
            suffix = '_dm_'
        elif obj_type == 'gx':
            suffix = '_gx_'
        else:
            assert obj_type == ''
            suffix = '_'
        getLocalTHisto(self.CAT_DEST, self.VIZ_DEST, self.SNAP, self.D_LOGSTART, self.D_LOGEND, self.D_BINS, self.start_time, HIST_NB_BINS=11, MASS_UNIT=1e10, suffix = suffix, inner = False)
        
    def fitDensProfs(self, dens_profs, ROverR200, r200s, method = 'einasto'):
        # dens_profs are defined at ROverR200
        
        if method == 'einasto':
            best_fits = np.zeros((dens_profs.shape[0], 3))
        elif method == 'alpha_beta_gamma':
            best_fits = np.zeros((dens_profs.shape[0], 5))
        else:
            best_fits = np.zeros((dens_profs.shape[0], 2))
        dens_profs_plus_r200_plus_obj_number = np.c_[dens_profs, r200s]
        dens_profs_plus_r200_plus_obj_number = np.c_[dens_profs_plus_r200_plus_obj_number, np.arange(dens_profs.shape[0])]
        with Pool(processes=openmp.omp_get_max_threads()) as pool:
            results = pool.map(partial(fitDensProfHelper, ROverR200, method), [dens_profs_plus_r200_plus_obj_number[obj] for obj in range(dens_profs.shape[0])])
        for result in results:
            x, obj = tuple(result)
            best_fits[int(obj)] = x
        return best_fits

cdef class CosmicShapesDirect(CosmicShapes):
    
    cdef float[:,:] xyz
    cdef float[:] masses
    cdef int[:,:] cat_arr
    cdef float[:] r200
    cdef object cat

    def __init__(self, float[:,:] xyz, float[:] masses, cat, float[:] r200, str CAT_DEST, str VIZ_DEST, str SNAP, float L_BOX, int MIN_NUMBER_PTCS, int D_LOGSTART, int D_LOGEND, int D_BINS, float M_TOL, int N_WALL, int N_MIN, str CENTER, double start_time):
        super().__init__(CAT_DEST, VIZ_DEST, SNAP, L_BOX, MIN_NUMBER_PTCS, D_LOGSTART, D_LOGEND, D_BINS, M_TOL, N_WALL, N_MIN, CENTER, start_time)
        assert xyz.shape[0] == masses.shape[0], "xyz.shape[0] must be equal to masses.shape[0]"
        self.xyz = xyz
        self.masses = masses
        self.cat = cat
        self.r200 = r200
            
    def calcLocalShapes(self):   
        
        print_status(rank,self.start_time,'Starting calcLocalShapes() with snap {0}'.format(self.SNAP))
        
        if rank == 0:
                                        
                print_status(rank, self.start_time, "Number of halos is {0}. The number of valid halos (sufficient-resolution ones) is {1}".format(len(self.cat), np.array([0 for x in self.cat if x != []]).shape[0]))
                
                # Morphology: Local Shape
                print_status(rank, self.start_time, "Calculating local-shape morphologies with {0} processors. The average number of ptcs in the Halos is {1}".format(len(os.sched_getaffinity(0)), np.average(np.array(list(map(lambda x: len([x for x in self.cat if x != []][x]), range(len([x for x in self.cat if x != []]))))))))
                d, q, s, minor, inter, major, halos_center, halo_m, succeeded = self.getMorphLocal(self.xyz, self.cat, self.masses, self.r200, self.L_BOX, self.MIN_NUMBER_PTCS, self.D_LOGSTART, self.D_LOGEND, self.D_BINS, self.M_TOL, self.N_WALL, self.N_MIN, self.CENTER)
                print_status(rank, self.start_time, "Finished morphologies")
            
                if succeeded != []:
                    minor_re = minor.reshape(minor.shape[0], -1)
                    inter_re = inter.reshape(inter.shape[0], -1)
                    major_re = major.reshape(major.shape[0], -1)
                else:
                    minor_re = np.array([])
                    inter_re = np.array([])
                    major_re = np.array([])
                
                # Writing
                cat_local = [[] for i in range(len(self.cat))] # We are removing those halos whose R200 shell does not converge (including where R200 is not even available)
                for success in succeeded:
                    if self.r200[success] != 0.0: 
                        cat_local[success] = np.int32(self.cat[success]).tolist() # Json does not understand np.datatypes.
                with open('{0}/cat_local_{1}.txt'.format(self.CAT_DEST, self.SNAP), 'w') as filehandle:
                    json.dump(cat_local, filehandle)
                np.savetxt('{0}/d_local_{1}.txt'.format(self.CAT_DEST, self.SNAP), d, fmt='%1.7e')
                np.savetxt('{0}/q_local_{1}.txt'.format(self.CAT_DEST, self.SNAP), q, fmt='%1.7e')
                np.savetxt('{0}/s_local_{1}.txt'.format(self.CAT_DEST, self.SNAP), s, fmt='%1.7e')
                np.savetxt('{0}/minor_local_{1}.txt'.format(self.CAT_DEST, self.SNAP), minor_re, fmt='%1.7e')
                np.savetxt('{0}/inter_local_{1}.txt'.format(self.CAT_DEST, self.SNAP), inter_re, fmt='%1.7e')
                np.savetxt('{0}/major_local_{1}.txt'.format(self.CAT_DEST, self.SNAP), major_re, fmt='%1.7e')
                np.savetxt('{0}/m_local_{1}.txt'.format(self.CAT_DEST, self.SNAP), halo_m, fmt='%1.7e')
                np.savetxt('{0}/centers_local_{1}.txt'.format(self.CAT_DEST, self.SNAP), halos_center, fmt='%1.7e')
                
                # Clean-up
                del d; del q; del s; del minor; del inter; del major; del halos_center; del halo_m; del succeeded
                
    def calcGlobalShapes(self):
        
        print_status(rank,self.start_time,'Starting calcGlobalShapes() with snap {0}'.format(self.SNAP))
        
        if rank == 0:
                
                # Morphology: Global Shape (with E1 at large radius)
                print_status(rank, self.start_time, "Calculating global morphologies with {0} processors. The average number of ptcs in the Halos is {1}".format(len(os.sched_getaffinity(0)), np.average(np.array(list(map(lambda x: len([x for x in self.cat if x != []][x]), range(len([x for x in self.cat if x != []]))))))))
                d, q, s, minor, inter, major, halos_center, halo_m = self.getMorphGlobal(self.xyz, self.cat, self.masses, self.r200, self.L_BOX, self.MIN_NUMBER_PTCS, self.M_TOL, self.N_WALL, self.N_MIN, self.CENTER)
                print_status(rank, self.start_time, "Finished morphologies")
            
                if d.shape[0] != 0:
                    d = np.reshape(d, (d.shape[0], 1)) # Has shape (number_of_halos, 1)
                    q = np.reshape(q, (q.shape[0], 1)) # Has shape (number_of_halos, 1)
                    s = np.reshape(s, (s.shape[0], 1)) # Has shape (number_of_halos, 1)
                    minor = minor.reshape((d.shape[0],d.shape[1],3)) # Has shape (number_of_halos, 1, 3)
                    inter = inter.reshape((d.shape[0],d.shape[1],3)) # Has shape (number_of_halos, 1, 3)
                    major = major.reshape((d.shape[0],d.shape[1],3)) # Has shape (number_of_halos, 1, 3)
                    minor_re = minor.reshape(minor.shape[0], -1)
                    inter_re = inter.reshape(inter.shape[0], -1)
                    major_re = major.reshape(major.shape[0], -1)
                else:
                    minor_re = np.array([])
                    inter_re = np.array([])
                    major_re = np.array([])
                
                # Writing
                with open('{0}/cat_global_{1}.txt'.format(self.CAT_DEST, self.SNAP), 'w') as filehandle:
                    json.dump(self.cat, filehandle)
                np.savetxt('{0}/d_global_{1}.txt'.format(self.CAT_DEST, self.SNAP), d, fmt='%1.7e')
                np.savetxt('{0}/q_global_{1}.txt'.format(self.CAT_DEST, self.SNAP), q, fmt='%1.7e')
                np.savetxt('{0}/s_global_{1}.txt'.format(self.CAT_DEST, self.SNAP), s, fmt='%1.7e')
                np.savetxt('{0}/minor_global_{1}.txt'.format(self.CAT_DEST, self.SNAP), minor_re, fmt='%1.7e')
                np.savetxt('{0}/inter_global_{1}.txt'.format(self.CAT_DEST, self.SNAP), inter_re, fmt='%1.7e')
                np.savetxt('{0}/major_global_{1}.txt'.format(self.CAT_DEST, self.SNAP), np.float32(major_re), fmt='%1.7e')
                np.savetxt('{0}/m_global_{1}.txt'.format(self.CAT_DEST, self.SNAP), halo_m, fmt='%1.7e')
                np.savetxt('{0}/centers_global_{1}.txt'.format(self.CAT_DEST, self.SNAP), halos_center, fmt='%1.7e')
                
                # Clean-up
                del d; del q; del s; del minor; del inter; del major; del halos_center; del halo_m
    
    def vizLocalShapes(self, obj_numbers):
        if rank == 0:
            # Retrieve shape information
            with open('{0}/cat_local_{1}.txt'.format(self.CAT_DEST, self.SNAP), 'r') as filehandle:
                obj_cat_local = json.load(filehandle)
            minor = np.loadtxt('{0}/minor_local_{1}.txt'.format(self.CAT_DEST, self.SNAP))
            inter = np.loadtxt('{0}/inter_local_{1}.txt'.format(self.CAT_DEST, self.SNAP))
            major = np.loadtxt('{0}/major_local_{1}.txt'.format(self.CAT_DEST, self.SNAP))
            d = np.loadtxt('{0}/d_local_{1}.txt'.format(self.CAT_DEST, self.SNAP)) # Has shape (number_of_objs, self.D_BINS+1)
            q = np.loadtxt('{0}/q_local_{1}.txt'.format(self.CAT_DEST, self.SNAP))
            s = np.loadtxt('{0}/s_local_{1}.txt'.format(self.CAT_DEST, self.SNAP))
            centers = np.loadtxt('{0}/centers_local_{1}.txt'.format(self.CAT_DEST, self.SNAP))
            if major.ndim == 2:
                major = major.reshape(major.shape[0], major.shape[1]//3, 3) # Has shape (number_of_objs, self.D_BINS+1, 3)
                inter = inter.reshape(inter.shape[0], inter.shape[1]//3, 3) # Has shape (number_of_objs, self.D_BINS+1, 3)
                minor = minor.reshape(minor.shape[0], minor.shape[1]//3, 3) # Has shape (number_of_objs, self.D_BINS+1, 3)
            else:
                if major.shape[0] == (self.D_BINS+1)*3:
                    major = major.reshape(1, self.D_BINS+1, 3)
                    inter = inter.reshape(1, self.D_BINS+1, 3)
                    minor = minor.reshape(1, self.D_BINS+1, 3)
            # Dealing with the case of 1 obj
            if d.ndim == 1 and d.shape[0] == self.D_BINS+1:
                d = d.reshape(1, self.D_BINS+1)
                q = q.reshape(1, self.D_BINS+1)
                s = s.reshape(1, self.D_BINS+1)
                centers = centers.reshape(1, 3)
            for obj_number in obj_numbers:
                if obj_number >= major.shape[0]:
                    raise ValueError("obj_number exceeds the maximum number. There are only {0} objects".format(major.shape[0]))
                else:
                    major_obj = major[obj_number]
                    inter_obj = inter[obj_number]
                    minor_obj = minor[obj_number]
                    d_obj = d[obj_number]
                    q_obj = q[obj_number]
                    s_obj = s[obj_number]
                    center = centers[obj_number]
                    obj = np.zeros((len(obj_cat_local[obj_number]),3), dtype = np.float32)
                    masses_obj = np.zeros((len(obj_cat_local[obj_number]),), dtype = np.float32)
                    for idx, ptc in enumerate(obj_cat_local[obj_number]):
                        obj[idx] = self.xyz.base[ptc]
                        masses_obj[idx] = self.masses.base[ptc]
                    obj = respectPBCNoRef(obj, self.L_BOX)
                    # Plotting
                    fig = pyplot.figure()
                    ax = Axes3D(fig, auto_add_to_figure = False)
                    fig.add_axes(ax)
                    # If obj contains too many particles, choose some randomly for display
                    if len(obj) > 5000:
                        rng = default_rng(seed=42)
                        choose = rng.choice(np.arange(len(obj)), (5000,), replace = False)
                    else:
                        choose = np.arange(len(obj))
                    ax.scatter(obj[choose,0],obj[choose,1],obj[choose,2],s=masses_obj[choose]/np.average(masses_obj[choose])*0.3, label = "Particles")
                    ax.scatter(center[0],center[1],center[2],s=50,c="r", label = "COM")
                    
                    ell = fibonacci_ellipsoid(d_obj[-1], q_obj[-1]*d_obj[-1], s_obj[-1]*d_obj[-1], samples=500)
                    rot_matrix = np.hstack((np.reshape(major_obj[-1]/np.linalg.norm(major_obj[-1]), (3,1)), np.reshape(inter_obj[-1]/np.linalg.norm(inter_obj[-1]), (3,1)), np.reshape(minor_obj[-1]/np.linalg.norm(minor_obj[-1]), (3,1))))
                    for j in range(len(ell)): # Transformation into the principal frame
                        ell[j] = np.dot(rot_matrix, np.array(ell[j]))
                    ell_x = np.array([x[0] for x in ell])
                    ell_y = np.array([x[1] for x in ell])
                    ell_z = np.array([x[2] for x in ell])
                    
                    ax.scatter(ell_x+center[0],ell_y+center[1],ell_z+center[2],s=1, c="g", label = "Inferred; a = {:.2f}, b = {:.2f}, c = {:.2f}".format(d_obj[-1], q_obj[-1]*d_obj[-1], s_obj[-1]*d_obj[-1]))
                    for idx in np.arange(self.D_BINS-self.D_BINS//5, self.D_BINS):
                        if idx == self.D_BINS-1:
                            ax.quiver(*center, major_obj[idx][0], major_obj[idx][1], major_obj[idx][2], length=d_obj[idx], color='m', label= "Major")
                            ax.quiver(*center, inter_obj[idx][0], inter_obj[idx][1], inter_obj[idx][2], length=q_obj[idx]*d_obj[idx], color='c', label = "Intermediate")
                            ax.quiver(*center, minor_obj[idx][0], minor_obj[idx][1], minor_obj[idx][2], length=s_obj[idx]*d_obj[idx], color='y', label = "Minor")
                        else:
                            ax.quiver(*center, major_obj[idx][0], major_obj[idx][1], major_obj[idx][2], length=d_obj[idx], color='m')
                            ax.quiver(*center, inter_obj[idx][0], inter_obj[idx][1], inter_obj[idx][2], length=q_obj[idx]*d_obj[idx], color='c')
                            ax.quiver(*center, minor_obj[idx][0], minor_obj[idx][1], minor_obj[idx][2], length=s_obj[idx]*d_obj[idx], color='y')
                    for special in np.arange(-self.D_BINS//5,-self.D_BINS//5+1):
                        ell = fibonacci_ellipsoid(d_obj[special], q_obj[special]*d_obj[special], s_obj[special]*d_obj[special], samples=500)
                        rot_matrix = np.hstack((np.reshape(major_obj[special]/np.linalg.norm(major_obj[special]), (3,1)), np.reshape(inter_obj[special]/np.linalg.norm(inter_obj[special]), (3,1)), np.reshape(minor_obj[special]/np.linalg.norm(minor_obj[special]), (3,1))))
                        for j in range(len(ell)): # Transformation into the principal frame
                            ell[j] = np.dot(rot_matrix, np.array(ell[j])) 
                        ell_x = np.array([x[0] for x in ell])
                        ell_y = np.array([x[1] for x in ell])
                        ell_z = np.array([x[2] for x in ell])
                        ax.scatter(ell_x+center[0],ell_y+center[1],ell_z+center[2],s=1, c="r", label = "Inferred; a = {:.2f}, b = {:.2f}, c = {:.2f}".format(d_obj[special], q_obj[special]*d_obj[special], s_obj[special]*d_obj[special]))
                        ax.quiver(*center, major_obj[special][0], major_obj[special][1], major_obj[special][2], length=d_obj[special], color='limegreen', label= "Major {0}".format(special))
                        ax.quiver(*center, inter_obj[special][0], inter_obj[special][1], inter_obj[special][2], length=q_obj[special]*d_obj[special], color='darkorange', label = "Intermediate {0}".format(special))
                        ax.quiver(*center, minor_obj[special][0], minor_obj[special][1], minor_obj[special][2], length=s_obj[special]*d_obj[special], color='indigo', label = "Minor {0}".format(special))
                    fontP = FontProperties()
                    fontP.set_size('xx-small')
                    plt.legend(bbox_to_anchor=(0.95, 1), loc='upper right', prop=fontP)        
                    plt.xlabel(r"x (cMpc/h)")
                    plt.ylabel(r"y (cMpc/h)")
                    ax.set_zlabel(r"z (cMpc/h)")
                    ax.set_box_aspect([1,1,1])
                    set_axes_equal(ax)
                    fig.savefig("{}/LocalObj{}_{}.pdf".format(self.VIZ_DEST, obj_number, self.SNAP), bbox_inches='tight')
        
    def vizGlobalShapes(self, obj_numbers):
        if rank == 0:
            # Retrieve shape information
            with open('{0}/cat_global_{1}.txt'.format(self.CAT_DEST, self.SNAP), 'r') as filehandle:
                obj_cat_global = json.load(filehandle)
            minor = np.loadtxt('{0}/minor_global_{1}.txt'.format(self.CAT_DEST, self.SNAP))
            inter = np.loadtxt('{0}/inter_global_{1}.txt'.format(self.CAT_DEST, self.SNAP))
            major = np.loadtxt('{0}/major_global_{1}.txt'.format(self.CAT_DEST, self.SNAP))
            d = np.loadtxt('{0}/d_global_{1}.txt'.format(self.CAT_DEST, self.SNAP)) # Has shape (number_of_objs, )
            q = np.loadtxt('{0}/q_global_{1}.txt'.format(self.CAT_DEST, self.SNAP))
            s = np.loadtxt('{0}/s_global_{1}.txt'.format(self.CAT_DEST, self.SNAP))
            centers = np.loadtxt('{0}/centers_global_{1}.txt'.format(self.CAT_DEST, self.SNAP))
            d = np.array(d, ndmin=1) # To deal with possible 0-d arrays that show up if number_of_objs == 1
            q = np.array(q, ndmin=1) # To deal with possible 0-d arrays that show up if number_of_objs == 1
            s = np.array(s, ndmin=1) # To deal with possible 0-d arrays that show up if number_of_objs == 1
            d = d.reshape(d.shape[0], 1) # Has shape (number_of_objs, 1)
            q = q.reshape(q.shape[0], 1) # Has shape (number_of_objs, 1)
            s = s.reshape(s.shape[0], 1) # Has shape (number_of_objs, 1)
            centers = centers.reshape(centers.shape[0], 1) # Has shape (number_of_objs, 1)
            if major.ndim == 2:
                major = major.reshape(major.shape[0], major.shape[1]//3, 3) # Has shape (number_of_objs, 1, 3)
                inter = inter.reshape(inter.shape[0], inter.shape[1]//3, 3) # Has shape (number_of_objs, 1, 3)
                minor = minor.reshape(minor.shape[0], minor.shape[1]//3, 3) # Has shape (number_of_objs, 1, 3)
            else:
                if major.shape[0] == 3:
                    major = major.reshape(1, 1, 3)
                    inter = inter.reshape(1, 1, 3)
                    minor = minor.reshape(1, 1, 3)
            for obj_number in obj_numbers:
                if obj_number >= major.shape[0]:
                    raise ValueError("obj_number exceeds the maximum number. There are only {0} objects".format(major.shape[0]))
                else:
                    major_obj = major[obj_number]
                    inter_obj = inter[obj_number]
                    minor_obj = minor[obj_number]
                    d_obj = d[obj_number]
                    q_obj = q[obj_number]
                    s_obj = s[obj_number]
                    center = centers[obj_number]
                    obj = np.zeros((len(obj_cat_global[obj_number]),3), dtype = np.float32)
                    masses_obj = np.zeros((len(obj_cat_global[obj_number]),), dtype = np.float32)
                    for idx, ptc in enumerate(obj_cat_global[obj_number]):
                        obj[idx] = self.xyz.base[ptc]
                        masses_obj[idx] = self.masses.base[ptc]
                    obj = respectPBCNoRef(obj, self.L_BOX)
                    # Plotting
                    fig = pyplot.figure()
                    ax = Axes3D(fig, auto_add_to_figure = False)
                    fig.add_axes(ax)
                    # If obj contains too many particles, choose some randomly for display
                    if len(obj) > 5000:
                        rng = default_rng(seed=42)
                        choose = rng.choice(np.arange(len(obj)), (5000,), replace = False)
                    else:
                        choose = np.arange(len(obj))
                    ax.scatter(obj[choose,0],obj[choose,1],obj[choose,2],s=masses_obj[choose]/np.average(masses_obj[choose])*0.3, label = "Particles")
                    ax.scatter(center[0],center[1],center[2],s=50,c="r", label = "COM")
                    
                    ell = fibonacci_ellipsoid(d_obj[-1], q_obj[-1]*d_obj[-1], s_obj[-1]*d_obj[-1], samples=500)
                    rot_matrix = np.hstack((np.reshape(major_obj[-1]/np.linalg.norm(major_obj[-1]), (3,1)), np.reshape(inter_obj[-1]/np.linalg.norm(inter_obj[-1]), (3,1)), np.reshape(minor_obj[-1]/np.linalg.norm(minor_obj[-1]), (3,1))))
                    for j in range(len(ell)): # Transformation into the principal frame
                        ell[j] = np.dot(rot_matrix, np.array(ell[j]))
                    ell_x = np.array([x[0] for x in ell])
                    ell_y = np.array([x[1] for x in ell])
                    ell_z = np.array([x[2] for x in ell])
                    ax.scatter(ell_x+center[0],ell_y+center[1],ell_z+center[2],s=1, c="g", label = "Inferred; a = {:.2f}, b = {:.2f}, c = {:.2f}".format(d_obj[-1], q_obj[-1]*d_obj[-1], s_obj[-1]*d_obj[-1]))
                    for idx in np.arange(self.D_BINS-self.D_BINS//5, self.D_BINS):
                        if idx == self.D_BINS-1:
                            ax.quiver(*center, major_obj[idx][0], major_obj[idx][1], major_obj[idx][2], length=d_obj[idx], color='m', label= "Major")
                            ax.quiver(*center, inter_obj[idx][0], inter_obj[idx][1], inter_obj[idx][2], length=q_obj[idx]*d_obj[idx], color='c', label = "Intermediate")
                            ax.quiver(*center, minor_obj[idx][0], minor_obj[idx][1], minor_obj[idx][2], length=s_obj[idx]*d_obj[idx], color='y', label = "Minor")
                        else:
                            ax.quiver(*center, major_obj[idx][0], major_obj[idx][1], major_obj[idx][2], length=d_obj[idx], color='m')
                            ax.quiver(*center, inter_obj[idx][0], inter_obj[idx][1], inter_obj[idx][2], length=q_obj[idx]*d_obj[idx], color='c')
                            ax.quiver(*center, minor_obj[idx][0], minor_obj[idx][1], minor_obj[idx][2], length=s_obj[idx]*d_obj[idx], color='y')
                    for special in np.arange(-self.D_BINS//5,-self.D_BINS//5+1):
                        ell = fibonacci_ellipsoid(d_obj[special], q_obj[special]*d_obj[special], s_obj[special]*d_obj[special], samples=500)
                        rot_matrix = np.hstack((np.reshape(major_obj[special]/np.linalg.norm(major_obj[special]), (3,1)), np.reshape(inter_obj[special]/np.linalg.norm(inter_obj[special]), (3,1)), np.reshape(minor_obj[special]/np.linalg.norm(minor_obj[special]), (3,1))))
                        for j in range(len(ell)): # Transformation into the principal frame
                            ell[j] = np.dot(rot_matrix, np.array(ell[j])) 
                        ell_x = np.array([x[0] for x in ell])
                        ell_y = np.array([x[1] for x in ell])
                        ell_z = np.array([x[2] for x in ell])
                        ax.scatter(ell_x+center[0],ell_y+center[1],ell_z+center[2],s=1, c="r", label = "Inferred; a = {:.2f}, b = {:.2f}, c = {:.2f}".format(d_obj[special], q_obj[special]*d_obj[special], s_obj[special]*d_obj[special]))
                        ax.quiver(*center, major_obj[special][0], major_obj[special][1], major_obj[special][2], length=d_obj[special], color='limegreen', label= "Major {0}".format(special))
                        ax.quiver(*center, inter_obj[special][0], inter_obj[special][1], inter_obj[special][2], length=q_obj[special]*d_obj[special], color='darkorange', label = "Intermediate {0}".format(special))
                        ax.quiver(*center, minor_obj[special][0], minor_obj[special][1], minor_obj[special][2], length=s_obj[special]*d_obj[special], color='indigo', label = "Minor {0}".format(special))
                    else:
                        ax.quiver(*center, major_obj[-1][0], major_obj[-1][1], major_obj[-1][2], length=d_obj[-1], color='m', label= "Major")
                        ax.quiver(*center, inter_obj[-1][0], inter_obj[-1][1], inter_obj[-1][2], length=q_obj[-1]*d_obj[-1], color='c', label = "Intermediate")
                        ax.quiver(*center, minor_obj[-1][0], minor_obj[-1][1], minor_obj[-1][2], length=s_obj[-1]*d_obj[-1], color='y', label = "Minor")
                      
                    fontP = FontProperties()
                    fontP.set_size('xx-small')
                    plt.legend(bbox_to_anchor=(0.95, 1), loc='upper right', prop=fontP)  
                    plt.xlabel(r"x (cMpc/h)")
                    plt.ylabel(r"y (cMpc/h)")
                    ax.set_zlabel(r"z (cMpc/h)")
                    ax.set_box_aspect([1,1,1])
                    set_axes_equal(ax)
                    fig.savefig("{}/GlobalObj{}_{}.pdf".format(self.VIZ_DEST, obj_number, self.SNAP), bbox_inches='tight')
                    
    def plotGlobalEpsHisto(self):
        with open('{0}/cat_global_{1}.txt'.format(self.CAT_DEST, self.SNAP), 'r') as filehandle:
            cat = json.load(filehandle)
        suffix = '_'
        getGlobalEpsHisto(cat, self.xyz, self.masses, self.L_BOX, self.VIZ_DEST, self.SNAP, suffix = suffix, HIST_NB_BINS = 11)

    def calcDensProfsDirectBinning(self, ROverR200):
        
        obj_keep = np.int32([1 if x != [] else 0 for x in self.cat])
        ROverR200, dens_profs = self.getDensProfsDirectBinning(self.cat, self.xyz, obj_keep, self.masses, self.r200, np.float32(ROverR200), self.L_BOX, self.CENTER)
        return dens_profs
    
    def calcDensProfsKernelBased(self, ROverR200):
        
        obj_keep = np.int32([1 if x != [] else 0 for x in self.cat])
        ROverR200, dens_profs = self.getDensProfsKernelBased(self.cat, self.xyz, obj_keep, self.masses, self.r200, np.float32(ROverR200), self.L_BOX, self.CENTER)
        return dens_profs
        

cdef class CosmicShapesGadgetHDF5(CosmicShapes):
    
    cdef str HDF5_SNAP_DEST
    cdef str HDF5_GROUP_DEST
    cdef int MIN_NUMBER_STAR_PTCS
    cdef int SNAP_MAX
    cdef float[:] r200
    
    def __init__(self, str HDF5_SNAP_DEST, str HDF5_GROUP_DEST, str CAT_DEST, str VIZ_DEST, str SNAP, int SNAP_MAX, float L_BOX, int MIN_NUMBER_PTCS, int MIN_NUMBER_STAR_PTCS, int D_LOGSTART, int D_LOGEND, int D_BINS, float M_TOL, int N_WALL, int N_MIN, str CENTER, double start_time):
        super().__init__(CAT_DEST, VIZ_DEST, SNAP, L_BOX, MIN_NUMBER_PTCS, D_LOGSTART, D_LOGEND, D_BINS, M_TOL, N_WALL, N_MIN, CENTER, start_time)
        self.HDF5_SNAP_DEST = HDF5_SNAP_DEST
        self.HDF5_GROUP_DEST = HDF5_GROUP_DEST
        self.MIN_NUMBER_STAR_PTCS = MIN_NUMBER_STAR_PTCS
        self.SNAP_MAX = SNAP_MAX
            
    def vizLocalShapes(self, obj_numbers, obj_type = 'dm'):
        if obj_type == 'dm':
            xyz, masses, dm_smoothing, dm_velxyz = getHDF5DMData(self.HDF5_SNAP_DEST, self.SNAP_MAX, self.SNAP)
            del dm_smoothing; del dm_velxyz
        else:
            xyz, fof_com, sh_com, nb_shs, masses, star_smoothing, is_star = getHDF5GxData(self.HDF5_SNAP_DEST, self.HDF5_GROUP_DEST, self.SNAP_MAX, self.SNAP)
            del fof_com; del sh_com; del nb_shs; del star_smoothing; del is_star
        if rank == 0:
            # Retrieve shape information for obj_type
            with open('{0}/cat_local_{1}_{2}.txt'.format(self.CAT_DEST, obj_type, self.SNAP), 'r') as filehandle:
                obj_cat_local = json.load(filehandle)
            minor = np.loadtxt('{0}/minor_local_{1}_{2}.txt'.format(self.CAT_DEST, obj_type, self.SNAP))
            inter = np.loadtxt('{0}/inter_local_{1}_{2}.txt'.format(self.CAT_DEST, obj_type, self.SNAP))
            major = np.loadtxt('{0}/major_local_{1}_{2}.txt'.format(self.CAT_DEST, obj_type, self.SNAP))
            d = np.loadtxt('{0}/d_local_{1}_{2}.txt'.format(self.CAT_DEST, obj_type, self.SNAP)) # Has shape (number_of_objs, self.D_BINS+1)
            q = np.loadtxt('{0}/q_local_{1}_{2}.txt'.format(self.CAT_DEST, obj_type, self.SNAP))
            s = np.loadtxt('{0}/s_local_{1}_{2}.txt'.format(self.CAT_DEST, obj_type, self.SNAP))
            centers = np.loadtxt('{0}/centers_local_{1}_{2}.txt'.format(self.CAT_DEST, obj_type, self.SNAP))
            if major.ndim == 2:
                major = major.reshape(major.shape[0], major.shape[1]//3, 3) # Has shape (number_of_objs, self.D_BINS+1, 3)
                inter = inter.reshape(inter.shape[0], inter.shape[1]//3, 3) # Has shape (number_of_objs, self.D_BINS+1, 3)
                minor = minor.reshape(minor.shape[0], minor.shape[1]//3, 3) # Has shape (number_of_objs, self.D_BINS+1, 3)
            else:
                if major.shape[0] == (self.D_BINS+1)*3:
                    major = major.reshape(1, self.D_BINS+1, 3)
                    inter = inter.reshape(1, self.D_BINS+1, 3)
                    minor = minor.reshape(1, self.D_BINS+1, 3)
            # Dealing with the case of 1 obj
            if d.ndim == 1 and d.shape[0] == self.D_BINS+1:
                d = d.reshape(1, self.D_BINS+1)
                q = q.reshape(1, self.D_BINS+1)
                s = s.reshape(1, self.D_BINS+1)
                centers = centers.reshape(1, 3)
            for obj_number in obj_numbers:
                if obj_number >= major.shape[0]:
                    raise ValueError("obj_number exceeds the maximum number. There are only {0} objects".format(major.shape[0]))
                else:
                    major_obj = major[obj_number]
                    inter_obj = inter[obj_number]
                    minor_obj = minor[obj_number]
                    d_obj = d[obj_number]
                    q_obj = q[obj_number]
                    s_obj = s[obj_number]
                    center = centers[obj_number]
                    obj = np.zeros((len(obj_cat_local[obj_number]),3), dtype = np.float32)
                    masses_obj = np.zeros((len(obj_cat_local[obj_number]),), dtype = np.float32)
                    for idx, ptc in enumerate(obj_cat_local[obj_number]):
                        obj[idx] = xyz[ptc]
                        masses_obj[idx] = masses[ptc]
                    obj = respectPBCNoRef(obj, self.L_BOX)
                    # Plotting
                    fig = pyplot.figure()
                    ax = Axes3D(fig, auto_add_to_figure = False)
                    fig.add_axes(ax)
                    # If obj contains too many particles, choose some randomly for display
                    if len(obj) > 5000:
                        rng = default_rng(seed=42)
                        choose = rng.choice(np.arange(len(obj)), (5000,), replace = False)
                    else:
                        choose = np.arange(len(obj))
                    ax.scatter(obj[choose,0],obj[choose,1],obj[choose,2],s=masses_obj[choose]/np.average(masses_obj[choose])*0.3, label = "Particles")
                    ax.scatter(center[0],center[1],center[2],s=50,c="r", label = "COM")
                    
                    ell = fibonacci_ellipsoid(d_obj[-1], q_obj[-1]*d_obj[-1], s_obj[-1]*d_obj[-1], samples=500)
                    rot_matrix = np.hstack((np.reshape(major_obj[-1]/np.linalg.norm(major_obj[-1]), (3,1)), np.reshape(inter_obj[-1]/np.linalg.norm(inter_obj[-1]), (3,1)), np.reshape(minor_obj[-1]/np.linalg.norm(minor_obj[-1]), (3,1))))
                    for j in range(len(ell)): # Transformation into the principal frame
                        ell[j] = np.dot(rot_matrix, np.array(ell[j]))
                    ell_x = np.array([x[0] for x in ell])
                    ell_y = np.array([x[1] for x in ell])
                    ell_z = np.array([x[2] for x in ell])
                    ax.scatter(ell_x+center[0],ell_y+center[1],ell_z+center[2],s=1, c="g", label = "Inferred; a = {:.2f}, b = {:.2f}, c = {:.2f}".format(d_obj[-1], q_obj[-1]*d_obj[-1], s_obj[-1]*d_obj[-1]))
                    for idx in np.arange(self.D_BINS-self.D_BINS//5, self.D_BINS):
                        if idx == self.D_BINS-1:
                            ax.quiver(*center, major_obj[idx][0], major_obj[idx][1], major_obj[idx][2], length=d_obj[idx], color='m', label= "Major")
                            ax.quiver(*center, inter_obj[idx][0], inter_obj[idx][1], inter_obj[idx][2], length=q_obj[idx]*d_obj[idx], color='c', label = "Intermediate")
                            ax.quiver(*center, minor_obj[idx][0], minor_obj[idx][1], minor_obj[idx][2], length=s_obj[idx]*d_obj[idx], color='y', label = "Minor")
                        else:
                            ax.quiver(*center, major_obj[idx][0], major_obj[idx][1], major_obj[idx][2], length=d_obj[idx], color='m')
                            ax.quiver(*center, inter_obj[idx][0], inter_obj[idx][1], inter_obj[idx][2], length=q_obj[idx]*d_obj[idx], color='c')
                            ax.quiver(*center, minor_obj[idx][0], minor_obj[idx][1], minor_obj[idx][2], length=s_obj[idx]*d_obj[idx], color='y')
                    for special in np.arange(-self.D_BINS//5,-self.D_BINS//5+1):
                        ell = fibonacci_ellipsoid(d_obj[special], q_obj[special]*d_obj[special], s_obj[special]*d_obj[special], samples=500)
                        rot_matrix = np.hstack((np.reshape(major_obj[special]/np.linalg.norm(major_obj[special]), (3,1)), np.reshape(inter_obj[special]/np.linalg.norm(inter_obj[special]), (3,1)), np.reshape(minor_obj[special]/np.linalg.norm(minor_obj[special]), (3,1))))
                        for j in range(len(ell)): # Transformation into the principal frame
                            ell[j] = np.dot(rot_matrix, np.array(ell[j])) 
                        ell_x = np.array([x[0] for x in ell])
                        ell_y = np.array([x[1] for x in ell])
                        ell_z = np.array([x[2] for x in ell])
                        ax.scatter(ell_x+center[0],ell_y+center[1],ell_z+center[2],s=1, c="r", label = "Inferred; a = {:.2f}, b = {:.2f}, c = {:.2f}".format(d_obj[special], q_obj[special]*d_obj[special], s_obj[special]*d_obj[special]))
                        ax.quiver(*center, major_obj[special][0], major_obj[special][1], major_obj[special][2], length=d_obj[special], color='limegreen', label= "Major {0}".format(special))
                        ax.quiver(*center, inter_obj[special][0], inter_obj[special][1], inter_obj[special][2], length=q_obj[special]*d_obj[special], color='darkorange', label = "Intermediate {0}".format(special))
                        ax.quiver(*center, minor_obj[special][0], minor_obj[special][1], minor_obj[special][2], length=s_obj[special]*d_obj[special], color='indigo', label = "Minor {0}".format(special))
                    fontP = FontProperties()
                    fontP.set_size('xx-small')
                    plt.legend(bbox_to_anchor=(0.95, 1), loc='upper right', prop=fontP)        
                    plt.xlabel(r"x (cMpc/h)")
                    plt.ylabel(r"y (cMpc/h)")
                    ax.set_zlabel(r"z (cMpc/h)")
                    ax.set_box_aspect([1,1,1])
                    set_axes_equal(ax)
                    fig.savefig("{}/Local{}Obj{}_{}.pdf".format(self.VIZ_DEST, obj_type.upper(), obj_number, self.SNAP), bbox_inches='tight')
        
    def vizGlobalShapes(self, obj_numbers, obj_type = 'dm'):
        if obj_type == 'dm':
            xyz, masses, dm_smoothing, dm_velxyz = getHDF5DMData(self.HDF5_SNAP_DEST, self.SNAP_MAX, self.SNAP)
            del dm_smoothing; del dm_velxyz
        else:
            xyz, fof_com, sh_com, nb_shs, masses, star_smoothing, is_star = getHDF5GxData(self.HDF5_SNAP_DEST, self.HDF5_GROUP_DEST, self.SNAP_MAX, self.SNAP)
            del fof_com; del sh_com; del nb_shs; del star_smoothing; del is_star
        if rank == 0:
            # Retrieve shape information for obj_type
            with open('{0}/cat_global_{1}_{2}.txt'.format(self.CAT_DEST, obj_type, self.SNAP), 'r') as filehandle:
                obj_cat_global = json.load(filehandle)
            minor = np.loadtxt('{0}/minor_global_{1}_{2}.txt'.format(self.CAT_DEST, obj_type, self.SNAP))
            inter = np.loadtxt('{0}/inter_global_{1}_{2}.txt'.format(self.CAT_DEST, obj_type, self.SNAP))
            major = np.loadtxt('{0}/major_global_{1}_{2}.txt'.format(self.CAT_DEST, obj_type, self.SNAP))
            d = np.loadtxt('{0}/d_global_{1}_{2}.txt'.format(self.CAT_DEST, obj_type, self.SNAP)) # Has shape (number_of_objs, )
            q = np.loadtxt('{0}/q_global_{1}_{2}.txt'.format(self.CAT_DEST, obj_type, self.SNAP))
            s = np.loadtxt('{0}/s_global_{1}_{2}.txt'.format(self.CAT_DEST, obj_type, self.SNAP))
            centers = np.loadtxt('{0}/centers_global_{1}_{2}.txt'.format(self.CAT_DEST, obj_type, self.SNAP))
            d = np.array(d, ndmin=1) # To deal with possible 0-d arrays that show up if number_of_objs == 1
            q = np.array(q, ndmin=1) # To deal with possible 0-d arrays that show up if number_of_objs == 1
            s = np.array(s, ndmin=1) # To deal with possible 0-d arrays that show up if number_of_objs == 1
            d = d.reshape(d.shape[0], 1) # Has shape (number_of_objs, 1)
            q = q.reshape(q.shape[0], 1) # Has shape (number_of_objs, 1)
            s = s.reshape(s.shape[0], 1) # Has shape (number_of_objs, 1)
            if major.ndim == 2:
                major = major.reshape(major.shape[0], major.shape[1]//3, 3) # Has shape (number_of_objs, 1, 3)
                inter = inter.reshape(inter.shape[0], inter.shape[1]//3, 3) # Has shape (number_of_objs, 1, 3)
                minor = minor.reshape(minor.shape[0], minor.shape[1]//3, 3) # Has shape (number_of_objs, 1, 3)
            else:
                if major.shape[0] == 3:
                    major = major.reshape(1, 1, 3)
                    inter = inter.reshape(1, 1, 3)
                    minor = minor.reshape(1, 1, 3)
            for obj_number in obj_numbers:
                if obj_number >= major.shape[0]:
                    raise ValueError("obj_number exceeds the maximum number. There are only {0} objects".format(major.shape[0]))
                else:
                    major_obj = major[obj_number]
                    inter_obj = inter[obj_number]
                    minor_obj = minor[obj_number]
                    d_obj = d[obj_number]
                    q_obj = q[obj_number]
                    s_obj = s[obj_number]
                    center = centers[obj_number]
                    obj = np.zeros((len(obj_cat_global[obj_number]),3), dtype = np.float32)
                    masses_obj = np.zeros((len(obj_cat_global[obj_number]),), dtype = np.float32)
                    for idx, ptc in enumerate(obj_cat_global[obj_number]):
                        obj[idx] = xyz[ptc]
                        masses_obj[idx] = masses[ptc]
                    obj = respectPBCNoRef(obj, self.L_BOX)
                    # Plotting
                    fig = pyplot.figure()
                    ax = Axes3D(fig, auto_add_to_figure = False)
                    fig.add_axes(ax)
                    # If obj contains too many particles, choose some randomly for display
                    if len(obj) > 5000:
                        rng = default_rng(seed=42)
                        choose = rng.choice(np.arange(len(obj)), (5000,), replace = False)
                    else:
                        choose = np.arange(len(obj))
                    ax.scatter(obj[choose,0],obj[choose,1],obj[choose,2],s=masses_obj[choose]/np.average(masses_obj[choose])*0.3, label = "Particles")
                    ax.scatter(center[0],center[1],center[2],s=50,c="r", label = "COM")
                    
                    ell = fibonacci_ellipsoid(d_obj[-1], q_obj[-1]*d_obj[-1], s_obj[-1]*d_obj[-1], samples=500)
                    rot_matrix = np.hstack((np.reshape(major_obj[-1]/np.linalg.norm(major_obj[-1]), (3,1)), np.reshape(inter_obj[-1]/np.linalg.norm(inter_obj[-1]), (3,1)), np.reshape(minor_obj[-1]/np.linalg.norm(minor_obj[-1]), (3,1))))
                    for j in range(len(ell)): # Transformation into the principal frame
                        ell[j] = np.dot(rot_matrix, np.array(ell[j]))
                    ell_x = np.array([x[0] for x in ell])
                    ell_y = np.array([x[1] for x in ell])
                    ell_z = np.array([x[2] for x in ell])
                    ax.scatter(ell_x+center[0],ell_y+center[1],ell_z+center[2],s=1, c="g", label = "Inferred; a = {:.2f}, b = {:.2f}, c = {:.2f}".format(d_obj[-1], q_obj[-1]*d_obj[-1], s_obj[-1]*d_obj[-1]))
                    for idx in np.arange(self.D_BINS-self.D_BINS//5, self.D_BINS):
                        if idx == self.D_BINS-1:
                            ax.quiver(*center, major_obj[idx][0], major_obj[idx][1], major_obj[idx][2], length=d_obj[idx], color='m', label= "Major")
                            ax.quiver(*center, inter_obj[idx][0], inter_obj[idx][1], inter_obj[idx][2], length=q_obj[idx]*d_obj[idx], color='c', label = "Intermediate")
                            ax.quiver(*center, minor_obj[idx][0], minor_obj[idx][1], minor_obj[idx][2], length=s_obj[idx]*d_obj[idx], color='y', label = "Minor")
                        else:
                            ax.quiver(*center, major_obj[idx][0], major_obj[idx][1], major_obj[idx][2], length=d_obj[idx], color='m')
                            ax.quiver(*center, inter_obj[idx][0], inter_obj[idx][1], inter_obj[idx][2], length=q_obj[idx]*d_obj[idx], color='c')
                            ax.quiver(*center, minor_obj[idx][0], minor_obj[idx][1], minor_obj[idx][2], length=s_obj[idx]*d_obj[idx], color='y')
                    for special in np.arange(-self.D_BINS//5,-self.D_BINS//5+1):
                        ell = fibonacci_ellipsoid(d_obj[special], q_obj[special]*d_obj[special], s_obj[special]*d_obj[special], samples=500)
                        rot_matrix = np.hstack((np.reshape(major_obj[special]/np.linalg.norm(major_obj[special]), (3,1)), np.reshape(inter_obj[special]/np.linalg.norm(inter_obj[special]), (3,1)), np.reshape(minor_obj[special]/np.linalg.norm(minor_obj[special]), (3,1))))
                        for j in range(len(ell)): # Transformation into the principal frame
                            ell[j] = np.dot(rot_matrix, np.array(ell[j])) 
                        ell_x = np.array([x[0] for x in ell])
                        ell_y = np.array([x[1] for x in ell])
                        ell_z = np.array([x[2] for x in ell])
                        ax.scatter(ell_x+center[0],ell_y+center[1],ell_z+center[2],s=1, c="r", label = "Inferred; a = {:.2f}, b = {:.2f}, c = {:.2f}".format(d_obj[special], q_obj[special]*d_obj[special], s_obj[special]*d_obj[special]))
                        ax.quiver(*center, major_obj[special][0], major_obj[special][1], major_obj[special][2], length=d_obj[special], color='limegreen', label= "Major {0}".format(special))
                        ax.quiver(*center, inter_obj[special][0], inter_obj[special][1], inter_obj[special][2], length=q_obj[special]*d_obj[special], color='darkorange', label = "Intermediate {0}".format(special))
                        ax.quiver(*center, minor_obj[special][0], minor_obj[special][1], minor_obj[special][2], length=s_obj[special]*d_obj[special], color='indigo', label = "Minor {0}".format(special))
                    else:
                        ax.quiver(*center, major_obj[-1][0], major_obj[-1][1], major_obj[-1][2], length=d_obj[-1], color='m', label= "Major")
                        ax.quiver(*center, inter_obj[-1][0], inter_obj[-1][1], inter_obj[-1][2], length=q_obj[-1]*d_obj[-1], color='c', label = "Intermediate")
                        ax.quiver(*center, minor_obj[-1][0], minor_obj[-1][1], minor_obj[-1][2], length=s_obj[-1]*d_obj[-1], color='y', label = "Minor")
                      
                    fontP = FontProperties()
                    fontP.set_size('xx-small')
                    plt.legend(bbox_to_anchor=(0.95, 1), loc='upper right', prop=fontP)   
                    plt.xlabel(r"x (cMpc/h)")
                    plt.ylabel(r"y (cMpc/h)")
                    ax.set_zlabel(r"z (cMpc/h)")
                    ax.set_box_aspect([1,1,1])
                    set_axes_equal(ax)
                    fig.savefig("{}/Global{}Obj{}_{}.pdf".format(self.VIZ_DEST, obj_type.upper(), obj_number, self.SNAP), bbox_inches='tight')
    
    def loadDMCat(self):
        
        print_status(rank,self.start_time,'Starting loadDMCat()')
        
        # Import hdf5 data
        print_status(rank,self.start_time,"Getting HDF5 raw data..")
        if rank == 0:
            nb_shs, sh_len, fof_dm_sizes, group_r200, halo_masses, fof_coms = getHDF5SHDMData(self.HDF5_GROUP_DEST, self.SNAP_MAX, self.SNAP)
            del fof_coms
            print_status(rank, self.start_time, "Finished HDF5 raw data")
            
            # Construct catalogue
            print_status(rank, self.start_time, "Call getCSHCat()")
            h_cat, h_r200, h_pass = getCSHCat(np.array(nb_shs), np.array(sh_len), np.array(fof_dm_sizes), group_r200, halo_masses, self.MIN_NUMBER_PTCS)
            print_status(rank, self.start_time, "Finished getCSHCat()")
            nb_shs_vec = np.array(nb_shs)
            h_cat_l = [[] for i in range(len(nb_shs))]
            corr = 0
            for i in range(len(nb_shs)):
                if h_pass[i] == 1:
                    h_cat_l[i] = (np.ma.masked_where(h_cat[i-corr] == 0, h_cat[i-corr]).compressed()-1).tolist()
                else:
                    corr += 1
            print_status(rank, self.start_time, "Constructed the CSH catalogue. The total number of halos with > 0 SHs is {0}, the total number of halos is {1}, the total number of SHs is {2}, the number of halos that have no SH is {3} and the total number of halos (CSH) that have sufficient resolution is {4}".format(nb_shs_vec[nb_shs_vec != 0].shape[0], len(nb_shs), len(sh_len), nb_shs_vec[nb_shs_vec == 0].shape[0], len([x for x in h_cat_l if x != []])))
            
            # Writing
            with open('{0}/h_cat_{1}.txt'.format(self.CAT_DEST, self.SNAP), 'w') as filehandle:
                json.dump(h_cat_l, filehandle)
            self.r200 = h_r200
            del nb_shs; del sh_len; del fof_dm_sizes; del group_r200; del h_cat; del halo_masses; del h_r200
    
    def loadGxCat(self):        
        print_status(rank,self.start_time,'Starting loadGxCat() with snap {0}'.format(self.SNAP))
        
        # Import hdf5 data
        print_status(rank,self.start_time,"Getting HDF5 raw data..")
        nb_shs, sh_len_gx, fof_gx_sizes = getHDF5SHGxData(self.HDF5_GROUP_DEST, self.SNAP_MAX, self.SNAP)
        if rank != 0:
            del nb_shs; del sh_len_gx; del fof_gx_sizes
        if rank == 0:
            # Construct gx catalogue
            print_status(rank, self.start_time, "Creating Gx CAT..")
            gx_cat, gx_pass = getGxCat(np.array(nb_shs), np.array(sh_len_gx), np.array(fof_gx_sizes), self.MIN_NUMBER_PTCS)
            print_status(rank, self.start_time, "Finished getGxCat()")
            gx_cat_l = [[] for i in range(len(nb_shs))]
            corr = 0
            for i in range(len(nb_shs)):
                if gx_pass[i] == 1:
                    gx_cat_l[i] = (np.ma.masked_where(gx_cat[i-corr] == 0, gx_cat[i-corr]).compressed()-1).tolist()
                else:
                    corr += 1
            print_status(rank, self.start_time, "Constructed the gx catalogue. The number of valid gxs (after discarding low-resolution ones) is {0}.".format(np.array([0 for x in gx_cat_l if x != []]).shape[0]))
            
            # Writing
            with open('{0}/gx_cat_{1}.txt'.format(self.CAT_DEST, self.SNAP), 'w') as filehandle:
                json.dump(gx_cat_l, filehandle)
            del nb_shs; del sh_len_gx; del fof_gx_sizes; del gx_cat
    
    def calcLocalShapesGx(self):
        
        print_status(rank,self.start_time,'Starting calcLocalShapesGx() with snap {0}'.format(self.SNAP))
        
        # Import hdf5 data
        print_status(rank,self.start_time,"Getting HDF5 raw data..")
        star_xyz, fof_com, sh_com, nb_shs, star_masses, star_smoothing, is_star = getHDF5GxData(self.HDF5_SNAP_DEST, self.HDF5_GROUP_DEST, self.SNAP_MAX, self.SNAP)
        del star_smoothing
        if rank != 0:
            del star_xyz; del fof_com; del sh_com; del nb_shs; del star_masses; del is_star
        if rank == 0:
                                    
            # Defining galaxies: Method 1: 1 halo = at most 1 galaxy
            print_status(rank, self.start_time, "Loading Gx CAT..")
            with open('{0}/gx_cat_{1}.txt'.format(self.CAT_DEST, self.SNAP), 'r') as filehandle:
                gx_cat = json.load(filehandle)
            print_status(rank, self.start_time, "Loaded Gx CAT. The number of valid gxs (after discarding low-resolution ones) is {0}.".format(np.array([0 for x in gx_cat if x != []]).shape[0]))
            
            # Morphology: Local Shape
            print_status(rank, self.start_time, "Calculating local-shape morphologies with {0} processors. The average number of ptcs in the gxs is {1}".format(len(os.sched_getaffinity(0)), np.average(np.array(list(map(lambda x: len([x for x in gx_cat if x != []][x]), range(len([x for x in gx_cat if x != []]))))))))
            d, q, s, minor, inter, major, gx_center, gx_m, success = self.getMorphLocal(star_xyz, gx_cat, star_masses, self.r200, self.L_BOX, self.MIN_NUMBER_STAR_PTCS, self.D_LOGSTART, self.D_LOGEND, self.D_BINS, self.M_TOL, self.N_WALL, self.N_MIN, self.CENTER)
            print_status(rank, self.start_time, "Finished morphologies")
        
            if success != []:
                minor_re = minor.reshape(minor.shape[0], -1)
                inter_re = inter.reshape(inter.shape[0], -1)
                major_re = major.reshape(major.shape[0], -1)
            else:
                minor_re = np.array([])
                inter_re = np.array([])
                major_re = np.array([])
            
            # Writing
            cat_local = [[] for i in range(len(gx_cat))] # We are removing those gxs whose R200 shell does not converge (including where R200 is not even available)
            for su in success:
                if self.r200[su] != 0.0: 
                    cat_local[su] = gx_cat[su]
            with open('{0}/cat_local_gx_{1}.txt'.format(self.CAT_DEST, self.SNAP), 'w') as filehandle:
                json.dump(cat_local, filehandle)
            np.savetxt('{0}/d_local_gx_{1}.txt'.format(self.CAT_DEST, self.SNAP), d, fmt='%1.7e')
            np.savetxt('{0}/q_local_gx_{1}.txt'.format(self.CAT_DEST, self.SNAP), q, fmt='%1.7e')
            np.savetxt('{0}/s_local_gx_{1}.txt'.format(self.CAT_DEST, self.SNAP), s, fmt='%1.7e')
            np.savetxt('{0}/minor_local_gx_{1}.txt'.format(self.CAT_DEST, self.SNAP), minor_re, fmt='%1.7e')
            np.savetxt('{0}/inter_local_gx_{1}.txt'.format(self.CAT_DEST, self.SNAP), inter_re, fmt='%1.7e')
            np.savetxt('{0}/major_local_gx_{1}.txt'.format(self.CAT_DEST, self.SNAP), major_re, fmt='%1.7e')
            np.savetxt('{0}/m_local_gx_{1}.txt'.format(self.CAT_DEST, self.SNAP), gx_m, fmt='%1.7e')
            np.savetxt('{0}/centers_local_gx_{1}.txt'.format(self.CAT_DEST, self.SNAP), gx_center, fmt='%1.7e')
        
            # Clean-up
            del d; del q; del s; del minor; del inter; del major; del gx_center; del gx_m; del success; del star_xyz; del star_masses # Note: del gx_cat here yields ! marks further up!
            
    def calcGlobalShapesGx(self):
        
        # Import hdf5 data
        print_status(rank,self.start_time,"Getting HDF5 raw data..")
        star_xyz, fof_com, sh_com, nb_shs, star_masses, star_smoothing, is_star = getHDF5GxData(self.HDF5_SNAP_DEST, self.HDF5_GROUP_DEST, self.SNAP_MAX, self.SNAP)
        del star_smoothing
        if rank != 0:
            del star_xyz; del fof_com; del sh_com; del nb_shs; del star_masses; del is_star
        if rank == 0:
            with open('{0}/h_cat_{1}.txt'.format(self.CAT_DEST, self.SNAP), 'r') as filehandle:
                h_cat = json.load(filehandle)
            print_status(rank, self.start_time, "Got HDF5 raw data. Number of halos is {0}. The number of valid halos (sufficient-resolution ones) is {1}".format(fof_com.shape[0], np.array([0 for x in h_cat if x != []]).shape[0]))
            
            # Load galaxies
            print_status(rank, self.start_time, "Loading Gx CAT..")
            with open('{0}/gx_cat_{1}.txt'.format(self.CAT_DEST, self.SNAP), 'r') as filehandle:
                gx_cat = json.load(filehandle)
            print_status(rank, self.start_time, "Loaded Gx CAT. The number of valid gxs (after discarding low-resolution ones) is {0}. Out of {1}, we discarded some star particles (star particles closer to SH that isn't CSH)".format(np.array([0 for x in gx_cat if x != []]).shape[0], star_xyz.shape[0]))
            
            # Morphology: Global Shape (with E1 at large radius)
            print_status(rank, self.start_time, "Calculating global morphologies with {0} processors. The average number of ptcs in the gxs is {1}".format(len(os.sched_getaffinity(0)), np.average(np.array(list(map(lambda x: len([x for x in gx_cat if x != []][x]), range(len([x for x in gx_cat if x != []]))))))))
            d, q, s, minor, inter, major, gx_center, gx_m = self.getMorphGlobal(star_xyz, gx_cat, star_masses, self.r200, self.L_BOX, self.MIN_NUMBER_STAR_PTCS, self.M_TOL, self.N_WALL, self.N_MIN, self.CENTER)
            print_status(rank, self.start_time, "Finished morphologies")
            
            if d.shape[0] != 0:
                d = np.reshape(d, (d.shape[0], 1)) # Has shape (number_of_gxs, 1)
                q = np.reshape(q, (q.shape[0], 1)) # Has shape (number_of_gxs, 1)
                s = np.reshape(s, (s.shape[0], 1)) # Has shape (number_of_gxs, 1)
                minor = minor.reshape((d.shape[0],d.shape[1],3)) # Has shape (number_of_gxs, 1, 3)
                inter = inter.reshape((d.shape[0],d.shape[1],3)) # Has shape (number_of_gxs, 1, 3)
                major = major.reshape((d.shape[0],d.shape[1],3)) # Has shape (number_of_gxs, 1, 3)
                minor_re = minor.reshape(minor.shape[0], -1)
                inter_re = inter.reshape(inter.shape[0], -1)
                major_re = major.reshape(major.shape[0], -1)
            else:
                minor_re = np.array([])
                inter_re = np.array([])
                major_re = np.array([])
            
            # Writing
            with open('{0}/cat_global_gx_{1}.txt'.format(self.CAT_DEST, self.SNAP), 'w') as filehandle:
                json.dump(gx_cat, filehandle)
            np.savetxt('{0}/d_global_gx_{1}.txt'.format(self.CAT_DEST, self.SNAP), d, fmt='%1.7e')
            np.savetxt('{0}/q_global_gx_{1}.txt'.format(self.CAT_DEST, self.SNAP), q, fmt='%1.7e')
            np.savetxt('{0}/s_global_gx_{1}.txt'.format(self.CAT_DEST, self.SNAP), s, fmt='%1.7e')
            np.savetxt('{0}/minor_global_gx_{1}.txt'.format(self.CAT_DEST, self.SNAP), minor_re, fmt='%1.7e')
            np.savetxt('{0}/inter_global_gx_{1}.txt'.format(self.CAT_DEST, self.SNAP), inter_re, fmt='%1.7e')
            np.savetxt('{0}/major_global_gx_{1}.txt'.format(self.CAT_DEST, self.SNAP), major_re, fmt='%1.7e')
            np.savetxt('{0}/m_global_gx_{1}.txt'.format(self.CAT_DEST, self.SNAP), gx_m, fmt='%1.7e')
            np.savetxt('{0}/centers_global_gx_{1}.txt'.format(self.CAT_DEST, self.SNAP), gx_center, fmt='%1.7e')
        
            # Clean-up
            del d; del q; del s; del minor; del inter; del major; del gx_center; del gx_m; del star_xyz; del star_masses
    
    def calcLocalShapesDM(self):   
           
        print_status(rank,self.start_time,'Starting calcLocalShapesDM() with snap {0}'.format(self.SNAP))
        
        # Import hdf5 data
        print_status(rank,self.start_time,"Getting HDF5 raw data..")
        
        dm_xyz, dm_masses, dm_smoothing, dm_velxyz = getHDF5DMData(self.HDF5_SNAP_DEST, self.SNAP_MAX, self.SNAP)
        del dm_smoothing
        if rank != 0:
            del dm_xyz; del dm_masses; del dm_velxyz
        if rank == 0:
            with open('{0}/h_cat_{1}.txt'.format(self.CAT_DEST, self.SNAP), 'r') as filehandle:
                cat = json.load(filehandle)
            print_status(rank, self.start_time, "Got HDF5 raw data. Number of halos is {0}. The number of valid halos (sufficient-resolution ones) is {1}".format(len(cat), np.array([0 for x in cat if x != []]).shape[0]))
                
            # Morphology: Local Shape
            print_status(rank, self.start_time, "Calculating local-shape morphologies with {0} processors. The average number of ptcs in the Halos is {1}".format(len(os.sched_getaffinity(0)), np.average(np.array(list(map(lambda x: len([x for x in cat if x != []][x]), range(len([x for x in cat if x != []]))))))))
            d, q, s, minor, inter, major, halos_center, halo_m, succeeded = self.getMorphLocal(dm_xyz, cat, dm_masses, self.r200, self.L_BOX, self.MIN_NUMBER_PTCS, self.D_LOGSTART, self.D_LOGEND, self.D_BINS, self.M_TOL, self.N_WALL, self.N_MIN, self.CENTER)
            print_status(rank, self.start_time, "Finished morphologies")
        
            if succeeded != []:
                minor_re = minor.reshape(minor.shape[0], -1)
                inter_re = inter.reshape(inter.shape[0], -1)
                major_re = major.reshape(major.shape[0], -1)
            else:
                minor_re = np.array([])
                inter_re = np.array([])
                major_re = np.array([])
            
            # Writing
            cat_local = [[] for i in range(len(cat))] # We are removing those halos whose R200 shell does not converge (including where R200 is not even available)
            for success in succeeded:
                if self.r200[success] != 0.0: 
                    cat_local[success] = cat[success]
            with open('{0}/cat_local_dm_{1}.txt'.format(self.CAT_DEST, self.SNAP), 'w') as filehandle:
                json.dump(cat_local, filehandle)
            np.savetxt('{0}/d_local_dm_{1}.txt'.format(self.CAT_DEST, self.SNAP), d, fmt='%1.7e')
            np.savetxt('{0}/q_local_dm_{1}.txt'.format(self.CAT_DEST, self.SNAP), q, fmt='%1.7e')
            np.savetxt('{0}/s_local_dm_{1}.txt'.format(self.CAT_DEST, self.SNAP), s, fmt='%1.7e')
            np.savetxt('{0}/minor_local_dm_{1}.txt'.format(self.CAT_DEST, self.SNAP), minor_re, fmt='%1.7e')
            np.savetxt('{0}/inter_local_dm_{1}.txt'.format(self.CAT_DEST, self.SNAP), inter_re, fmt='%1.7e')
            np.savetxt('{0}/major_local_dm_{1}.txt'.format(self.CAT_DEST, self.SNAP), major_re, fmt='%1.7e')
            np.savetxt('{0}/m_local_dm_{1}.txt'.format(self.CAT_DEST, self.SNAP), halo_m, fmt='%1.7e')
            np.savetxt('{0}/centers_local_dm_{1}.txt'.format(self.CAT_DEST, self.SNAP), halos_center, fmt='%1.7e')
            
            # Clean-up
            del d; del q; del s; del minor; del inter; del major; del halos_center; del halo_m; del succeeded
            del dm_xyz; del dm_masses; del dm_velxyz
            
    def calcGlobalShapesDM(self):
        
        print_status(rank,self.start_time,'Starting calcGlobalShapesDM() with snap {0}'.format(self.SNAP))
        
        # Import hdf5 data
        print_status(rank,self.start_time,"Getting HDF5 raw data..")
        
        dm_xyz, dm_masses, dm_smoothing, dm_velxyz = getHDF5DMData(self.HDF5_SNAP_DEST, self.SNAP_MAX, self.SNAP)
        del dm_smoothing
        if rank != 0:
            del dm_xyz; del dm_masses; del dm_velxyz
        if rank == 0:
            with open('{0}/h_cat_{1}.txt'.format(self.CAT_DEST, self.SNAP), 'r') as filehandle:
                cat = json.load(filehandle)
            print_status(rank, self.start_time, "Got HDF5 raw data. Number of halos is {0}. The number of valid halos (sufficient-resolution ones) is {1}".format(len(cat), np.array([0 for x in cat if x != []]).shape[0]))
                
            # Morphology: Global Shape (with E1 at large radius)
            print_status(rank, self.start_time, "Calculating global morphologies with {0} processors. The average number of ptcs in the Halos is {1}".format(len(os.sched_getaffinity(0)), np.average(np.array(list(map(lambda x: len([x for x in cat if x != []][x]), range(len([x for x in cat if x != []]))))))))
            d, q, s, minor, inter, major, halos_center, halo_m = self.getMorphGlobal(dm_xyz, cat, dm_masses, self.r200, self.L_BOX, self.MIN_NUMBER_PTCS, self.M_TOL, self.N_WALL, self.N_MIN, self.CENTER)
            print_status(rank, self.start_time, "Finished morphologies")
        
            if d.shape[0] != 0:
                d = np.reshape(d, (d.shape[0], 1)) # Has shape (number_of_halos, 1)
                q = np.reshape(q, (q.shape[0], 1)) # Has shape (number_of_halos, 1)
                s = np.reshape(s, (s.shape[0], 1)) # Has shape (number_of_halos, 1)
                minor = minor.reshape((d.shape[0],d.shape[1],3)) # Has shape (number_of_halos, 1, 3)
                inter = inter.reshape((d.shape[0],d.shape[1],3)) # Has shape (number_of_halos, 1, 3)
                major = major.reshape((d.shape[0],d.shape[1],3)) # Has shape (number_of_halos, 1, 3)
                minor_re = minor.reshape(minor.shape[0], -1)
                inter_re = inter.reshape(inter.shape[0], -1)
                major_re = major.reshape(major.shape[0], -1)
            else:
                minor_re = np.array([])
                inter_re = np.array([])
                major_re = np.array([])
            
            # Writing
            with open('{0}/cat_global_dm_{1}.txt'.format(self.CAT_DEST, self.SNAP), 'w') as filehandle:
                json.dump(cat, filehandle)
            np.savetxt('{0}/d_global_dm_{1}.txt'.format(self.CAT_DEST, self.SNAP), d, fmt='%1.7e')
            np.savetxt('{0}/q_global_dm_{1}.txt'.format(self.CAT_DEST, self.SNAP), q, fmt='%1.7e')
            np.savetxt('{0}/s_global_dm_{1}.txt'.format(self.CAT_DEST, self.SNAP), s, fmt='%1.7e')
            np.savetxt('{0}/minor_global_dm_{1}.txt'.format(self.CAT_DEST, self.SNAP), minor_re, fmt='%1.7e')
            np.savetxt('{0}/inter_global_dm_{1}.txt'.format(self.CAT_DEST, self.SNAP), inter_re, fmt='%1.7e')
            np.savetxt('{0}/major_global_dm_{1}.txt'.format(self.CAT_DEST, self.SNAP), major_re, fmt='%1.7e')
            np.savetxt('{0}/m_global_dm_{1}.txt'.format(self.CAT_DEST, self.SNAP), halo_m, fmt='%1.7e')
            np.savetxt('{0}/centers_global_dm_{1}.txt'.format(self.CAT_DEST, self.SNAP), halos_center, fmt='%1.7e')
            
            # Clean-up
            del d; del q; del s; del minor; del inter; del major; del halos_center; del halo_m
            del dm_xyz; del dm_masses; del dm_velxyz
            
    def calcLocalVelShapesDM(self):
           
        print_status(rank,self.start_time,'Starting calcLocalVelShapesDM() with snap {0}'.format(self.SNAP))
        
        # Import hdf5 data
        print_status(rank,self.start_time,"Getting HDF5 raw data..")
        
        dm_xyz, dm_masses, dm_smoothing, dm_velxyz = getHDF5DMData(self.HDF5_SNAP_DEST, self.SNAP_MAX, self.SNAP)
        del dm_smoothing
        if rank != 0:
            del dm_xyz; del dm_masses; del dm_velxyz
        if rank == 0:
            with open('{0}/h_cat_{1}.txt'.format(self.CAT_DEST, self.SNAP), 'r') as filehandle:
                cat = json.load(filehandle)
            print_status(rank, self.start_time, "Got HDF5 raw data. Number of halos is {0}. The number of valid halos (sufficient-resolution ones) is {1}".format(len(cat), np.array([0 for x in cat if x != []]).shape[0]))
            
            # Morphology: Local Shape
            print_status(rank, self.start_time, "Calculating local-shape morphologies with {0} processors. The average number of ptcs in the Halos is {1}".format(len(os.sched_getaffinity(0)), np.average(np.array(list(map(lambda x: len([x for x in cat if x != []][x]), range(len([x for x in cat if x != []]))))))))
            d, q, s, minor, inter, major, halos_center, halo_m, succeeded = self.getMorphLocalVelDisp(dm_xyz, dm_velxyz, cat, dm_masses, self.r200, self.L_BOX, self.MIN_NUMBER_PTCS, self.D_LOGSTART, self.D_LOGEND, self.D_BINS, self.M_TOL, self.N_WALL, self.N_MIN, self.CENTER)
            print_status(rank, self.start_time, "Finished morphologies")
        
            if succeeded != []:
                minor_re = minor.reshape(minor.shape[0], -1)
                inter_re = inter.reshape(inter.shape[0], -1)
                major_re = major.reshape(major.shape[0], -1)
            else:
                minor_re = np.array([])
                inter_re = np.array([])
                major_re = np.array([])
            
            # Writing
            cat_local = [[] for i in range(len(cat))] # We are removing those halos whose R200 shell does not converge (including where R200 is not even available)
            for success in succeeded:
                if self.r200[success] != 0.0: 
                    cat_local[success] = cat[success]
            with open('{0}/cat_local_vdm_{1}.txt'.format(self.CAT_DEST, self.SNAP), 'w') as filehandle:
                json.dump(cat_local, filehandle)
            np.savetxt('{0}/d_local_vdm_{1}.txt'.format(self.CAT_DEST, self.SNAP), d, fmt='%1.7e')
            np.savetxt('{0}/q_local_vdm_{1}.txt'.format(self.CAT_DEST, self.SNAP), q, fmt='%1.7e')
            np.savetxt('{0}/s_local_vdm_{1}.txt'.format(self.CAT_DEST, self.SNAP), s, fmt='%1.7e')
            np.savetxt('{0}/minor_local_vdm_{1}.txt'.format(self.CAT_DEST, self.SNAP), minor_re, fmt='%1.7e')
            np.savetxt('{0}/inter_local_vdm_{1}.txt'.format(self.CAT_DEST, self.SNAP), inter_re, fmt='%1.7e')
            np.savetxt('{0}/major_local_vdm_{1}.txt'.format(self.CAT_DEST, self.SNAP), major_re, fmt='%1.7e')
            np.savetxt('{0}/m_local_vdm_{1}.txt'.format(self.CAT_DEST, self.SNAP), halo_m, fmt='%1.7e')
            np.savetxt('{0}/centers_local_vdm_{1}.txt'.format(self.CAT_DEST, self.SNAP), halos_center, fmt='%1.7e')
            
            # Clean-up
            del d; del q; del s; del minor; del inter; del major; del halos_center; del halo_m; del succeeded
            del dm_xyz; del dm_masses; del dm_velxyz
    
    def calcGlobalVelShapesDM(self):
        
        print_status(rank,self.start_time,'Starting calcGlobalVelShapesDM() with snap {0}'.format(self.SNAP))
        
        # Import hdf5 data
        print_status(rank,self.start_time,"Getting HDF5 raw data..")
        
        dm_xyz, dm_masses, dm_smoothing, dm_velxyz = getHDF5DMData(self.HDF5_SNAP_DEST, self.SNAP_MAX, self.SNAP)
        del dm_smoothing
        if rank != 0:
            del dm_xyz; del dm_masses; del dm_velxyz
        if rank == 0:
            with open('{0}/h_cat_{1}.txt'.format(self.CAT_DEST, self.SNAP), 'r') as filehandle:
                cat = json.load(filehandle)
            print_status(rank, self.start_time, "Got HDF5 raw data. Number of halos is {0}. The number of valid halos (sufficient-resolution ones) is {1}".format(len(cat), np.array([0 for x in cat if x != []]).shape[0]))
                
            # Velocity dispersion morphologies
            # Morphology: Global Shape (with E1 at R200)
            print_status(rank, self.start_time, "Calculating vdisp morphologies with {0} processors. The average number of ptcs in the Halos is {1}".format(len(os.sched_getaffinity(0)), np.average(np.array(list(map(lambda x: len([x for x in cat if x != []][x]), range(len([x for x in cat if x != []]))))))))
            d, q, s, minor, inter, major, halos_center, halo_m = self.getMorphGlobalVelDisp(dm_xyz, dm_velxyz, cat, dm_masses, self.r200, self.L_BOX, self.MIN_NUMBER_PTCS, self.M_TOL, self.N_WALL, self.N_MIN, self.CENTER)
            print_status(rank, self.start_time, "Finished morphologies")
        
            if d.shape[0] != 0:
                d = np.reshape(d, (d.shape[0], 1)) # Has shape (number_of_halos, 1)
                q = np.reshape(q, (q.shape[0], 1)) # Has shape (number_of_halos, 1)
                s = np.reshape(s, (s.shape[0], 1)) # Has shape (number_of_halos, 1)
                minor = minor.reshape((d.shape[0],d.shape[1],3)) # Has shape (number_of_halos, 1, 3)
                inter = inter.reshape((d.shape[0],d.shape[1],3)) # Has shape (number_of_halos, 1, 3)
                major = major.reshape((d.shape[0],d.shape[1],3)) # Has shape (number_of_halos, 1, 3)
                minor_re = minor.reshape(minor.shape[0], -1)
                inter_re = inter.reshape(inter.shape[0], -1)
                major_re = major.reshape(major.shape[0], -1)
            else:
                minor_re = np.array([])
                inter_re = np.array([])
                major_re = np.array([])
            
            # Writing
            with open('{0}/cat_global_vdm_{1}.txt'.format(self.CAT_DEST, self.SNAP), 'w') as filehandle:
                json.dump(cat, filehandle)
            np.savetxt('{0}/d_global_vdm_{1}.txt'.format(self.CAT_DEST, self.SNAP), d, fmt='%1.7e')
            np.savetxt('{0}/q_global_vdm_{1}.txt'.format(self.CAT_DEST, self.SNAP), q, fmt='%1.7e')
            np.savetxt('{0}/s_global_vdm_{1}.txt'.format(self.CAT_DEST, self.SNAP), s, fmt='%1.7e')
            np.savetxt('{0}/minor_global_vdm_{1}.txt'.format(self.CAT_DEST, self.SNAP), minor_re, fmt='%1.7e')
            np.savetxt('{0}/inter_global_vdm_{1}.txt'.format(self.CAT_DEST, self.SNAP), inter_re, fmt='%1.7e')
            np.savetxt('{0}/major_global_vdm_{1}.txt'.format(self.CAT_DEST, self.SNAP), major_re, fmt='%1.7e')
            np.savetxt('{0}/m_global_vdm_{1}.txt'.format(self.CAT_DEST, self.SNAP), halo_m, fmt='%1.7e')
            np.savetxt('{0}/centers_global_vdm_{1}.txt'.format(self.CAT_DEST, self.SNAP), halos_center, fmt='%1.7e')
            
            # Clean-up
            del d; del q; del s; del minor; del inter; del major; del halos_center; del halo_m
            del dm_xyz; del dm_masses; del dm_velxyz
            
    def calcDensProfsDirectBinning(self, ROverR200, obj_type = ''):
        
        if obj_type == 'dm':
            with open('{0}/h_cat_{1}.txt'.format(self.CAT_DEST, self.SNAP), 'r') as filehandle:
                cat = json.load(filehandle)
            xyz, masses, smoothing, velxyz = getHDF5DMData(self.HDF5_SNAP_DEST, self.SNAP_MAX, self.SNAP)
            del smoothing; del velxyz
        elif obj_type == 'gx':
            with open('{0}/gx_cat_{1}.txt'.format(self.CAT_DEST, self.SNAP), 'r') as filehandle:
                cat = json.load(filehandle)
            xyz, fof_com, sh_com, nb_shs, masses, smoothing, is_star = getHDF5GxData(self.HDF5_SNAP_DEST, self.HDF5_GROUP_DEST, self.SNAP_MAX, self.SNAP)
            del smoothing; del fof_com; del sh_com; del nb_shs; del is_star
        else:
            raise ValueError("For a GadgetHDF5 simulation, 'obj_type' must be either 'dm' or 'gx'")
        obj_keep = np.int32([1 if x != [] else 0 for x in cat])
        ROverR200, dens_profs = self.getDensProfsDirectBinning(cat, xyz, obj_keep, masses, self.r200, np.float32(ROverR200), self.L_BOX, self.CENTER)
        return dens_profs
    
    def calcDensProfsKernelBased(self, ROverR200, obj_type = ''):
        
        if obj_type == 'dm':
            with open('{0}/h_cat_{1}.txt'.format(self.CAT_DEST, self.SNAP), 'r') as filehandle:
                cat = json.load(filehandle)
            xyz, masses, smoothing, velxyz = getHDF5DMData(self.HDF5_SNAP_DEST, self.SNAP_MAX, self.SNAP)
            del smoothing; del velxyz
        elif obj_type == 'gx':
            with open('{0}/gx_cat_{1}.txt'.format(self.CAT_DEST, self.SNAP), 'r') as filehandle:
                cat = json.load(filehandle)
            xyz, fof_com, sh_com, nb_shs, masses, smoothing, is_star = getHDF5GxData(self.HDF5_SNAP_DEST, self.HDF5_GROUP_DEST, self.SNAP_MAX, self.SNAP)
            del smoothing; del fof_com; del sh_com; del nb_shs; del is_star
        else:
            raise ValueError("For a GadgetHDF5 simulation, 'obj_type' must be either 'dm' or 'gx'")
        obj_keep = np.int32([1 if x != [] else 0 for x in cat])
        ROverR200, dens_profs = self.getDensProfsKernelBased(cat, xyz, obj_keep, masses, self.r200, np.float32(ROverR200), self.L_BOX, self.CENTER)
        return dens_profs
    
    def plotGlobalEpsHisto(self, obj_type = ''):
        if obj_type == 'dm':
            suffix = '_dm_'
            xyz, masses, smoothing, velxyz = getHDF5DMData(self.HDF5_SNAP_DEST, self.SNAP_MAX, self.SNAP)
            del smoothing; del velxyz
            with open('{0}/cat_global_dm_{1}.txt'.format(self.CAT_DEST, self.SNAP), 'r') as filehandle:
                cat = json.load(filehandle)
        elif obj_type == 'gx':
            suffix = '_gx_'
            with open('{0}/cat_global_gx_{1}.txt'.format(self.CAT_DEST, self.SNAP), 'r') as filehandle:
                cat = json.load(filehandle)
            xyz, fof_com, sh_com, nb_shs, masses, star_smoothing, is_star = getHDF5GxData(self.HDF5_SNAP_DEST, self.HDF5_GROUP_DEST, self.SNAP_MAX, self.SNAP)
            del fof_com; del sh_com; del nb_shs; del star_smoothing; del is_star
        else:
            raise ValueError("For a GadgetHDF5 simulation, 'obj_type' must be either 'dm' or 'gx'")
        
        getGlobalEpsHisto(cat, xyz, masses, self.L_BOX, self.VIZ_DEST, self.SNAP, suffix = suffix, HIST_NB_BINS = 11)