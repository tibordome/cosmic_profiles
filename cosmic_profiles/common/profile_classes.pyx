#cython: language_level=3
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 13:17:51 2021
"""

import numpy as np
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
from cosmic_profiles.common.python_routines import print_status, set_axes_equal, fibonacci_ellipsoid, respectPBCNoRef, getCoM, findMode
from cosmic_profiles.shape_profs.shape_profs_tools import getGlobalEpsHisto, getLocalTHisto, getShapeProfiles
from cosmic_profiles.dens_profs.dens_profs_tools import getDensityProfiles, fitDensProfHelper
from cosmic_profiles.gadget_hdf5.get_hdf5 import getHDF5Data, getHDF5GxData, getHDF5SHDMData, getHDF5SHGxData, getHDF5DMData
from cosmic_profiles.gadget_hdf5.gen_catalogues import getCSHCat, getGxCat
from cosmic_profiles.cython_helpers.helper_class cimport CythonHelpers
from cosmic_profiles.shape_profs.shape_profs_algos cimport runEllShellAlgo, runEllAlgo, runEllVDispAlgo
from cosmic_profiles.dens_profs.dens_profs_algos import getDensProfsDirectBinning, getDensProfsKernelBased
import time
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

@cython.embedsignature(True)
cdef class CosmicProfiles:
    """ Parent class governing low-level cosmic shape calculations
    
    Its public methods are ``fetchCatLocal()``, ``fetchCatGlobal()``, ``getDensProfsDirectBinning()``,
    ``getDensProfsKernelBased()``, ``runS1()``, ``runE1()``, ``runE1VelDisp()``, ``getObjMorphLocal()``, ``getObjMorphGlobal()``, 
    ``getObjMorphLocalVelDisp()``, ``getObjMorphGlobalVelDisp()``, ``getMorphLocal()``, ``getMorphGlobal()``, 
    ``getMorphLocalVelDisp()``, ``getMorphGlobalVelDisp()``, ``drawShapeProfiles()``, ``plotLocalTHisto()``, 
    ``fitDensProfs()``, ``fetchDensProfsBestFits()``, ``fetchDensProfsDirectBinning()``,
    ``fetchDensProfsKernelBased()`` and ``fetchShapeCat()``."""
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
    cdef float SAFE # Units: Mpc/h. Ellipsoidal radius will be maxdist(COM,point)+SAFE where point is any point in the point cloud. The larger the better.
    cdef double start_time
    
    def __init__(self, str CAT_DEST, str VIZ_DEST, str SNAP, float L_BOX, int MIN_NUMBER_PTCS, int D_LOGSTART, int D_LOGEND, int D_BINS, float M_TOL, int N_WALL, int N_MIN, str CENTER, double start_time):
        """        
        :param CAT_DEST: catalogue destination
        :type CAT_DEST: string
        :param VIZ_DEST: visualisation folder destination
        :type VIZ_DEST: string
        :param L_BOX: simulation box side length
        :type L_BOX: float, units: Mpc/h
        :param MIN_NUMBER_PTCS: minimum number of particles for object to qualify for morphology calculation
        :type MIN_NUMBER_PTCS: int
        :param D_LOGSTART: logarithm of minimum ellipsoidal radius of interest, in units of R200 of parent halo
        :type D_LOGSTART: int
        :param D_LOGEND: logarithm of maximum ellipsoidal radius of interest, in units of R200 of parent halo
        :type D_LOGEND: int
        :param D_BINS: number of ellipsoidal radii of interest minus 1 (i.e. number of bins)
        :type D_BINS: int
        :param M_TOL: convergence tolerance, eigenvalue fractions must differ by less than ``M_TOL``
            for iteration to stop
        :type M_TOL: float
        :param N_WALL: maximum permissible number of iterations
        :type N_WALL: float
        :param N_MIN: minimum number of particles (DM or star particle) in any iteration; 
            if undercut, shape is unclassified
        :type N_MIN: int
        :param CENTER: shape quantities will be calculated with respect to CENTER = 'mode' (point of highest density)
            or 'com' (center of mass) of each halo
        :type CENTER: str
        :param start_time: time of start of object initialization
        :type start_time: float"""
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
        
    def calcMassesCenters(self, cat, float[:,:] xyz, float[:] masses, int MIN_NUMBER_PTCS, float L_BOX, str CENTER):
        """ Calculate total mass and centers of objects
        
        :param cat: list of indices defining the objects
        :type cat: list of length N1, each consisting of a list of int indices
        :param xyz: positions of all simulation particles
        :type xyz: (N2,3) floats, N2 >> N1
        :param masses: masses of all simulation particles
        :type masses: (N2,) floats
        :param MIN_NUMBER_PTCS: minimum number of particles for object to qualify for morphology calculation
        :type MIN_NUMBER_PTCS: int
        :param L_BOX: box size
        :type L_BOX: float
        :param CENTER: density profiles will be calculated with respect to CENTER = 'mode' (point of highest density)
            or 'com' (center of mass) of each halo
        :type CENTER: str
        :return centers, m: centers and masses
        :rtype: (N,3) and (N,) floats"""
        # Transform cat to int[:,:]
        cdef int nb_objs = len(cat)
        cdef int p
        cdef int[:] obj_size = np.zeros((nb_objs,), dtype = np.int32)
        cdef int[:] obj_pass = np.zeros((nb_objs,), dtype = np.int32)
        for p in range(nb_objs):
            if len(cat[p]) >= MIN_NUMBER_PTCS: # Only add objects that have sufficient resolution
                obj_size[p] = len(cat[p]) 
                obj_pass[p] = 1
        cdef int nb_pass = np.sum(obj_pass.base)
        cdef int[:] idxs_compr = np.zeros((nb_objs,), dtype = np.int32)
        idxs_compr.base[obj_pass.base.nonzero()[0]] = np.arange(np.sum(obj_pass.base))
        cdef int[:,:] cat_arr = np.zeros((nb_pass,np.max([len(cat[p]) for p in range(nb_objs)])), dtype = np.int32)
        for p in range(nb_objs):
            if obj_pass[p] == 1:
                cat_arr.base[idxs_compr[p],:obj_size[p]] = np.array(cat[p])
    
        # Calculate centers and total masses of objects
        cdef float[:] m = np.zeros((nb_pass,), dtype = np.float32)
        cdef int n
        cdef float[:,:] centers = np.zeros((nb_pass,3), dtype = np.float32)
        for p in range(nb_objs): # Calculate centers of objects
            if obj_pass[p] == 1:
                xyz_ = respectPBCNoRef(xyz.base[cat_arr.base[idxs_compr[p],:obj_size[p]]], L_BOX)
                if CENTER == 'mode':
                    centers.base[idxs_compr[p]] = findMode(xyz_, masses.base[cat_arr.base[idxs_compr[p],:obj_size[p]]], 1000)
                else:
                    centers.base[idxs_compr[p]] = getCoM(xyz_, masses.base[cat_arr.base[idxs_compr[p],:obj_size[p]]])
        for p in prange(nb_objs, schedule = 'dynamic', nogil = True): # Calculate total mass of objects
            if obj_pass[p] == 1:
                for n in range(obj_size[p]):
                    m[idxs_compr[p]] = m[idxs_compr[p]] + masses[cat_arr[idxs_compr[p],n]]
        
        return centers.base, m.base # Only rank = 0 content matters
    
    def fetchMassesCenters(self, obj_type):
        """ Calculate total mass and centers of objects
        
        :param obj_type: either 'dm' or 'gx' for CosmicProfilesGadgetHDF5 or '' for CosmicProfilesDirect
        :type obj_type: string
        :return centers, m: centers and masses
        :rtype: (N,3) and (N,) floats"""
        if obj_type == 'dm':
            suffix = '_dm_'
        elif obj_type == 'gx':
            suffix = '_gx_'
        else:
            assert obj_type == ''
            suffix = '_'
        ms = np.loadtxt('{0}/m{1}{2}.txt'.format(self.CAT_DEST, suffix, self.SNAP))
        centers = np.loadtxt('{0}/centers{1}{2}.txt'.format(self.CAT_DEST, suffix, self.SNAP))
        return centers, ms
    
    def fetchCatLocal(self, obj_type = 'dm'):
        """ Fetch local halo/gx catalogue
        
        :param obj_type: either 'dm' or 'gx' for CosmicProfilesGadgetHDF5 or '' for CosmicProfilesDirect
        :type obj_type: string
        :return cat_local: list of indices defining the objects
        :type cat_local: list of length N1, each consisting of a list of int indices"""
        print_status(rank,self.start_time,'Starting fetchCatLocal() with snap {0}'.format(self.SNAP))
        
        if rank == 0:
            if obj_type == 'dm':
                suffix = '_dm_'
            elif obj_type == 'gx':
                suffix = '_gx_'
            else:
                assert obj_type == ''
                suffix = '_'
            with open('{0}/cat_local{1}{2}.txt'.format(self.CAT_DEST, suffix, self.SNAP), 'r') as filehandle:
                cat_local = json.load(filehandle)
            return cat_local
        else:
            return None
    
    def fetchCatGlobal(self, obj_type = 'dm'):
        """ Fetch global halo/gx catalogue
        
        :param obj_type: either 'dm' or 'gx' for CosmicProfilesGadgetHDF5 or '' for CosmicProfilesDirect
        :type obj_type: string
        :return cat_global: list of indices defining the objects
        :type cat_global: list of length N1, each consisting of a list of int indices"""
        print_status(rank,self.start_time,'Starting fetchCatGlobal() with snap {0}'.format(self.SNAP))
        
        if rank == 0:
            if obj_type == 'dm':
                suffix = '_dm_'
            elif obj_type == 'gx':
                suffix = '_gx_'
            else:
                assert obj_type == ''
                suffix = '_'
            with open('{0}/cat_global{1}{2}.txt'.format(self.CAT_DEST, suffix, self.SNAP), 'r') as filehandle:
                cat_global = json.load(filehandle)
            return cat_global
        else:
            return None
    
    cdef float[:,:] getObjMorphLocal(self, float[:,:] morph_info, float r200, float[:] log_d, float[:,:] xyz, float[:,:] xyz_princ, float[:] masses, int[:] shell, float[:] center, complex[::1,:] shape_tensor, double[::1] eigval, complex[::1,:] eigvec, float M_TOL, int N_WALL, int N_MIN) nogil:
        """ Calculates the local axis ratios
        
        The local morphology is calculated for the ellipsoidal radius range [ ``r200`` x ``log_d`` [0], ``r200`` x ``log_d`` [-1]] 
        from the center of the point cloud
        
        :param morph_info: Array to be filled with morphological info. For each column, 1st entry: d,
            2nd entry: q, 3rd entry: s, 4th to 6th: normalized major axis, 7th to 9th: normalized intermediate axis,
            10th to 12th: normalized minor axis
        :type morph_info: (12,N) floats
        :param r200: R_200 (mean not critical) radius of the parent halo
        :type r200: (N2,) float array
        :param log_d: logarithmically equally spaced ellipsoidal radius array of interest, in units of R_200 
            radius of the parent halo, e.g. np.logspace(-2,1,100)
        :type log_d: (N3,) floats
        :param xyz: positions of particles in point cloud
        :type xyz: (N1 x 3) floats
        :param xyz_princ: position arrays transformed into principal frame (varies from iteration to iteration)
        :type xyz_princ: (N1 x 3) floats, zeros
        :param masses: masses of the particles expressed in unit mass
        :type masses: (N1 x 1) floats
        :param shell: indices of points that fall into shell (varies from iteration to iteration)
        :type shell: (N,) ints, zeros
        :param center: center of point cloud
        :type center: (3,) floats
        :param shape_tensor: shape tensor array to be filled
        :type shape_tensor: (3,3) complex, zeros
        :param eigval: eigenvalue array to be filled
        :type eigval: (3,) double, zeros
        :param eigvec: eigenvector array to be filled
        :type eigvec: (3,3) double, zeros
        :param M_TOL: convergence tolerance, eigenvalue fractions must differ by less than ``M_TOL``
            for iteration to stop
        :type M_TOL: float
        :param N_WALL: maximum permissible number of iterations
        :type N_WALL: float
        :param N_MIN: minimum number of particles (DM or star particle) in any iteration; 
            if undercut, shape is unclassified
        :type N_MIN: int
        :param CENTER: shape quantities will be calculated with respect to CENTER = 'mode' (point of highest density)
            or 'com' (center of mass) of each halo
        :type CENTER: str
        :return: ``morph_info`` containing d, q, s, eigframe info in each column, for each ellipsoidal radius
        :rtype: (12,N) float array"""
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
            morph_info[:,i] = runEllAlgo(morph_info[:,i], xyz, xyz_princ, masses, shell, center, shape_tensor, eigval, eigvec, morph_info[0,i], M_TOL, N_WALL, N_MIN)
        
        # Discard if r200 ellipsoid did not converge
        closest_idx = 0
        for i in range(nb_shells):
            if (r200*log_d[i] - r200)**2 < (r200*log_d[closest_idx] - r200)**2:
                closest_idx = i
        if morph_info[1,closest_idx] == 0: # Return empty morph_info if R200 ellipsoid did not converge
            morph_info[:,:] = 0.0
        return morph_info
    
    cdef float[:] getObjMorphGlobal(self, float[:] morph_info, float r200, float[:,:] xyz, float[:,:] xyz_princ, float[:] masses, int[:] ellipsoid, float[:] center, complex[::1,:] shape_tensor, double[::1] eigval, complex[::1,:] eigvec, float M_TOL, int N_WALL, int N_MIN) nogil:
        """ Calculates the global axis ratios and eigenframe of the point cloud
        
        :param morph_info: Array to be filled with morphological info. 1st entry: d,
            2nd entry: q, 3rd entry: s, 4th to 6th: normalized major axis, 7th to 9th: normalized intermediate axis,
            10th to 12th: normalized minor axis
        :type morph_info: (12,) floats
        :param r200: R_200 (mean not critical) radius of the parent halo
        :type r200: (N2,) float array
        :param xyz: positions of particles in point cloud
        :type xyz: (N1 x 3) floats
        :param xyz_princ: position arrays transformed into principal frame (varies from iteration to iteration)
        :type xyz_princ: (N1 x 3) floats, zeros
        :param masses: masses of the particles expressed in unit mass
        :type masses: (N1 x 1) floats
        :param ellipsoid: indices of points that fall into ellipsoid (varies from iteration to iteration)
        :type ellipsoid: (N,) ints, zeros
        :param center: center of point cloud
        :type center: (3,) floats
        :param shape_tensor: shape tensor array to be filled
        :type shape_tensor: (3,3) complex, zeros
        :param eigval: eigenvalue array to be filled
        :type eigval: (3,) double, zeros
        :param eigvec: eigenvector array to be filled
        :type eigvec: (3,3) double, zeros
        :param M_TOL: convergence tolerance, eigenvalue fractions must differ by less than ``M_TOL``
            for iteration to stop
        :type M_TOL: float
        :param N_WALL: maximum permissible number of iterations
        :type N_WALL: float
        :param N_MIN: minimum number of particles (DM or star particle) in any iteration; 
            if undercut, shape is unclassified
        :type N_MIN: int
        :param CENTER: shape quantities will be calculated with respect to CENTER = 'mode' (point of highest density)
            or 'com' (center of mass) of each halo
        :type CENTER: str
        :return: ``morph_info`` containing d, q, s, eigframe info
        :rtype: (12,) float array"""
        # Return if problematic
        morph_info[:] = 0.0
        if CythonHelpers.getLocalSpread(xyz) == 0.0: # Too low resolution = no points in this object
            morph_info[:] = 0.0
            return morph_info
        morph_info[0] = r200+self.SAFE
        
        # Retrieve morphology
        morph_info[:] = runEllAlgo(morph_info[:], xyz, xyz_princ, masses, ellipsoid, center, shape_tensor, eigval, eigvec, morph_info[0], M_TOL, N_WALL, N_MIN)
        return morph_info
    
    cdef float[:,:] getObjMorphLocalVelDisp(self, float[:,:] morph_info, float r200, float[:] log_d, float[:,:] xyz, float[:,:] vxyz, float[:,:] xyz_princ, float[:] masses, int[:] shell, float[:] center, float[:] vcenter, complex[::1,:] shape_tensor, double[::1] eigval, complex[::1,:] eigvec, float M_TOL, int N_WALL, int N_MIN) nogil:
        """ Calculates the local axis ratios of the velocity dispersion tensor 
        
        The local morphology is calculated for the ellipsoidal radius range [ ``r200`` x ``log_d`` [0], ``r200`` x ``log_d`` [-1]] 
        from the center of the point cloud
        
        :param morph_info: Array to be filled with morphological info. For each column, 1st entry: d,
            2nd entry: q, 3rd entry: s, 4th to 6th: normalized major axis, 7th to 9th: normalized intermediate axis,
            10th to 12th: normalized minor axis
        :type morph_info: (12,N) floats
        :param r200: R_200 (mean not critical) radius of the parent halo
        :type r200: (N2,) float array
        :param log_d: logarithmically equally spaced ellipsoidal radius array of interest, in units of R_200 
            radius of the parent halo, e.g. np.logspace(-2,1,100)
        :type log_d: (N3,) floats
        :param xyz: positions of particles in point cloud
        :type xyz: (N1 x 3) floats
        :param vxyz: velocity array
        :type vxyz: (N x 3) floats
        :param xyz_princ: position arrays transformed into principal frame (varies from iteration to iteration)
        :type xyz_princ: (N1 x 3) floats, zeros
        :param masses: masses of the particles expressed in unit mass
        :type masses: (N1 x 1) floats
        :param shell: indices of points that fall into shell (varies from iteration to iteration)
        :type shell: (N,) ints, zeros
        :param center: center of point cloud
        :type center: (3,) floats
        :param vcenter: velocity-center of point cloud
        :type vcenter: (3,) floats
        :param shape_tensor: shape tensor array to be filled
        :type shape_tensor: (3,3) complex, zeros
        :param eigval: eigenvalue array to be filled
        :type eigval: (3,) double, zeros
        :param eigvec: eigenvector array to be filled
        :type eigvec: (3,3) double, zeros
        :param M_TOL: convergence tolerance, eigenvalue fractions must differ by less than ``M_TOL``
            for iteration to stop
        :type M_TOL: float
        :param N_WALL: maximum permissible number of iterations
        :type N_WALL: float
        :param N_MIN: minimum number of particles (DM or star particle) in any iteration; 
            if undercut, shape is unclassified
        :type N_MIN: int
        :param CENTER: shape quantities will be calculated with respect to CENTER = 'mode' (point of highest density)
            or 'com' (center of mass) of each halo
        :type CENTER: str
        :return: ``morph_info`` containing d (= ``r200``), q, s, eigframe info
        :rtype: (12,) float array"""
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
            morph_info[:,i] = runEllVDispAlgo(morph_info[:,i], xyz, vxyz, xyz_princ, masses, shell, center, vcenter, shape_tensor, eigval, eigvec, morph_info[0,i], M_TOL, N_WALL, N_MIN)
        
        # Discard if r200 ellipsoid did not converge
        closest_idx = 0
        for i in range(nb_shells):
            if (r200*log_d[i] - r200)**2 < (r200*log_d[closest_idx] - r200)**2:
                closest_idx = i
        if morph_info[1,closest_idx] == 0: # Return empty morph_info if R200 ellipsoid did not converge
            morph_info[:,:] = 0.0
        return morph_info
    
    cdef float[:] getObjMorphGlobalVelDisp(self, float[:] morph_info, float r200, float[:,:] xyz, float[:,:] vxyz, float[:,:] xyz_princ, float[:] masses, int[:] ellipsoid, float[:] center, float[:] vcenter, complex[::1,:] shape_tensor, double[::1] eigval, complex[::1,:] eigvec, float M_TOL, int N_WALL, int N_MIN) nogil:
        """ Calculates the global axis ratios and eigenframe of the velocity dispersion tensor
        
        :param morph_info: Array to be filled with morphological info. 1st entry: d,
            2nd entry: q, 3rd entry: s, 4th to 6th: normalized major axis, 7th to 9th: normalized intermediate axis,
            10th to 12th: normalized minor axis
        :type morph_info: (12,) floats
        :param r200: R_200 (mean not critical) radius of the parent halo
        :type r200: (N2,) float array
        :param xyz: positions of particles in point cloud
        :type xyz: (N1 x 3) floats
        :param vxyz: velocity array
        :type vxyz: (N x 3) floats
        :param xyz_princ: position arrays transformed into principal frame (varies from iteration to iteration)
        :type xyz_princ: (N1 x 3) floats, zeros
        :param masses: masses of the particles expressed in unit mass
        :type masses: (N1 x 1) floats
        :param ellipsoid: indices of points that fall into ellipsoid (varies from iteration to iteration)
        :type ellipsoid: (N,) ints, zeros
        :param center: center of point cloud
        :type center: (3,) floats
        :param vcenter: velocity-center of point cloud
        :type vcenter: (3,) floats
        :param shape_tensor: shape tensor array to be filled
        :type shape_tensor: (3,3) complex, zeros
        :param eigval: eigenvalue array to be filled
        :type eigval: (3,) double, zeros
        :param eigvec: eigenvector array to be filled
        :type eigvec: (3,3) double, zeros
        :param d: distance from the center, kept fixed during iterative procedure
        :type d: float
        :param delta_d: thickness of the shell in real space (constant across shells in logarithmic space)
        :type delta_d: float
        :param M_TOL: convergence tolerance, eigenvalue fractions must differ by less than ``M_TOL``
            for iteration to stop
        :type M_TOL: float
        :param N_WALL: maximum permissible number of iterations
        :type N_WALL: float
        :param N_MIN: minimum number of particles (DM or star particle) in any iteration; 
            if undercut, shape is unclassified
        :type N_MIN: int
        :param CENTER: shape quantities will be calculated with respect to CENTER = 'mode' (point of highest density)
            or 'com' (center of mass) of each halo
        :type CENTER: str
        :return: ``morph_info`` containing d (= ``r200``), q, s, eigframe info
        :rtype: (12,) float array"""
        # Return if problematic
        morph_info[:] = 0.0
        if CythonHelpers.getLocalSpread(xyz) == 0.0: # Too low resolution = no points in this object
            morph_info[:] = 0.0
            return morph_info
        morph_info[0] = r200+self.SAFE
        
        # Retrieve morphology
        morph_info[:] = runEllVDispAlgo(morph_info[:], xyz, vxyz, xyz_princ, masses, ellipsoid, center, vcenter, shape_tensor, eigval, eigvec, morph_info[0], M_TOL, N_WALL, N_MIN)
        return morph_info
    
    def getMorphLocal(self, float[:,:] xyz, cat, float[:] masses, float[:] r200, float L_BOX, int MIN_NUMBER_PTCS, int D_LOGSTART, int D_LOGEND, int D_BINS, int M_TOL, int N_WALL, int N_MIN, str CENTER):
        """ Calculates the local shape catalogue
        
        Calls ``getObjMorphLocal()`` in a parallelized manner.\n
        Calculates the axis ratios for the range [ ``r200`` x 10**(``D_LOGSTART``), ``r200`` x 10**(``D_LOGEND``)] from the centers, for each object.
        
        :param xyz: positions of all (DM or star) particles in simulation box
        :type xyz: (N1 x 3) floats
        :param cat: each entry of the list is a list containing indices of particles belonging to an object
        :type cat: list of length N2
        :param masses: masses of the particles expressed in unit mass
        :type masses: (N1 x 1) floats
        :param r200: each entry of the list gives the R_200 (mean not critical) radius of the parent halo
        :type r200: list of length N2
        :param L_BOX: simulation box side length
        :type L_BOX: float, units: Mpc/h
        :param MIN_NUMBER_PTCS: minimum number of particles for object to qualify for morphology calculation
        :type MIN_NUMBER_PTCS: int
        :param D_LOGSTART: logarithm of minimum ellipsoidal radius of interest, in units of R200 of parent halo
        :type D_LOGSTART: int
        :param D_LOGEND: logarithm of maximum ellipsoidal radius of interest, in units of R200 of parent halo
        :type D_LOGEND: int
        :param D_BINS: number of ellipsoidal radii of interest minus 1 (i.e. number of bins)
        :type D_BINS: int
        :param M_TOL: convergence tolerance, eigenvalue fractions must differ by less than ``M_TOL``
            for iteration to stop
        :type M_TOL: float
        :param N_WALL: maximum permissible number of iterations
        :type N_WALL: float
        :param N_MIN: minimum number of particles (DM or star particle) in any iteration; 
            if undercut, shape is unclassified
        :type N_MIN: int
        :param CENTER: shape quantities will be calculated with respect to CENTER = 'mode' (point of highest density)
            or 'com' (center of mass) of each halo
        :type CENTER: str
        :return: d, q, s, eigframe, centers, masses, l_succeed: list of object indices for which morphology could be determined at R200 (length: N3)
        :rtype: (N3, ``D_BINS`` + 1) floats (for d, q, s, eigframe (x3)), (N3, 3) floats (for centers), (N3,) floats (for masses), N3-list of ints for l_succeed
        """
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
        """ Calculates the overall shape catalogue
        
        Calls ``getObjMorphGlobal()`` in a parallelized manner.\n
        Calculates the overall axis ratios and eigenframe for each object.
        
        :param xyz: positions of all (DM or star) particles in simulation box
        :type xyz: (N1 x 3) floats
        :param cat: each entry of the list is a list containing indices of particles belonging to an object
        :type cat: list of length N2
        :param masses: masses of the particles expressed in unit mass
        :type masses: (N1 x 1) floats
        :param r200: each entry of the list gives the R_200 (mean not critical) radius of the parent halo
        :type r200: list of length N2
        :param L_BOX: simulation box side length
        :type L_BOX: float, units: Mpc/h
        :param MIN_NUMBER_PTCS: minimum number of particles for object to qualify for morphology calculation
        :type MIN_NUMBER_PTCS: int
        :param M_TOL: convergence tolerance, eigenvalue fractions must differ by less than ``M_TOL``
            for iteration to stop
        :type M_TOL: float
        :param N_WALL: maximum permissible number of iterations
        :type N_WALL: float
        :param N_MIN: minimum number of particles (DM or star particle) in any iteration; 
            if undercut, shape is unclassified
        :type N_MIN: int
        :param CENTER: shape quantities will be calculated with respect to CENTER = 'mode' (point of highest density)
            or 'com' (center of mass) of each halo
        :type CENTER: str
        :return: d, q, s, eigframe, centers, masses
        :rtype: (N3,) floats (for d, q, s, eigframe (x3)), (N3, 3) floats (for centers), (N3,) floats (for masses)
        """
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
        """ Calculates the local velocity dispersion shape catalogue
        
        Calls ``getObjMorphLocalVelDisp()`` in a parallelized manner.\n
        Calculates the overall axis ratios and eigenframe for each object.
        
        :param xyz: positions of all (DM or star) particles in simulation box
        :type xyz: (N1 x 3) floats
        :param vxyz: velocities of all (DM or star) particles in simulation box
        :type vxyz: (N1 x 3) floats
        :param cat: each entry of the list is a list containing indices of particles belonging to an object
        :type cat: list of length N2
        :param masses: masses of the particles expressed in unit mass
        :type masses: (N1 x 1) floats
        :param r200: each entry of the list gives the R_200 (mean not critical) radius of the parent halo
        :type r200: list of length N2
        :param L_BOX: simulation box side length
        :type L_BOX: float, units: Mpc/h
        :param MIN_NUMBER_PTCS: minimum number of particles for object to qualify for morphology calculation
        :type MIN_NUMBER_PTCS: int
        :param D_LOGSTART: logarithm of minimum ellipsoidal radius of interest, in units of R200 of parent halo
        :type D_LOGSTART: int
        :param D_LOGEND: logarithm of maximum ellipsoidal radius of interest, in units of R200 of parent halo
        :type D_LOGEND: int
        :param D_BINS: number of ellipsoidal radii of interest mi
        :param M_TOL: convergence tolerance, eigenvalue fractions must differ by less than ``M_TOL``
            for iteration to stop
        :type M_TOL: float
        :param N_WALL: maximum permissible number of iterations
        :type N_WALL: float
        :param N_MIN: minimum number of particles (DM or star particle) in any iteration; 
            if undercut, shape is unclassified
        :type N_MIN: int
        :param CENTER: shape quantities will be calculated with respect to CENTER = 'mode' (point of highest density)
            or 'com' (center of mass) of each halo
        :type CENTER: str
        :return: d, q, s, eigframe, centers, masses, l_succeed: list of object indices for which morphology could be determined at R200 (length: N3)
        :rtype: (N3, ``D_BINS`` + 1) floats (for d, q, s, eigframe (x3)), (N3, 3) floats (for centers), (N3,) floats (for masses), N3-list of ints for l_succeed
        """
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
        """ Calculates the global velocity dipsersion shape catalogue
        
        Calls ``getObjMorphGlobalVelDisp()`` in a parallelized manner.\n
        Calculates the overall axis ratios and eigenframe for each object.
        
        :param xyz: positions of all (DM or star) particles in simulation box
        :type xyz: (N1 x 3) floats
        :param vxyz: velocities of all (DM or star) particles in simulation box
        :type vxyz: (N1 x 3) floats
        :param cat: each entry of the list is a list containing indices of particles belonging to an object
        :type cat: list of length N2
        :param masses: masses of the particles expressed in unit mass
        :type masses: (N1 x 1) floats
        :param r200: R_200 (mean not critical) radii of the parent halos
        :type r200: (N2,) floats
        :param L_BOX: simulation box side length
        :type L_BOX: float, units: Mpc/h
        :param MIN_NUMBER_PTCS: minimum number of particles for object to qualify for morphology calculation
        :type MIN_NUMBER_PTCS: int
        :param M_TOL: convergence tolerance, eigenvalue fractions must differ by less than ``M_TOL``
            for iteration to stop
        :type M_TOL: float
        :param N_WALL: maximum permissible number of iterations
        :type N_WALL: float
        :param N_MIN: minimum number of particles (DM or star particle) in any iteration; 
            if undercut, shape is unclassified
        :type N_MIN: int
        :param CENTER: shape quantities will be calculated with respect to CENTER = 'mode' (point of highest density)
            or 'com' (center of mass) of each halo
        :type CENTER: str
        :return: d, q, s, eigframe, centers, masses
        :rtype: (N3, ``D_BINS`` + 1) floats (for d, q, s, eigframe (x3)), (N3, 3) floats (for centers), (N3,) floats (for masses)
        """
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
        """ Draws some simplistic shape profiles
        
        :param obj_type: either 'dm' or 'gx' for CosmicProfilesGadgetHDF5 or '' for CosmicProfilesDirect
        :type obj_type: string"""
        print_status(rank,self.start_time,'Starting drawShapeProfiles() with snap {0}'.format(self.SNAP))
        
        if rank == 0:
            if obj_type == 'dm':
                suffix = '_dm_'
            elif obj_type == 'gx':
                suffix = '_gx_'
            else:
                assert obj_type == ''
                suffix = '_'
            obj_masses, obj_centers, d, q, s, major_full = self.fetchShapeCat(True, obj_type)
            getShapeProfiles(self.VIZ_DEST, self.SNAP, self.D_LOGSTART, self.D_LOGEND, self.D_BINS, self.start_time, obj_masses, obj_centers, d, q, s, major_full, MASS_UNIT=1e10, suffix = suffix)
            
    def plotLocalTHisto(self, obj_type = ''):
        """ Plot the triaxiality-histogram
        
        :param obj_type: either 'dm' or 'gx' for CosmicProfilesGadgetHDF5 or '' for CosmicProfilesDirect
        :type obj_type: string"""
        print_status(rank,self.start_time,'Starting plotLocalTHisto() with snap {0}'.format(self.SNAP))
        
        if rank == 0:
            if obj_type == 'dm':
                suffix = '_dm_'
            elif obj_type == 'gx':
                suffix = '_gx_'
            else:
                assert obj_type == ''
                suffix = '_'
            obj_masses, obj_centers, d, q, s, major_full = self.fetchShapeCat(True, obj_type)
            getLocalTHisto(self.CAT_DEST, self.VIZ_DEST, self.SNAP, self.D_LOGSTART, self.D_LOGEND, self.D_BINS, self.start_time, obj_masses, obj_centers, d, q, s, major_full, HIST_NB_BINS=11, MASS_UNIT=1e10, suffix = suffix, inner = False)
        
    def fitDensProfs(self, dens_profs, ROverR200, cat, r200s, method = 'einasto'):
        """ Fit the density profiles ``dens_profs``, defined at ``ROverR200``
        
        The fit assumes a density profile model specified in ``method``
        
        :param dens_profs: density profiles to be fit, units are irrelevant since fitting
            will be done on normalized profiles
        :type dens_profs: (N2, r_res) floats
        :param ROverR200: radii at which ``dens_profs`` are defined
        :type ROverR200: (r_res,) floats
        :param cat: each entry of the list is a list containing indices of particles belonging to an object,
            list is non-entry for an object only if ``dens_profs`` has a corresponding row
        :type cat: list of length N
        :param r200s: R_200 (mean not critical) radii of the parent halos
        :type r200s: (N,) floats, N > N2
        :param method: string describing density profile model assumed for fitting
        :type method: string, either `einasto`, `alpha_beta_gamma`, `hernquist`, `nfw`
        """
        print_status(rank,self.start_time,'Starting fitDensProfs() with snap {0}'.format(self.SNAP))
        
        if rank == 0:
            r200s = np.int32([r200s[i] for i in range(len(cat)) if cat[i] != []])
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
            np.savetxt('{0}/dens_prof_best_fits_{1}_{2}.txt'.format(self.CAT_DEST, method, self.SNAP), best_fits, fmt='%1.7e')
            np.savetxt('{0}/best_fits_r_over_r200_{1}_{2}.txt'.format(self.CAT_DEST, method, self.SNAP), ROverR200, fmt='%1.7e')
    
    def fetchDensProfsBestFits(self, method):
        """ Fetch best-fit results for density profile fitting
        
        :param method: string describing density profile model assumed for fitting
        :type method: string, either `einasto`, `alpha_beta_gamma`, `hernquist`, `nfw`
        :return: best-fits for each object, and normalized radii used to calculate best-fits
        :rtype: (N2, n) floats, where n is the number of free parameters in the model ``method``,
            and (N3,) floats"""
        print_status(rank,self.start_time,'Starting fetchDensProfsBestFits() with snap {0}'.format(self.SNAP))
        
        if rank == 0:
            try:
                best_fits = np.loadtxt('{0}/dens_prof_best_fits_{1}_{2}.txt'.format(self.CAT_DEST, method, self.SNAP))
                ROverR200 = np.loadtxt('{0}/best_fits_r_over_r200_{1}_{2}.txt'.format(self.CAT_DEST, method, self.SNAP))
            except OSError as e:
                raise ValueError("{0} Need to perform density profile fitting first.".format(e))
            return best_fits, ROverR200
        else:
            return None, None
    
    def fetchDensProfsDirectBinning(self):
        """ Fetch direct-binning-based density profiles
        
        For this method to succeed, the density profiles must have been calculated
        and stored beforehand, preferably via calcDensProfsDirectBinning.
        
        :return: density profiles, and normalized radii at which these are defined
        :rtype: (N2, n) floats, where n is the number of free parameters in the model,
            and (N3,) floats"""
        print_status(rank,self.start_time,'Starting fetchDensProfsDirectBinning() with snap {0}'.format(self.SNAP))
        
        if rank == 0:
            try:
                dens_profs = np.loadtxt('{0}/dens_profs_db_{1}.txt'.format(self.CAT_DEST, self.SNAP))
                ROverR200 = np.loadtxt('{0}/r_over_r200_db_{1}.txt'.format(self.CAT_DEST, self.SNAP))
            except OSError as e:
                raise ValueError("{0} Need to perform direct-binning-based density profile estimation first.".format(e))
            return dens_profs, ROverR200
        else:
            return None, None
    
    def fetchDensProfsKernelBased(self):
        """ Fetch kernel-based density profiles
        
        For this method to succeed, the density profiles must have been calculated
        and stored beforehand, preferably via calcDensProfsKernelBased.
        
        :return: density profiles, and normalized radii at which these are defined
        :rtype: (N2, n) floats, where n is the number of free parameters in the model,
            and (N3,) floats"""
        print_status(rank,self.start_time,'Starting fetchDensProfsDirectBinning() with snap {0}'.format(self.SNAP))
        
        if rank == 0:
            try:
                dens_profs = np.loadtxt('{0}/dens_profs_kb_{1}.txt'.format(self.CAT_DEST, self.SNAP))
                ROverR200 = np.loadtxt('{0}/r_over_r200_kb_{1}.txt'.format(self.CAT_DEST, self.SNAP))
            except OSError as e:
                raise ValueError("{0} Need to perform direct-binning-based density profile estimation first.".format(e))
            return dens_profs, ROverR200
        else:
            return None, None
    
    def fetchShapeCat(self, local, obj_type):
        """ Fetch all relevant shape-related data
        
        :param local: whether to read in local or global shape data
        :type local: boolean
        :param obj_type: either 'dm' or 'gx' or '' (latter in case of CosmicProfilesDirect 
            instance), depending on what object type we are interested in
        :type obj_type: string
        :return: obj_masses, obj_centers, d, q, s, major_full
        :rtype: (number_of_objs,) float array, (number_of_objs, 3) float array, 3 x (number_of_objs, D_BINS+1) 
            float arrays, (number_of_objs, D_BINS+1, 3) float array; D_BINS = self.D_BINS if 
            local == True or D_BINS = 0
        :raises:
            ValueError: if some data is not yet available for loading"""
        print_status(rank,self.start_time,'Starting fetchShapeCat() with snap {0}'.format(self.SNAP))
        
        if rank == 0:
            try:
                if obj_type == 'dm':
                    suffix = '_dm_'
                elif obj_type == 'gx':
                    suffix = '_gx'
                else:
                    suffix = ''
                d = np.loadtxt('{0}/d_{1}{2}{3}.txt'.format(self.CAT_DEST, "local" if local == True else "global", suffix, self.SNAP)) # Has shape (number_of_objs, D_BINS+1)
                q = np.loadtxt('{0}/q_{1}{2}{3}.txt'.format(self.CAT_DEST, "local" if local == True else "global", suffix, self.SNAP))
                s = np.loadtxt('{0}/s_{1}{2}{3}.txt'.format(self.CAT_DEST, "local" if local == True else "global", suffix, self.SNAP))
                if local == False:
                    d = np.array(d, ndmin=1)
                    d = d.reshape(d.shape[0], 1) # Has shape (number_of_objs, 1)
                    q = np.array(q, ndmin=1)
                    q = q.reshape(q.shape[0], 1) # Has shape (number_of_objs, 1)
                    s = np.array(s, ndmin=1)
                    s = s.reshape(s.shape[0], 1) # Has shape (number_of_objs, 1)
                else:
                    # Dealing with the case of 1 obj
                    if d.ndim == 1 and d.shape[0] == self.D_BINS+1:
                        d = d.reshape(1, self.D_BINS+1)
                        q = q.reshape(1, self.D_BINS+1)
                        s = s.reshape(1, self.D_BINS+1)
                major_full = np.loadtxt('{0}/major_{1}{2}{3}.txt'.format(self.CAT_DEST, "local" if local == True else "global", suffix, self.SNAP))
                if major_full.ndim == 2:
                    major_full = major_full.reshape(major_full.shape[0], major_full.shape[1]//3, 3) # Has shape (number_of_objs, D_BINS+1, 3)
                else:
                    if local == True:
                        if major_full.shape[0] == (self.D_BINS+1)*3:
                            major_full = major_full.reshape(1, self.D_BINS+1, 3)
                    else:
                        if major_full.shape[0] == 3:
                            major_full = major_full.reshape(1, 1, 3)
                obj_masses = np.loadtxt('{0}/m_{1}{2}{3}.txt'.format(self.CAT_DEST, "local" if local == True else "global", suffix, self.SNAP)) # Has shape (number_of_hs,)
                obj_centers = np.loadtxt('{0}/centers_{1}{2}{3}.txt'.format(self.CAT_DEST, "local" if local == True else "global", suffix, self.SNAP)) # Has shape (number_of_hs,3)
                return obj_masses, obj_centers, d, q, s, major_full
            except OSError as e: # Components for snap are not available 
                raise ValueError('Calling fetchShapeProfs() for snap {0} threw OSError: {1}.'.format(self.SNAP))
        else:
            return None, None, None, None, None, None

cdef class CosmicProfilesDirect(CosmicProfiles):
    """ Subclass to calculate morphology for already identified objects
    
    The particle indices of the objects identified are stored in ``cat``.\n
    
    The public methods are ``fetchCat()``, ``calcGlobalShapes()``, ``calcLocalShapes()``,
    ``plotGlobalEpsHisto()``, ``vizGlobalShapes()``, ``vizLocalShapes()``, 
    ``calcDensProfsDirectBinning()``, ``calcDensProfsKernelBased()`` and ``drawDensityProfiles()``."""
    cdef float[:,:] xyz
    cdef float[:] masses
    cdef int[:,:] cat_arr
    cdef float[:] r200
    cdef object cat

    def __init__(self, float[:,:] xyz, float[:] masses, cat, float[:] r200, str CAT_DEST, str VIZ_DEST, str SNAP, float L_BOX, int MIN_NUMBER_PTCS, int D_LOGSTART, int D_LOGEND, int D_BINS, float M_TOL, int N_WALL, int N_MIN, str CENTER, double start_time):
        """      
        :param xyz: positions of all (DM or star) particles in simulation box
        :type xyz: (N1 x 3) floats
        :param masses: masses of the particles expressed in unit mass (= 10^10 M_sun/h)
        :type masses: (N1 x 1) floats
        :param cat: each entry of the list is a list containing indices of particles belonging to an object
        :type cat: list of length N2
        :param r200: each entry of the list gives the R_200 (mean not critical) radius of the parent halo
        :type r200: list of length N2
        :param CAT_DEST: catalogue destination
        :type CAT_DEST: string
        :param VIZ_DEST: visualisation folder destination
        :type VIZ_DEST: string
        :param SNAP: e.g. '024'
        :type SNAP: string
        :param L_BOX: simulation box side length
        :type L_BOX: float, units: Mpc/h
        :param MIN_NUMBER_PTCS: minimum number of particles for object to qualify for morphology calculation
        :type MIN_NUMBER_PTCS: int
        :param D_LOGSTART: logarithm of minimum ellipsoidal radius of interest, in units of R200 of parent halo
        :type D_LOGSTART: int
        :param D_LOGEND: logarithm of maximum ellipsoidal radius of interest, in units of R200 of parent halo
        :type D_LOGEND: int
        :param D_BINS: number of ellipsoidal radii of interest minus 1 (i.e. number of bins)
        :type D_BINS: int
        :param M_TOL: convergence tolerance, eigenvalue fractions must differ by less than ``M_TOL``
            for iteration to stop
        :type M_TOL: float
        :param N_WALL: maximum permissible number of iterations
        :type N_WALL: float
        :param N_MIN: minimum number of particles (DM or star particle) in any iteration; 
            if undercut, shape is unclassified
        :type N_MIN: int
        :param CENTER: shape quantities will be calculated with respect to CENTER = 'mode' (point of highest density)
            or 'com' (center of mass) of each halo
        :type CENTER: str
        :param start_time: time of start of object initialization
        :type start_time: float"""
        super().__init__(CAT_DEST, VIZ_DEST, SNAP, L_BOX, MIN_NUMBER_PTCS, D_LOGSTART, D_LOGEND, D_BINS, M_TOL, N_WALL, N_MIN, CENTER, start_time)
        assert xyz.shape[0] == masses.shape[0], "xyz.shape[0] must be equal to masses.shape[0]"
        self.xyz = xyz.base
        self.masses = masses.base
        self.cat = cat
        self.r200 = r200.base
        if rank == 0:
            obj_centers, obj_masses = self.calcMassesCenters(cat, xyz, masses, MIN_NUMBER_PTCS, L_BOX, CENTER)
            np.savetxt('{0}/m_{1}.txt'.format(self.CAT_DEST, self.SNAP), obj_masses, fmt='%1.7e')
            np.savetxt('{0}/centers_{1}.txt'.format(self.CAT_DEST, self.SNAP), obj_centers, fmt='%1.7e')
        
    def fetchCat(self):
        """ Fetch catalogue
        
        :return cat: list of indices defining the objects
        :type cat: list of length N1, each consisting of a list of int indices"""
        print_status(rank,self.start_time,'Starting fetchCat() with snap {0}'.format(self.SNAP))
        
        if rank == 0:
            return self.cat
        else:
            return None
            
    def calcLocalShapes(self):   
        """ Calculates and saves local object shape catalogues"""  
        print_status(rank,self.start_time,'Starting calcLocalShapes() with snap {0}'.format(self.SNAP))
        
        if rank == 0:
                                        
                print_status(rank, self.start_time, "Number of halos is {0}. The number of valid halos (sufficient-resolution ones) is {1}".format(len(self.cat), np.array([0 for x in self.cat if x != []]).shape[0]))
                
                # Morphology: Local Shape
                print_status(rank, self.start_time, "Calculating local-shape morphologies with {0} processors. The average number of ptcs in the Halos is {1}".format(len(os.sched_getaffinity(0)), np.average(np.array(list(map(lambda x: len([x for x in self.cat if x != []][x]), range(len([x for x in self.cat if x != []]))))))))
                d, q, s, minor, inter, major, halos_center, halo_m, succeeded = self.getMorphLocal(self.xyz.base, self.cat, self.masses.base, self.r200.base, self.L_BOX, self.MIN_NUMBER_PTCS, self.D_LOGSTART, self.D_LOGEND, self.D_BINS, self.M_TOL, self.N_WALL, self.N_MIN, self.CENTER)
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
        """ Calculates and saves global object shape catalogues"""
        print_status(rank,self.start_time,'Starting calcGlobalShapes() with snap {0}'.format(self.SNAP))
        
        if rank == 0:
                
                # Morphology: Global Shape (with E1 at large radius)
                print_status(rank, self.start_time, "Calculating global morphologies with {0} processors. The average number of ptcs in the Halos is {1}".format(len(os.sched_getaffinity(0)), np.average(np.array(list(map(lambda x: len([x for x in self.cat if x != []][x]), range(len([x for x in self.cat if x != []]))))))))
                d, q, s, minor, inter, major, halos_center, halo_m = self.getMorphGlobal(self.xyz.base, self.cat, self.masses.base, self.r200.base, self.L_BOX, self.MIN_NUMBER_PTCS, self.M_TOL, self.N_WALL, self.N_MIN, self.CENTER)
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
        """ Visualize local shape of objects with numbers ``obj_numbers``
        
        :param obj_numbers: list of object indices for which to visualize local shapes
        :type obj_numbers: list of ints"""
        print_status(rank,self.start_time,'Starting vizLocalShapes() with snap {0}'.format(self.SNAP))
        
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
                    plt.xlabel(r"x (Mpc/h)")
                    plt.ylabel(r"y (Mpc/h)")
                    ax.set_zlabel(r"z (Mpc/h)")
                    ax.set_box_aspect([1,1,1])
                    set_axes_equal(ax)
                    fig.savefig("{}/LocalObj{}_{}.pdf".format(self.VIZ_DEST, obj_number, self.SNAP), bbox_inches='tight')
        
    def vizGlobalShapes(self, obj_numbers):
        """ Visualize global shape of objects with numbers ``obj_numbers``
        
        :param obj_numbers: list of object indices for which to visualize global shapes
        :type obj_numbers: list of ints"""
        print_status(rank,self.start_time,'Starting vizGlobalShapes() with snap {0}'.format(self.SNAP))

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
                    plt.xlabel(r"x (Mpc/h)")
                    plt.ylabel(r"y (Mpc/h)")
                    ax.set_zlabel(r"z (Mpc/h)")
                    ax.set_box_aspect([1,1,1])
                    set_axes_equal(ax)
                    fig.savefig("{}/GlobalObj{}_{}.pdf".format(self.VIZ_DEST, obj_number, self.SNAP), bbox_inches='tight')
                    
    def plotGlobalEpsHisto(self):
        """ Plot ellipticity histogram"""
        print_status(rank,self.start_time,'Starting plotGlobalEpsHisto() with snap {0}'.format(self.SNAP))
        
        if rank == 0:
            with open('{0}/cat_global_{1}.txt'.format(self.CAT_DEST, self.SNAP), 'r') as filehandle:
                cat = json.load(filehandle)
            suffix = '_'
            getGlobalEpsHisto(cat, self.xyz, self.masses, self.L_BOX, self.VIZ_DEST, self.SNAP, suffix = suffix, HIST_NB_BINS = 11)

    def calcDensProfsDirectBinning(self, ROverR200):
        """ Calculate direct-binning-based density profiles
        
        :param ROverR200: At which unitless radial values to calculate density profiles
        :type ROverR200: float array"""
        print_status(rank,self.start_time,'Starting calcDensProfsDirectBinning() with snap {0}'.format(self.SNAP))
        
        if rank == 0:
            obj_keep = np.int32([1 if x != [] else 0 for x in self.cat])
            ROverR200, dens_profs = getDensProfsDirectBinning(self.cat, self.xyz.base, obj_keep, self.masses.base, self.r200.base, np.float32(ROverR200), self.MIN_NUMBER_PTCS, self.L_BOX, self.CENTER)
            
            MASS_UNIT = 1e+10
            np.savetxt('{0}/dens_profs_db_{1}.txt'.format(self.CAT_DEST, self.SNAP), dens_profs*MASS_UNIT, fmt='%1.7e') # In units of M_sun*h^2/Mpc^3
            np.savetxt('{0}/r_over_r200_db_{1}.txt'.format(self.CAT_DEST, self.SNAP), ROverR200, fmt='%1.7e')
    
    def calcDensProfsKernelBased(self, ROverR200):
        """ Calculate kernel-based density profiles
        
        :param ROverR200: At which unitless radial values to calculate density profiles
        :type ROverR200: float array"""
        print_status(rank,self.start_time,'Starting calcDensProfsKernelBased() with snap {0}'.format(self.SNAP))
        
        if rank == 0:
            obj_keep = np.int32([1 if x != [] else 0 for x in self.cat])
            ROverR200, dens_profs = getDensProfsKernelBased(self.cat, self.xyz.base, obj_keep, self.masses.base, self.r200.base, np.float32(ROverR200), self.MIN_NUMBER_PTCS, self.L_BOX, self.CENTER)
            
            MASS_UNIT = 1e+10
            np.savetxt('{0}/dens_profs_kb_{1}.txt'.format(self.CAT_DEST, self.SNAP), dens_profs*MASS_UNIT, fmt='%1.7e') # In units of M_sun*h^2/Mpc^3
            np.savetxt('{0}/r_over_r200_kb_{1}.txt'.format(self.CAT_DEST, self.SNAP), ROverR200, fmt='%1.7e')

    def drawDensityProfiles(self, dens_profs, ROverR200, cat, r200s, method):
        """ Draws some simplistic density profiles
        
        :param dens_profs: density profiles to be fit, in units of M_sun*h^2/(Mpc)**3
        :type dens_profs: (N2, r_res) floats
        :param ROverR200: radii at which ``dens_profs`` are defined
        :type ROverR200: (r_res,) floats
        :param cat: each entry of the list is a list containing indices of particles belonging to an object,
            list is non-entry for an object only if ``dens_profs`` has a corresponding row
        :type cat: list of length N
        :param r200s: R_200 (mean not critical) radii of the parent halos
        :type r200s: (N,) floats, N > N2
        :param method: string describing density profile model assumed for fitting
        :type method: string, either `einasto`, `alpha_beta_gamma`, `hernquist`, `nfw`
        """
        print_status(rank,self.start_time,'Starting drawDensityProfiles() with snap {0}'.format(self.SNAP))
        
        if rank == 0:
            suffix = '_'
            with open('{0}/cat_{1}.txt'.format(self.CAT_DEST, self.SNAP), 'r') as filehandle:
                cat = json.load(filehandle)
            # Calculate obj_masses and obj_centers
            obj_centers, obj_masses = self.fetchMassesCenters('')
            best_fits, fits_ROverR200 = self.fetchDensProfsBestFits(method)
            getDensityProfiles(self.VIZ_DEST, self.SNAP, cat, r200s, fits_ROverR200, dens_profs, ROverR200, obj_masses, obj_centers, method, self.start_time, MASS_UNIT=1e10, suffix = suffix)
    
cdef class CosmicProfilesGadgetHDF5(CosmicProfiles):
    """ Subclass to calculate morphology for yet to-be-identified objects in Gadget HDF5 simulation
    
    The public methods are ``fetchR200s()``, ``fetchHaloCat()``, ``fetchGxCat()``, ``calcGlobalShapesDM()``, 
    ``calcLocalShapesDM()``, ``calcGlobalShapesGx()``, ``calcLocalShapesGx()``, ``calcGlobalVelShapesDM()``, 
    ``calcLocalVelShapesDM()``, ``loadDMCat()``, ``plotGlobalEpsHisto()``, 
    ``vizGlobalShapes()``, ``vizLocalShapes()``, ``calcDensProfsDirectBinning()``,
    ``calcDensProfsKernelBased()`` and ``drawDensityProfiles()``."""
    cdef str HDF5_SNAP_DEST
    cdef str HDF5_GROUP_DEST
    cdef int MIN_NUMBER_STAR_PTCS
    cdef int SNAP_MAX
    cdef float[:] r200
    
    def __init__(self, str HDF5_SNAP_DEST, str HDF5_GROUP_DEST, str CAT_DEST, str VIZ_DEST, str SNAP, int SNAP_MAX, float L_BOX, int MIN_NUMBER_PTCS, int MIN_NUMBER_STAR_PTCS, int D_LOGSTART, int D_LOGEND, int D_BINS, float M_TOL, int N_WALL, int N_MIN, str CENTER, double start_time):
        """ 
        :param HDF5_SNAP_DEST: where we can find the snapshot
        :type HDF5_SNAP_DEST: string
        :param HDF5_GROUP_DEST: where we can find the group files
        :type HDF5_GROUP_DEST: string
        :param CAT_DEST: catalogue destination
        :type CAT_DEST: string
        :param VIZ_DEST: visualisation folder destination
        :type VIZ_DEST: string
        :param SNAP: e.g. '024'
        :type SNAP: string
        :param SNAP_MAX: e.g. '024'
        :type SNAP_MAX: string
        :param L_BOX: simulation box side length
        :type L_BOX: float, units: Mpc/h
        :param MIN_NUMBER_PTCS: minimum number of particles for halo to qualify for morphology calculation
        :type MIN_NUMBER_PTCS: int
        :param MIN_NUMBER_STAR_PTCS: minimum number of particles for galaxy (to-be-identified) to qualify for morphology calculation
        :type MIN_NUMBER_STAR_PTCS: int
        :param D_LOGSTART: logarithm of minimum ellipsoidal radius of interest, in units of R200 of parent halo
        :type D_LOGSTART: int
        :param D_LOGEND: logarithm of maximum ellipsoidal radius of interest, in units of R200 of parent halo
        :type D_LOGEND: int
        :param D_BINS: number of ellipsoidal radii of interest minus 1 (i.e. number of bins)
        :type D_BINS: int
        :param M_TOL: convergence tolerance, eigenvalue fractions must differ by less than ``M_TOL``
            for iteration to stop
        :type M_TOL: float
        :param N_WALL: maximum permissible number of iterations
        :type N_WALL: float
        :param N_MIN: minimum number of particles (DM or star particle) in any iteration; 
            if undercut, shape is unclassified
        :type N_MIN: int
        :param CENTER: shape quantities will be calculated with respect to CENTER = 'mode' (point of highest density)
            or 'com' (center of mass) of each halo
        :type CENTER: str
        :param start_time: time of start of object initialization
        :type start_time: float"""
        super().__init__(CAT_DEST, VIZ_DEST, SNAP, L_BOX, MIN_NUMBER_PTCS, D_LOGSTART, D_LOGEND, D_BINS, M_TOL, N_WALL, N_MIN, CENTER, start_time)
        self.HDF5_SNAP_DEST = HDF5_SNAP_DEST
        self.HDF5_GROUP_DEST = HDF5_GROUP_DEST
        self.MIN_NUMBER_STAR_PTCS = MIN_NUMBER_STAR_PTCS
        self.SNAP_MAX = SNAP_MAX
            
    def fetchR200s(self):
        """ Fetch the virial radii"""
        print_status(rank,self.start_time,'Starting fetchR200s() with snap {0}'.format(self.SNAP))
        
        if rank == 0:
            return self.r200.base
        else:
            return None
    
    def fetchHaloCat(self):
        """ Fetch halo catalogue"""
        print_status(rank,self.start_time,'Starting fetchHaloCat() with snap {0}'.format(self.SNAP))
        
        if rank == 0:
            with open('{0}/h_cat_{1}.txt'.format(self.CAT_DEST, self.SNAP), 'r') as filehandle:
                h_cat = json.load(filehandle)
            return h_cat
        else:
            return None
    
    def fetchGxCat(self):
        """ Fetch gx catalogue"""
        print_status(rank,self.start_time,'Starting fetchGxCat() with snap {0}'.format(self.SNAP))
        
        if rank == 0:
            with open('{0}/gx_cat_{1}.txt'.format(self.CAT_DEST, self.SNAP), 'r') as filehandle:
                gx_cat = json.load(filehandle)
            return gx_cat
    
    def vizLocalShapes(self, obj_numbers, obj_type = 'dm'):
        """ Visualize local shape of objects with numbers ``obj_numbers``"""
        print_status(rank,self.start_time,'Starting vizLocalShapes() with snap {0}'.format(self.SNAP))
        
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
                    plt.xlabel(r"x (Mpc/h)")
                    plt.ylabel(r"y (Mpc/h)")
                    ax.set_zlabel(r"z (Mpc/h)")
                    ax.set_box_aspect([1,1,1])
                    set_axes_equal(ax)
                    fig.savefig("{}/Local{}Obj{}_{}.pdf".format(self.VIZ_DEST, obj_type.upper(), obj_number, self.SNAP), bbox_inches='tight')
        
    def vizGlobalShapes(self, obj_numbers, obj_type = 'dm'):
        """ Visualize global shape of objects with numbers ``obj_numbers``"""   
        print_status(rank,self.start_time,'Starting vizGlobalShapes() with snap {0}'.format(self.SNAP))
        
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
                    plt.xlabel(r"x (Mpc/h)")
                    plt.ylabel(r"y (Mpc/h)")
                    ax.set_zlabel(r"z (Mpc/h)")
                    ax.set_box_aspect([1,1,1])
                    set_axes_equal(ax)
                    fig.savefig("{}/Global{}Obj{}_{}.pdf".format(self.VIZ_DEST, obj_type.upper(), obj_number, self.SNAP), bbox_inches='tight')
    
    def loadDMCat(self):
        """ Loads halo (more precisely: CSH) catalogues from FOF data
        
        Stores R200 as self.r200"""
        print_status(rank,self.start_time,'Starting loadDMCat() with snap {0}'.format(self.SNAP))
        
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
            
            dm_xyz, dm_masses, dm_smoothing, dm_velxyz = getHDF5DMData(self.HDF5_SNAP_DEST, self.SNAP_MAX, self.SNAP)
            del dm_smoothing; del dm_velxyz
            obj_centers, obj_masses = self.calcMassesCenters(h_cat_l, dm_xyz, dm_masses, self.MIN_NUMBER_PTCS, self.L_BOX, self.CENTER)
            np.savetxt('{0}/m_dm_{1}.txt'.format(self.CAT_DEST, self.SNAP), obj_masses, fmt='%1.7e')
            np.savetxt('{0}/centers_dm_{1}.txt'.format(self.CAT_DEST, self.SNAP), obj_centers, fmt='%1.7e')
            del dm_xyz; del dm_masses
    
    def loadGxCat(self):
        """ Loads galaxy catalogues from HDF5 data
        
        To discard wind particles, add the line
        `gx_cat_l = [[x for x in gx if is_star[x]] for gx in gx_cat_l]` before
        saving the catalogue."""
        print_status(rank,self.start_time,'Starting loadGxCat() with snap {0}'.format(self.SNAP))
        
        # Import hdf5 data
        print_status(rank,self.start_time,"Getting HDF5 raw data..")
        nb_shs, sh_len_gx, fof_gx_sizes = getHDF5SHGxData(self.HDF5_GROUP_DEST, self.SNAP_MAX, self.SNAP)
        if rank != 0:
            del nb_shs; del sh_len_gx; del fof_gx_sizes
        if rank == 0:
            # Construct gx catalogue
            print_status(rank, self.start_time, "Creating Gx CAT..")
            gx_cat, gx_pass = getGxCat(np.array(nb_shs), np.array(sh_len_gx), np.array(fof_gx_sizes), self.MIN_NUMBER_STAR_PTCS)
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
            
            star_xyz, fof_com, sh_com, nb_shs, star_masses, star_smoothing, is_star = getHDF5GxData(self.HDF5_SNAP_DEST, self.HDF5_GROUP_DEST, self.SNAP_MAX, self.SNAP)
            obj_centers, obj_masses = self.calcMassesCenters(gx_cat_l, star_xyz, star_masses, self.MIN_NUMBER_STAR_PTCS, self.L_BOX, self.CENTER)
            np.savetxt('{0}/m_gx_{1}.txt'.format(self.CAT_DEST, self.SNAP), obj_masses, fmt='%1.7e')
            np.savetxt('{0}/centers_gx_{1}.txt'.format(self.CAT_DEST, self.SNAP), obj_centers, fmt='%1.7e')
            del star_xyz; del fof_com; del sh_com; del nb_shs; del star_masses; del star_smoothing; del is_star
    
    def calcLocalShapesGx(self):
        """ Calculates and saves local galaxy shape catalogues"""
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
        """ Calculates and saves global galaxy shape catalogues"""
        print_status(rank,self.start_time,'Starting calcGlobalShapesGx() with snap {0}'.format(self.SNAP))
        
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
        """ Calculates and saves local halo shape catalogues"""
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
        """ Calculates and saves global halo shape catalogues"""
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
        """ Calculates and saves local velocity dispersion tensor shape catalogues"""
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
        """ Calculates and saves global velocity dispersion tensor shape catalogues"""
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
        """ Calculate direct-binning-based density profiles
        
        :param ROverR200: At which unitless radial values to calculate density profiles
        :type ROverR200: float array
        :param obj_type: either 'dm' or 'gx', depending on what catalogue 
            the ellipticity histogram should be plotted for
        :type obj_type: string
        """
        print_status(rank,self.start_time,'Starting calcDensProfsDirectBinning() with snap {0}'.format(self.SNAP))

        if rank == 0:
            if obj_type == 'dm':
                with open('{0}/h_cat_{1}.txt'.format(self.CAT_DEST, self.SNAP), 'r') as filehandle:
                    cat = json.load(filehandle)
                xyz, masses, smoothing, velxyz = getHDF5DMData(self.HDF5_SNAP_DEST, self.SNAP_MAX, self.SNAP)
                del smoothing; del velxyz
                MIN_NB_PTCS = self.MIN_NUMBER_PTCS
            elif obj_type == 'gx':
                with open('{0}/gx_cat_{1}.txt'.format(self.CAT_DEST, self.SNAP), 'r') as filehandle:
                    cat = json.load(filehandle)
                xyz, fof_com, sh_com, nb_shs, masses, smoothing, is_star = getHDF5GxData(self.HDF5_SNAP_DEST, self.HDF5_GROUP_DEST, self.SNAP_MAX, self.SNAP)
                del smoothing; del fof_com; del sh_com; del nb_shs; del is_star
                MIN_NB_PTCS = self.MIN_NUMBER_STAR_PTCS
            else:
                raise ValueError("For a GadgetHDF5 simulation, 'obj_type' must be either 'dm' or 'gx'")
            obj_keep = np.int32([1 if x != [] else 0 for x in cat])
            ROverR200, dens_profs = getDensProfsDirectBinning(cat, xyz, obj_keep, masses, self.r200, np.float32(ROverR200), MIN_NB_PTCS, self.L_BOX, self.CENTER)
            
            MASS_UNIT = 1e+10
            np.savetxt('{0}/dens_profs_db_{1}.txt'.format(self.CAT_DEST, self.SNAP), dens_profs*MASS_UNIT, fmt='%1.7e') # In units of M_sun*h^2/Mpc^3
            np.savetxt('{0}/r_over_r200_db_{1}.txt'.format(self.CAT_DEST, self.SNAP), ROverR200, fmt='%1.7e')
    
    def calcDensProfsKernelBased(self, ROverR200, obj_type = ''):
        """ Calculate kernel-based density profiles
        
        :param ROverR200: At which unitless radial values to calculate density profiles
        :type ROverR200: float array
        :param obj_type: either 'dm' or 'gx', depending on what catalogue 
            the ellipticity histogram should be plotted for
        :type obj_type: string
        """
        print_status(rank,self.start_time,'Starting calcDensProfsKernelBased() with snap {0}'.format(self.SNAP))
        
        if rank == 0:
            if obj_type == 'dm':
                with open('{0}/h_cat_{1}.txt'.format(self.CAT_DEST, self.SNAP), 'r') as filehandle:
                    cat = json.load(filehandle)
                xyz, masses, smoothing, velxyz = getHDF5DMData(self.HDF5_SNAP_DEST, self.SNAP_MAX, self.SNAP)
                del smoothing; del velxyz
                MIN_NB_PTCS = self.MIN_NUMBER_DM_PTCS
            elif obj_type == 'gx':
                with open('{0}/gx_cat_{1}.txt'.format(self.CAT_DEST, self.SNAP), 'r') as filehandle:
                    cat = json.load(filehandle)
                xyz, fof_com, sh_com, nb_shs, masses, smoothing, is_star = getHDF5GxData(self.HDF5_SNAP_DEST, self.HDF5_GROUP_DEST, self.SNAP_MAX, self.SNAP)
                del smoothing; del fof_com; del sh_com; del nb_shs; del is_star
                MIN_NB_PTCS = self.MIN_NUMBER_STAR_PTCS
            else:
                raise ValueError("For a GadgetHDF5 simulation, 'obj_type' must be either 'dm' or 'gx'")
            obj_keep = np.int32([1 if x != [] else 0 for x in cat])
            ROverR200, dens_profs = getDensProfsKernelBased(cat, xyz, obj_keep, masses, self.r200, np.float32(ROverR200), MIN_NB_PTCS, self.L_BOX, self.CENTER)
            
            MASS_UNIT = 1e+10
            np.savetxt('{0}/dens_profs_kb_{1}.txt'.format(self.CAT_DEST, self.SNAP), dens_profs*MASS_UNIT, fmt='%1.7e') # In units of M_sun*h^2/Mpc^3
            np.savetxt('{0}/r_over_r200_kb_{1}.txt'.format(self.CAT_DEST, self.SNAP), ROverR200, fmt='%1.7e')
    
    def plotGlobalEpsHisto(self, obj_type = ''):
        """ Plot ellipticity histogram
        
        :param obj_type: either 'dm' or 'gx', depending on what catalogue 
            the ellipticity histogram should be plotted for
        :type obj_type: string"""
        print_status(rank,self.start_time,'Starting plotGlobalEpsHisto() with snap {0}'.format(self.SNAP))
        
        if rank == 0:
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
            
    def drawDensityProfiles(self, dens_profs, ROverR200, cat, r200s, method, obj_type = ''):
        """ Draws some simplistic density profiles
        
        :param dens_profs: density profiles to be fit, in units of M_sun*h^2/(Mpc)**3
        :type dens_profs: (N2, r_res) floats
        :param ROverR200: radii at which ``dens_profs`` are defined
        :type ROverR200: (r_res,) floats
        :param cat: each entry of the list is a list containing indices of particles belonging to an object,
            list is non-entry for an object only if ``dens_profs`` has a corresponding row
        :type cat: list of length N
        :param r200s: R_200 (mean not critical) radii of the parent halos
        :type r200s: (N,) floats, N > N2
        :param method: string describing density profile model assumed for fitting
        :type method: string, either `einasto`, `alpha_beta_gamma`, `hernquist`, `nfw`
        :param obj_type: either 'dm' or 'gx' for CosmicProfilesGadgetHDF5 or '' for CosmicProfilesDirect
        :type obj_type: string"""
        print_status(rank,self.start_time,'Starting drawDensityProfiles() with snap {0}'.format(self.SNAP))
        
        if rank == 0:
            if obj_type == 'dm':
                suffix = '_dm_'
                with open('{0}/h_cat_{1}.txt'.format(self.CAT_DEST, self.SNAP), 'r') as filehandle:
                    cat = json.load(filehandle)
            elif obj_type == 'gx':
                suffix = '_gx_'
                with open('{0}/gx_cat_{1}.txt'.format(self.CAT_DEST, self.SNAP), 'r') as filehandle:
                    cat = json.load(filehandle)
            else:
                raise ValueError("For a GadgetHDF5 simulation, 'obj_type' must be either 'dm' or 'gx'")
            # Calculate obj_masses and obj_centers
            obj_centers, obj_masses = self.fetchMassesCenters(obj_type)
            best_fits, fits_ROverR200 = self.fetchDensProfsBestFits(method)
            getDensityProfiles(self.VIZ_DEST, self.SNAP, cat, r200s, fits_ROverR200, dens_profs, ROverR200, obj_masses, obj_centers, method, self.start_time, MASS_UNIT=1e10, suffix = suffix)
