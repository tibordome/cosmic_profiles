#cython: language_level=3
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot
from numpy.random import default_rng
from matplotlib.font_manager import FontProperties
import matplotlib
matplotlib.rcParams.update({'font.size': 13})
import json
import h5py
from pathos.multiprocessing import ProcessingPool as Pool
from functools import partial
import os
from mpl_toolkits.mplot3d import Axes3D
import time
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
cimport cython

@cython.embedsignature(True)
cdef class CosmicProfiles:
    """ Parent class governing low-level cosmic shape calculations
    
    Its public methods are ``fetchCatLocal()``, ``fetchCatGlobal()``, ``getDensProfsDirectBinning()``,
    ``getDensProfsKernelBased()``, ``runS1()``, ``runE1()``, ``runE1VelDisp()``, ``getObjMorphLocal()``, ``getObjMorphGlobal()``, 
    ``getObjMorphLocalVelDisp()``, ``getObjMorphGlobalVelDisp()``, ``getMorphLocal()``, ``getMorphGlobal()``, 
    ``getMorphLocalVelDisp()``, ``getMorphGlobalVelDisp()``, ``drawShapeProfiles()``, ``plotLocalTHisto()``, 
    ``fitDensProfs()``, ``fetchDensProfsBestFits()``, ``fetchDensProfsDirectBinning()``,
    ``fetchDensProfsKernelBased()`` and ``fetchShapeCat()``."""
    
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
        return
    
    def fetchMassesCenters(self, obj_type):
        """ Calculate total mass and centers of objects
        
        :param obj_type: either 'dm' or 'gx' for CosmicProfilesGadgetHDF5 or '' for CosmicProfilesDirect
        :type obj_type: string
        :return centers, m: centers and masses
        :rtype: (N,3) and (N,) floats"""
        return
    
    def fetchCatLocal(self, obj_type = 'dm'):
        """ Fetch local halo/gx catalogue
        
        :param obj_type: either 'dm' or 'gx' for CosmicProfilesGadgetHDF5 or '' for CosmicProfilesDirect
        :type obj_type: string
        :return cat_local: list of indices defining the objects
        :type cat_local: list of length N1, each consisting of a list of int indices"""
        return
    
    def fetchCatGlobal(self, obj_type = 'dm'):
        """ Fetch global halo/gx catalogue
        
        :param obj_type: either 'dm' or 'gx' for CosmicProfilesGadgetHDF5 or '' for CosmicProfilesDirect
        :type obj_type: string
        :return cat_global: list of indices defining the objects
        :type cat_global: list of length N1, each consisting of a list of int indices"""
        return
    
    def getObjMorphLocal(self, float[:,:] morph_info, float r200, float[:] log_d, float[:,:] xyz, float[:,:] xyz_princ, float[:] masses, int[:] shell, float[:] center, complex[::1,:] shape_tensor, double[::1] eigval, complex[::1,:] eigvec, float M_TOL, int N_WALL, int N_MIN):
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
        return
    
    def getObjMorphGlobal(self, float[:] morph_info, float r200, float[:,:] xyz, float[:,:] xyz_princ, float[:] masses, int[:] ellipsoid, float[:] center, complex[::1,:] shape_tensor, double[::1] eigval, complex[::1,:] eigvec, float M_TOL, int N_WALL, int N_MIN):
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
        return
    
    def getObjMorphLocalVelDisp(self, float[:,:] morph_info, float r200, float[:] log_d, float[:,:] xyz, float[:,:] vxyz, float[:,:] xyz_princ, float[:] masses, int[:] shell, float[:] center, float[:] vcenter, complex[::1,:] shape_tensor, double[::1] eigval, complex[::1,:] eigvec, float M_TOL, int N_WALL, int N_MIN):
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
        return
    
    def getObjMorphGlobalVelDisp(self, float[:] morph_info, float r200, float[:,:] xyz, float[:,:] vxyz, float[:,:] xyz_princ, float[:] masses, int[:] ellipsoid, float[:] center, float[:] vcenter, complex[::1,:] shape_tensor, double[::1] eigval, complex[::1,:] eigvec, float M_TOL, int N_WALL, int N_MIN):
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
        return
    
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
        return
    
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
        return
    
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
        return
    
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
        return
    
    def drawShapeProfiles(self, obj_type = ''):
        """ Draws some simplistic shape profiles
        
        :param obj_type: either 'dm' or 'gx' for CosmicProfilesGadgetHDF5 or '' for CosmicProfilesDirect
        :type obj_type: string"""
        return
    
    def plotLocalTHisto(self, obj_type = ''):
        """ Plot the triaxiality-histogram
        
        :param obj_type: either 'dm' or 'gx' for CosmicProfilesGadgetHDF5 or '' for CosmicProfilesDirect
        :type obj_type: string"""
        return
    
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
        return
    
    def fetchDensProfsBestFits(self, method):
        """ Fetch best-fit results for density profile fitting
        
        :param method: string describing density profile model assumed for fitting
        :type method: string, either `einasto`, `alpha_beta_gamma`, `hernquist`, `nfw`
        :return: best-fits for each object, and normalized radii used to calculate best-fits
        :rtype: (N2, n) floats, where n is the number of free parameters in the model ``method``,
            and (N3,) floats"""
        return
    
    def fetchDensProfsDirectBinning(self):
        """ Fetch direct-binning-based density profiles
        
        For this method to succeed, the density profiles must have been calculated
        and stored beforehand, preferably via calcDensProfsDirectBinning.
        
        :return: density profiles, and normalized radii at which these are defined
        :rtype: (N2, n) floats, where n is the number of free parameters in the model,
            and (N3,) floats"""
        return
    
    def fetchDensProfsKernelBased(self):
        """ Fetch kernel-based density profiles
        
        For this method to succeed, the density profiles must have been calculated
        and stored beforehand, preferably via calcDensProfsKernelBased.
        
        :return: density profiles, and normalized radii at which these are defined
        :rtype: (N2, n) floats, where n is the number of free parameters in the model,
            and (N3,) floats"""
        return
    
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
        return

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
        
    def fetchCat(self):
        """ Fetch catalogue
        
        :return cat: list of indices defining the objects
        :type cat: list of length N1, each consisting of a list of int indices"""
        return
            
    def calcLocalShapes(self):   
        """ Calculates and saves local object shape catalogues"""  
        return
                
    def calcGlobalShapes(self):
        """ Calculates and saves global object shape catalogues"""
        return
    
    def vizLocalShapes(self, obj_numbers):
        """ Visualize local shape of objects with numbers ``obj_numbers``
        
        :param obj_numbers: list of object indices for which to visualize local shapes
        :type obj_numbers: list of ints"""
        return
        
    def vizGlobalShapes(self, obj_numbers):
        """ Visualize global shape of objects with numbers ``obj_numbers``
        
        :param obj_numbers: list of object indices for which to visualize global shapes
        :type obj_numbers: list of ints"""
        return
                    
    def plotGlobalEpsHisto(self):
        """ Plot ellipticity histogram"""
        return

    def calcDensProfsDirectBinning(self, ROverR200):
        """ Calculate direct-binning-based density profiles
        
        :param ROverR200: At which unitless radial values to calculate density profiles
        :type ROverR200: float array"""
        return
    
    def calcDensProfsKernelBased(self, ROverR200):
        """ Calculate kernel-based density profiles
        
        :param ROverR200: At which unitless radial values to calculate density profiles
        :type ROverR200: float array"""
        return

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
        return
    
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
        return
    
    def fetchHaloCat(self):
        """ Fetch halo catalogue"""
        return
    
    def fetchGxCat(self):
        """ Fetch gx catalogue"""
        return
    
    def vizLocalShapes(self, obj_numbers, obj_type = 'dm'):
        """ Visualize local shape of objects with numbers ``obj_numbers``"""
        return
        
    def vizGlobalShapes(self, obj_numbers, obj_type = 'dm'):
        """ Visualize global shape of objects with numbers ``obj_numbers``"""   
        return
    
    def loadDMCat(self):
        """ Loads halo (more precisely: CSH) catalogues from FOF data
        
        Stores R200 as self.r200"""
        return
    
    def loadGxCat(self):
        """ Loads galaxy catalogues from HDF5 data
        
        To discard wind particles, add the line
        `gx_cat_l = [[x for x in gx if is_star[x]] for gx in gx_cat_l]` before
        saving the catalogue."""
        return
    
    def calcLocalShapesGx(self):
        """ Calculates and saves local galaxy shape catalogues"""
        return
            
    def calcGlobalShapesGx(self):
        """ Calculates and saves global galaxy shape catalogues"""
        return
    
    def calcLocalShapesDM(self):   
        """ Calculates and saves local halo shape catalogues"""
        return
            
    def calcGlobalShapesDM(self):
        """ Calculates and saves global halo shape catalogues"""
        return
            
    def calcLocalVelShapesDM(self):
        """ Calculates and saves local velocity dispersion tensor shape catalogues"""
        return
    
    def calcGlobalVelShapesDM(self):
        """ Calculates and saves global velocity dispersion tensor shape catalogues"""
        return
            
    def calcDensProfsDirectBinning(self, ROverR200, obj_type = ''):
        """ Calculate direct-binning-based density profiles
        
        :param ROverR200: At which unitless radial values to calculate density profiles
        :type ROverR200: float array
        :param obj_type: either 'dm' or 'gx', depending on what catalogue 
            the ellipticity histogram should be plotted for
        :type obj_type: string
        """
        return
    
    def calcDensProfsKernelBased(self, ROverR200, obj_type = ''):
        """ Calculate kernel-based density profiles
        
        :param ROverR200: At which unitless radial values to calculate density profiles
        :type ROverR200: float array
        :param obj_type: either 'dm' or 'gx', depending on what catalogue 
            the ellipticity histogram should be plotted for
        :type obj_type: string
        """
        return
    
    def plotGlobalEpsHisto(self, obj_type = ''):
        """ Plot ellipticity histogram
        
        :param obj_type: either 'dm' or 'gx', depending on what catalogue 
            the ellipticity histogram should be plotted for
        :type obj_type: string"""
        return
            
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
        return