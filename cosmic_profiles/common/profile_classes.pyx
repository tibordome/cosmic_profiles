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
from cosmic_profiles.common.python_routines import print_status, set_axes_equal, fibonacci_ellipsoid, respectPBCNoRef, calcCoM
from cosmic_profiles.shape_profs.shape_profs_tools import getGlobalEpsHist, getLocalEpsHist, getLocalTHist, getGlobalTHist, getShapeProfs
from cosmic_profiles.dens_profs.dens_profs_tools import getDensProfs, fitDensProfHelper
from cosmic_profiles.gadget_hdf5.get_hdf5 import getHDF5Data, getHDF5GxData, getHDF5SHDMData, getHDF5SHGxData, getHDF5DMData
from cosmic_profiles.gadget_hdf5.gen_catalogues import calcCSHCat, calcGxCat
from cosmic_profiles.cython_helpers.helper_class cimport CythonHelpers
from cosmic_profiles.shape_profs.shape_profs_algos cimport runEllShellAlgo, runEllAlgo, runEllVDispAlgo
from cosmic_profiles.shape_profs.shape_profs_algos import calcMorphLocal, calcMorphGlobal, calcMorphLocalVelDisp, calcMorphGlobalVelDisp
from cosmic_profiles.dens_profs.dens_profs_algos import calcMassesCenters, calcDensProfsDirectBinning, calcDensProfsKernelBased
import time
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

@cython.embedsignature(True)
cdef class CosmicBase:
    """ Parent class governing high-level cosmic shape calculations
    
    Its public methods are ``getR200s()``, ``getMassesCentersBase()``, ``getIdxCatLocalBase()``,
    ``getIdxCatGlobalBase()``, ``getIdxCatVelLocalBase()``, ``getIdxCatVelGlobalBase()``, 
    ``getShapeCatLocalBase()``, ``getShapeCatGlobalBase()``, ``getShapeCatVelLocalBase()``, 
    ``getShapeCatVelGlobalBase()``, ``dumpShapeCatLocalBase()``, ``dumpShapeCatGlobalBase()``, 
    ``dumpShapeCatVelLocalBase()``, ``dumpShapeCatVelGlobalBase()``, ``plotShapeProfsBase()``,
    ``plotLocalTHistBase()``, ``plotGlobalTHistBase()``, ``getDensProfsBestFitsBase()``,
    ``getConcentrationsBase()``, ``getDensProfsDirectBinningBase()``, ``getDensProfsKernelBasedBase()``,
    ``getObjInfoLocalBase()``, ``getObjInfoGlobalBase()``, ``getObjInfoVelLocalBase()``,
    ``getObjInfoVelGlobalBase()``"""
    cdef str SNAP
    cdef float L_BOX
    cdef double start_time
    cdef float[:] r200
    cdef str CENTER
    cdef float SAFE # Units: Mpc/h. Ellipsoidal radius will be maxdist(COM,point)+SAFE where point is any point in the point cloud. The larger the better.
    cdef float MASS_UNIT
    cdef int MIN_NUMBER_PTCS
    
    def __init__(self, str SNAP, float L_BOX, int MIN_NUMBER_PTCS, str CENTER, double start_time):
        """
        :param SNAP: snapshot identifier, e.g. '024'
        :type SNAP: string
        :param L_BOX: simulation box side length
        :type L_BOX: float, units: Mpc/h
        :param MIN_NUMBER_PTCS: minimum number of particles for object to qualify for morphology calculation
        :type MIN_NUMBER_PTCS: int
        :param CENTER: shape quantities will be calculated with respect to CENTER = 'mode' (point of highest density)
            or 'com' (center of mass) of each halo
        :type CENTER: str
        :param start_time: time of start of object initialization
        :type start_time: float"""
        self.SNAP = SNAP
        self.L_BOX = L_BOX
        self.CENTER = CENTER
        self.MIN_NUMBER_PTCS = MIN_NUMBER_PTCS
        self.start_time = start_time
        self.SAFE = 6
        self.MASS_UNIT = 1e10
    
    def getR200s(self):
        """ Get overdensity radii"""
        print_status(rank,self.start_time,'Starting getR200s() with snap {0}'.format(self.SNAP))
        
        if rank == 0:
            return self.r200.base
        else:
            return None
        
    def getMassesCentersBase(self, float[:,:] xyz, float[:] masses, idx_cat, int MIN_NUMBER_PTCS):
        """ Calculate total mass and centers of objects
        
        :param xyz: positions of all simulation particles
        :type xyz: (N2,3) floats, N2 >> N1
        :param masses: masses of all simulation particles
        :type masses: (N2,) floats
        :param idx_cat: each entry of the list is a list containing indices of particles belonging to an object
        :type idx_cat: list of length N1
        :param MIN_NUMBER_PTCS: minimum number of particles for object to qualify for morphology calculation
        :type MIN_NUMBER_PTCS: int
        :return centers, m: centers and masses
        :rtype: (N,3) and (N,) floats"""
        centers, m = calcMassesCenters(xyz.base, masses.base, idx_cat, MIN_NUMBER_PTCS, self.L_BOX, self.CENTER)
        return centers, m
    
    def getIdxCatLocalBase(self, float[:,:] xyz, float[:] masses, idx_cat, int MIN_NUMBER_PTCS, float D_LOGSTART, float D_LOGEND, int D_BINS, float M_TOL, int N_WALL, int N_MIN):
        """ Calculate index catalogue corresponding to objects whose local shapes can be determined
        
        :param xyz: positions of all simulation particles
        :type xyz: (N2,3) floats, N2 >> N1
        :param masses: masses of all simulation particles
        :type masses: (N2,) floats
        :param idx_cat: each entry of the list is a list containing indices of particles belonging to an object
        :type idx_cat: list of length N1
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
        :return idx_cat_local: each entry of the list is a list containing indices of particles belonging to an 
            object whose local shape calculation (i.e. shape profiles) converged, empty entry if not converged
        :rtype: list of length N1"""
        if rank == 0:
            d, q, s, minor, inter, major, halos_center, obj_m, succeeded = calcMorphLocal(xyz.base, masses.base, self.r200.base, idx_cat, self.L_BOX, MIN_NUMBER_PTCS, D_LOGSTART, D_LOGEND, D_BINS, M_TOL, N_WALL, N_MIN, self.CENTER)
            del d; del q; del s; del minor; del inter; del major; del halos_center; del obj_m
            print_status(rank, self.start_time, "Finished calcMorphLocal()")
        
            idx_cat_local = [[] for i in range(len(idx_cat))] # We are removing those halos whose R200 shell does not converge (including where R200 is not even available)
            for success in succeeded:
                if self.r200[success] != 0.0: 
                    idx_cat_local[success] = np.int32(idx_cat[success]).tolist()
            
            return idx_cat_local
        else:
            return None
        
    def getIdxCatGlobalBase(self, idx_cat, int MIN_NUMBER_PTCS):
        """ Calculate index catalogue corresponding to objects whose global shapes can be determined
        
        :param idx_cat: each entry of the list is a list containing indices of particles belonging to an object
        :type idx_cat: list of length N1
        :param MIN_NUMBER_PTCS: minimum number of particles for object to qualify for morphology calculation
        :type MIN_NUMBER_PTCS: int
        :return idx_cat_global: each entry of the list is a list containing indices of particles belonging to an 
            object whose global shape calculation converged, empty entry if not converged
        :rtype: list of length N1"""
        if rank == 0:
            
            idx_cat_global = [[] for i in range(len(idx_cat))] # We are removing those halos whose R200 shell does not converge (including where R200 is not even available)
            for obj in range(len(idx_cat_global)):
                if len(idx_cat[obj]) >= MIN_NUMBER_PTCS: 
                    idx_cat_global[obj] = np.int32(idx_cat[obj]).tolist()
            return idx_cat_global
        else:
            return None
        
    def getIdxCatVelLocalBase(self, float[:,:] xyz, float[:,:] velxyz, float[:] masses, idx_cat, int MIN_NUMBER_PTCS, float D_LOGSTART, float D_LOGEND, int D_BINS, float M_TOL, int N_WALL, int N_MIN):
        """ Calculate index catalogue corresponding to objects whose local shapes can be determined
        
        :param xyz: positions of all simulation particles
        :type xyz: (N2,3) floats, N2 >> N1
        :param velxyz: velocity array
        :type velxyz: (N2,3) floats
        :param masses: masses of all simulation particles
        :type masses: (N2,) floats
        :param idx_cat: each entry of the list is a list containing indices of particles belonging to an object
        :type idx_cat: list of length N1
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
        :return idx_cat_local: each entry of the list is a list containing indices of particles belonging to an 
            object whose local shape calculation (i.e. shape profiles) converged, empty entry if not converged
        :rtype: list of length N1"""
        if rank == 0:
            d, q, s, minor, inter, major, halos_center, obj_m, succeeded = calcMorphLocalVelDisp(xyz.base, velxyz.base, masses.base, self.r200.base, idx_cat, self.L_BOX, MIN_NUMBER_PTCS, D_LOGSTART, D_LOGEND, D_BINS, M_TOL, N_WALL, N_MIN, self.CENTER)
            del d; del q; del s; del minor; del inter; del major; del halos_center; del obj_m
            print_status(rank, self.start_time, "Finished calcMorphLocalVelDisp()")
            
            idx_cat_local = [[] for i in range(len(idx_cat))] # We are removing those halos whose R200 shell does not converge (including where R200 is not even available)
            for success in succeeded:
                if self.r200[success] != 0.0: 
                    idx_cat_local[success] = np.int32(idx_cat[success]).tolist()
            
            return idx_cat_local
        else:
            return None
        
    def getIdxCatVelGlobalBase(self, idx_cat, int MIN_NUMBER_PTCS):
        """ Calculate index catalogue corresponding to objects whose global velocity shapes can be determined
        
        :param idx_cat: each entry of the list is a list containing indices of particles belonging to an object
        :type idx_cat: list of length N1
        :param MIN_NUMBER_PTCS: minimum number of particles for object to qualify for morphology calculation
        :type MIN_NUMBER_PTCS: int
        :return idx_cat_global: each entry of the list is a list containing indices of particles belonging to an 
            object whose global velocity shape calculation converged, empty entry if not converged
        :rtype: list of length N1"""
        if rank == 0:
            
            idx_cat_global = [[] for i in range(len(idx_cat))] # We are removing those halos whose R200 shell does not converge (including where R200 is not even available)
            for obj in range(len(idx_cat_global)):
                if len(idx_cat[obj]) >= MIN_NUMBER_PTCS: 
                    idx_cat_global[obj] = np.int32(idx_cat[obj]).tolist()
            return idx_cat_global
        else:
            return None
    
    def getShapeCatLocalBase(self, float[:,:] xyz, float[:] masses, idx_cat, int MIN_NUMBER_PTCS, float D_LOGSTART, float D_LOGEND, int D_BINS, float M_TOL, int N_WALL, int N_MIN):
        """ Get all relevant local shape data
        
        :param xyz: positions of all simulation particles
        :type xyz: (N2,3) floats, N2 >> N1
        :param masses: masses of all simulation particles
        :type masses: (N2,) floats
        :param idx_cat: each entry of the list is a list containing indices of particles belonging to an object
        :type idx_cat: list of length N1
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
        :return: d, q, s, minor, inter, major, obj_center, obj_m, succeeded (list of indices of converged objects)
        :rtype: 3 x (number_of_objs, D_BINS+1) float arrays, 
            3 x (number_of_objs, D_BINS+1, 3) float arrays, 
            (number_of_objs,3) float array, (number_of_objs,) float array,
            list of length N3
        """
        if rank == 0:
            d, q, s, minor, inter, major, obj_center, obj_m, succeeded = calcMorphLocal(xyz.base, masses.base, self.r200.base, idx_cat, self.L_BOX, MIN_NUMBER_PTCS, D_LOGSTART, D_LOGEND, D_BINS, M_TOL, N_WALL, N_MIN, self.CENTER)
            print_status(rank, self.start_time, "Finished calcMorphLocal()")
        
            if succeeded == []:
                minor = np.array([])
                inter = np.array([])
                major = np.array([])
            
            return d, q, s, minor, inter, major, obj_center, obj_m, succeeded
        else:
            return None, None, None, None, None, None, None, None, None
    
    def getShapeCatGlobalBase(self, float[:,:] xyz, float[:] masses, idx_cat, int MIN_NUMBER_PTCS, float M_TOL, int N_WALL, int N_MIN):
        """ Get all relevant global shape data
        
        :param xyz: positions of all simulation particles
        :type xyz: (N2,3) floats, N2 >> N1
        :param masses: masses of all simulation particles
        :type masses: (N2,) floats
        :param idx_cat: each entry of the list is a list containing indices of particles belonging to an object
        :type idx_cat: list of length N1
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
        :return: d, q, s, minor, inter, major, obj_center, obj_m
        :rtype: 3 x (number_of_objs, D_BINS+1) float arrays, 
            3 x (number_of_objs, D_BINS+1, 3) float arrays, 
            (number_of_objs,3) float array, (number_of_objs,) float array
        """
        if rank == 0:
            d, q, s, minor, inter, major, obj_center, obj_m = calcMorphGlobal(xyz.base, masses.base, self.r200.base, idx_cat, self.L_BOX, MIN_NUMBER_PTCS, M_TOL, N_WALL, N_MIN, self.CENTER, self.SAFE)
            print_status(rank, self.start_time, "Finished calcMorphGlobal()")
        
            if d.shape[0] != 0:
                d = np.reshape(d, (d.shape[0], 1)) # Has shape (number_of_halos, 1)
                q = np.reshape(q, (q.shape[0], 1)) # Has shape (number_of_halos, 1)
                s = np.reshape(s, (s.shape[0], 1)) # Has shape (number_of_halos, 1)
                minor = minor.reshape((d.shape[0],d.shape[1],3)) # Has shape (number_of_halos, 1, 3)
                inter = inter.reshape((d.shape[0],d.shape[1],3)) # Has shape (number_of_halos, 1, 3)
                major = major.reshape((d.shape[0],d.shape[1],3)) # Has shape (number_of_halos, 1, 3)
            else:
                minor = np.array([])
                inter = np.array([])
                major = np.array([])
            return d, q, s, minor, inter, major, obj_center, obj_m
        else:
            return None, None, None, None, None, None, None, None
        
    def getShapeCatVelLocalBase(self, float[:,:] xyz, float[:,:] velxyz, float[:] masses, idx_cat, int MIN_NUMBER_PTCS, float D_LOGSTART, float D_LOGEND, int D_BINS, float M_TOL, int N_WALL, int N_MIN):
        """ Get all relevant local velocity shape data
        
        :param xyz: positions of all simulation particles
        :type xyz: (N2,3) floats, N2 >> N1
        :param velxyz: velocity array
        :type velxyz: (N2,3) floats
        :param masses: masses of all simulation particles
        :type masses: (N2,) floats
        :param idx_cat: each entry of the list is a list containing indices of particles belonging to an object
        :type idx_cat: list of length N1
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
        :return: d, q, s, minor, inter, major, obj_center, obj_m, succeeded (list of indices of converged objects)
        :rtype: 3 x (number_of_objs, D_BINS+1) float arrays, 
            3 x (number_of_objs, D_BINS+1, 3) float arrays, 
            (number_of_objs,3) float array, (number_of_objs,) float array,
            list of length N3
        """
        if rank == 0:
            d, q, s, minor, inter, major, obj_center, obj_m, succeeded = calcMorphLocalVelDisp(xyz.base, velxyz.base, masses.base, self.r200.base, idx_cat, self.L_BOX, MIN_NUMBER_PTCS, D_LOGSTART, D_LOGEND, D_BINS, M_TOL, N_WALL, N_MIN, self.CENTER)
            print_status(rank, self.start_time, "Finished calcMorphLocalVelDisp()")
        
            if succeeded == []:
                minor = np.array([])
                inter = np.array([])
                major = np.array([])
            
            return d, q, s, minor, inter, major, obj_center, obj_m, succeeded
        else:
            return None, None, None, None, None, None, None, None, None
    
    def getShapeCatVelGlobalBase(self, float[:,:] xyz, float[:,:] velxyz, float[:] masses, idx_cat, int MIN_NUMBER_PTCS, float M_TOL, int N_WALL, int N_MIN):
        """ Get all relevant global velocity shape data
        
        :param xyz: positions of all simulation particles
        :type xyz: (N2,3) floats, N2 >> N1
        :param velxyz: velocity array
        :type velxyz: (N2,3) floats
        :param masses: masses of all simulation particles
        :type masses: (N2,) floats
        :param idx_cat: each entry of the list is a list containing indices of particles belonging to an object
        :type idx_cat: list of length N1
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
        :return: d, q, s, minor, inter, major, obj_center, obj_m
        :rtype: 3 x (number_of_objs, D_BINS+1) float arrays, 
            3 x (number_of_objs, D_BINS+1, 3) float arrays, 
            (number_of_objs,3) float array, (number_of_objs,) float array
        """
        if rank == 0:
            d, q, s, minor, inter, major, obj_center, obj_m = calcMorphGlobalVelDisp(xyz.base, velxyz.base, masses.base, self.r200.base, idx_cat, self.L_BOX, MIN_NUMBER_PTCS, M_TOL, N_WALL, N_MIN, self.CENTER, self.SAFE)
            print_status(rank, self.start_time, "Finished calcMorphGlobalVelDisp")
            
            if d.shape[0] != 0:
                d = np.reshape(d, (d.shape[0], 1)) # Has shape (number_of_halos, 1)
                q = np.reshape(q, (q.shape[0], 1)) # Has shape (number_of_halos, 1)
                s = np.reshape(s, (s.shape[0], 1)) # Has shape (number_of_halos, 1)
                minor = minor.reshape((d.shape[0],d.shape[1],3)) # Has shape (number_of_halos, 1, 3)
                inter = inter.reshape((d.shape[0],d.shape[1],3)) # Has shape (number_of_halos, 1, 3)
                major = major.reshape((d.shape[0],d.shape[1],3)) # Has shape (number_of_halos, 1, 3)
            else:
                minor = np.array([])
                inter = np.array([])
                major = np.array([])
            return d, q, s, minor, inter, major, obj_center, obj_m
        else:
            return None, None, None, None, None, None, None, None
    
    def dumpShapeCatLocalBase(self, float[:,:] xyz, float[:] masses, idx_cat, int MIN_NUMBER_PTCS, float D_LOGSTART, float D_LOGEND, int D_BINS, float M_TOL, int N_WALL, int N_MIN, str CAT_DEST, str suffix):
        """ Dumps all relevant local shape data into ``CAT_DEST``
        
        :param xyz: positions of all simulation particles
        :type xyz: (N2,3) floats, N2 >> N1
        :param masses: masses of all simulation particles
        :type masses: (N2,) floats
        :param idx_cat: each entry of the list is a list containing indices of particles belonging to an object
        :type idx_cat: list of length N1
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
        :param CAT_DEST: catalogue destination
        :type CAT_DEST: string
        :param suffix: suffix for file names
        :type suffix: string
        """
        if rank == 0:
            d, q, s, minor, inter, major, obj_center, obj_m, succeeded = self.getShapeCatLocalBase(xyz.base, masses.base, idx_cat, MIN_NUMBER_PTCS, D_LOGSTART, D_LOGEND, D_BINS, M_TOL, N_WALL, N_MIN)
            
            if succeeded != []:
                minor = minor.reshape(minor.shape[0], -1)
                inter = inter.reshape(inter.shape[0], -1)
                major = major.reshape(major.shape[0], -1)
            else:
                minor = np.array([])
                inter = np.array([])
                major = np.array([])
            
            idx_cat_local = [[] for i in range(len(idx_cat))] # We are removing those halos whose R200 shell does not converge (including where R200 is not even available)
            for success in succeeded:
                if self.r200[success] != 0.0: 
                    idx_cat_local[success] = np.int32(idx_cat[success]).tolist()
            with open('{0}/idx_cat_local{1}{2}.txt'.format(CAT_DEST, suffix, self.SNAP), 'w') as filehandle:
                json.dump(idx_cat_local, filehandle)
            np.savetxt('{0}/d_local{1}{2}.txt'.format(CAT_DEST, suffix, self.SNAP), d, fmt='%1.7e')
            np.savetxt('{0}/q_local{1}{2}.txt'.format(CAT_DEST, suffix, self.SNAP), q, fmt='%1.7e')
            np.savetxt('{0}/s_local{1}{2}.txt'.format(CAT_DEST, suffix, self.SNAP), s, fmt='%1.7e')
            np.savetxt('{0}/minor_local{1}{2}.txt'.format(CAT_DEST, suffix, self.SNAP), minor, fmt='%1.7e')
            np.savetxt('{0}/inter_local{1}{2}.txt'.format(CAT_DEST, suffix, self.SNAP), inter, fmt='%1.7e')
            np.savetxt('{0}/major_local{1}{2}.txt'.format(CAT_DEST, suffix, self.SNAP), major, fmt='%1.7e')
            np.savetxt('{0}/m_local{1}{2}.txt'.format(CAT_DEST, suffix, self.SNAP), obj_m, fmt='%1.7e')
            np.savetxt('{0}/centers_local{1}{2}.txt'.format(CAT_DEST, suffix, self.SNAP), obj_center, fmt='%1.7e')
            del idx_cat_local; del d; del q; del s; del minor; del inter; del major; del obj_center; del obj_m; del succeeded
        
    def dumpShapeCatGlobalBase(self, float[:,:] xyz, float[:] masses, idx_cat, int MIN_NUMBER_PTCS, float M_TOL, int N_WALL, int N_MIN, str CAT_DEST, str suffix):
        """ Dumps all relevant global shape data into ``CAT_DEST``
        
        :param xyz: positions of all simulation particles
        :type xyz: (N2,3) floats, N2 >> N1
        :param masses: masses of all simulation particles
        :type masses: (N2,) floats
        :param idx_cat: each entry of the list is a list containing indices of particles belonging to an object
        :type idx_cat: list of length N1
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
        :param CAT_DEST: catalogue destination
        :type CAT_DEST: string
        :param suffix: suffix for file names
        :type suffix: string
        """
        if rank == 0:
            idx_cat_global = self.getIdxCatGlobalBase(idx_cat, MIN_NUMBER_PTCS)
            d, q, s, minor, inter, major, obj_center, obj_m = self.getShapeCatGlobalBase(xyz.base, masses.base, idx_cat, MIN_NUMBER_PTCS, M_TOL, N_WALL, N_MIN)
            
            if d.shape[0] != 0:
                minor = minor.reshape(minor.shape[0], -1)
                inter = inter.reshape(inter.shape[0], -1)
                major = major.reshape(major.shape[0], -1)
            else:
                minor = np.array([])
                inter = np.array([])
                major = np.array([])
            
            with open('{0}/idx_cat_global{1}{2}.txt'.format(CAT_DEST, suffix, self.SNAP), 'w') as filehandle:
                json.dump(idx_cat_global, filehandle)
            np.savetxt('{0}/d_global{1}{2}.txt'.format(CAT_DEST, suffix, self.SNAP), d, fmt='%1.7e')
            np.savetxt('{0}/q_global{1}{2}.txt'.format(CAT_DEST, suffix, self.SNAP), q, fmt='%1.7e')
            np.savetxt('{0}/s_global{1}{2}.txt'.format(CAT_DEST, suffix, self.SNAP), s, fmt='%1.7e')
            np.savetxt('{0}/minor_global{1}{2}.txt'.format(CAT_DEST, suffix, self.SNAP), minor, fmt='%1.7e')
            np.savetxt('{0}/inter_global{1}{2}.txt'.format(CAT_DEST, suffix, self.SNAP), inter, fmt='%1.7e')
            np.savetxt('{0}/major_global{1}{2}.txt'.format(CAT_DEST, suffix, self.SNAP), major, fmt='%1.7e')
            np.savetxt('{0}/m_global{1}{2}.txt'.format(CAT_DEST, suffix, self.SNAP), obj_m, fmt='%1.7e')
            np.savetxt('{0}/centers_global{1}{2}.txt'.format(CAT_DEST, suffix, self.SNAP), obj_center, fmt='%1.7e')
            del idx_cat_global; del d; del q; del s; del minor; del inter; del major; del obj_center; del obj_m
    
    def dumpShapeCatVelLocalBase(self, float[:,:] xyz, float[:,:] velxyz, float[:] masses, idx_cat, int MIN_NUMBER_PTCS, float D_LOGSTART, float D_LOGEND, int D_BINS, float M_TOL, int N_WALL, int N_MIN, str CAT_DEST, str suffix):
        """ Dumps all relevant local velocity shape data into ``CAT_DEST``
        
        :param xyz: positions of all simulation particles
        :type xyz: (N2,3) floats, N2 >> N1
        :param velxyz: velocity array
        :type velxyz: (N2,3) floats
        :param masses: masses of all simulation particles
        :type masses: (N2,) floats
        :param idx_cat: each entry of the list is a list containing indices of particles belonging to an object
        :type idx_cat: list of length N1
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
        :param CAT_DEST: catalogue destination
        :type CAT_DEST: string
        :param suffix: suffix for file names
        :type suffix: string
        """
        if rank == 0:
            d, q, s, minor, inter, major, obj_center, obj_m, succeeded = self.getShapeCatVelLocalBase(xyz.base, velxyz.base, masses.base, idx_cat, MIN_NUMBER_PTCS, D_LOGSTART, D_LOGEND, D_BINS, M_TOL, N_WALL, N_MIN)
            
            if succeeded != []:
                minor = minor.reshape(minor.shape[0], -1)
                inter = inter.reshape(inter.shape[0], -1)
                major = major.reshape(major.shape[0], -1)
            else:
                minor = np.array([])
                inter = np.array([])
                major = np.array([])
            
            idx_cat_local = [[] for i in range(len(idx_cat))] # We are removing those halos whose R200 shell does not converge (including where R200 is not even available)
            for success in succeeded:
                if self.r200[success] != 0.0: 
                    idx_cat_local[success] = np.int32(idx_cat[success]).tolist()
            with open('{0}/idx_cat_local{1}{2}.txt'.format(CAT_DEST, suffix, self.SNAP), 'w') as filehandle:
                json.dump(idx_cat_local, filehandle)
            np.savetxt('{0}/d_local{1}{2}.txt'.format(CAT_DEST, suffix, self.SNAP), d, fmt='%1.7e')
            np.savetxt('{0}/q_local{1}{2}.txt'.format(CAT_DEST, suffix, self.SNAP), q, fmt='%1.7e')
            np.savetxt('{0}/s_local{1}{2}.txt'.format(CAT_DEST, suffix, self.SNAP), s, fmt='%1.7e')
            np.savetxt('{0}/minor_local{1}{2}.txt'.format(CAT_DEST, suffix, self.SNAP), minor, fmt='%1.7e')
            np.savetxt('{0}/inter_local{1}{2}.txt'.format(CAT_DEST, suffix, self.SNAP), inter, fmt='%1.7e')
            np.savetxt('{0}/major_local{1}{2}.txt'.format(CAT_DEST, suffix, self.SNAP), major, fmt='%1.7e')
            np.savetxt('{0}/m_local{1}{2}.txt'.format(CAT_DEST, suffix, self.SNAP), obj_m, fmt='%1.7e')
            np.savetxt('{0}/centers_local{1}{2}.txt'.format(CAT_DEST, suffix, self.SNAP), obj_center, fmt='%1.7e')
            del idx_cat_local; del d; del q; del s; del minor; del inter; del major; del obj_center; del obj_m; del succeeded
    
    def dumpShapeCatVelGlobalBase(self, float[:,:] xyz, float[:,:] velxyz, float[:] masses, idx_cat, int MIN_NUMBER_PTCS, float M_TOL, int N_WALL, int N_MIN, str CAT_DEST, str suffix):
        """ Dumps all relevant global velocity shape data into ``CAT_DEST``
        
        :param xyz: positions of all simulation particles
        :type xyz: (N2,3) floats, N2 >> N1
        :param velxyz: velocity array
        :type velxyz: (N2,3) floats
        :param masses: masses of all simulation particles
        :type masses: (N2,) floats
        :param idx_cat: each entry of the list is a list containing indices of particles belonging to an object
        :type idx_cat: list of length N1
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
        :param CAT_DEST: catalogue destination
        :type CAT_DEST: string
        :param suffix: suffix for file names
        :type suffix: string
        """
        
        if rank == 0:
            idx_cat_global = self.getIdxCatGlobalBase(idx_cat, MIN_NUMBER_PTCS)
            d, q, s, minor, inter, major, obj_center, obj_m = self.getShapeCatGlobalBase(xyz.base, velxyz.base, masses.base, idx_cat, MIN_NUMBER_PTCS, M_TOL, N_WALL, N_MIN)
            
            if d.shape[0] != 0:
                minor = minor.reshape(minor.shape[0], -1)
                inter = inter.reshape(inter.shape[0], -1)
                major = major.reshape(major.shape[0], -1)
            else:
                minor = np.array([])
                inter = np.array([])
                major = np.array([])
            
            with open('{0}/idx_cat_global{1}{2}.txt'.format(CAT_DEST, suffix, self.SNAP), 'w') as filehandle:
                json.dump(idx_cat_global, filehandle)
            np.savetxt('{0}/d_global{1}{2}.txt'.format(CAT_DEST, suffix, self.SNAP), d, fmt='%1.7e')
            np.savetxt('{0}/q_global{1}{2}.txt'.format(CAT_DEST, suffix, self.SNAP), q, fmt='%1.7e')
            np.savetxt('{0}/s_global{1}{2}.txt'.format(CAT_DEST, suffix, self.SNAP), s, fmt='%1.7e')
            np.savetxt('{0}/minor_global{1}{2}.txt'.format(CAT_DEST, suffix, self.SNAP), minor, fmt='%1.7e')
            np.savetxt('{0}/inter_global{1}{2}.txt'.format(CAT_DEST, suffix, self.SNAP), inter, fmt='%1.7e')
            np.savetxt('{0}/major_global{1}{2}.txt'.format(CAT_DEST, suffix, self.SNAP), major, fmt='%1.7e')
            np.savetxt('{0}/m_global{1}{2}.txt'.format(CAT_DEST, suffix, self.SNAP), obj_m, fmt='%1.7e')
            np.savetxt('{0}/centers_global{1}{2}.txt'.format(CAT_DEST, suffix, self.SNAP), obj_center, fmt='%1.7e')
            del idx_cat_global; del d; del q; del s; del minor; del inter; del major; del obj_center; del obj_m
    
    def plotShapeProfsBase(self, float[:,:] xyz, float[:] masses, idx_cat, int MIN_NUMBER_PTCS, float D_LOGSTART, float D_LOGEND, int D_BINS, float M_TOL, int N_WALL, int N_MIN, str VIZ_DEST, str suffix = ''):
        """ Draws shape profiles, also mass bin-decomposed ones
        
        :param xyz: positions of all simulation particles
        :type xyz: (N2,3) floats, N2 >> N1
        :param masses: masses of all simulation particles
        :type masses: (N2,) floats
        :param idx_cat: each entry of the list is a list containing indices of particles belonging to an object
        :type idx_cat: list of length N1
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
        :param VIZ_DEST: visualization folder
        :type VIZ_DEST: string
        :param suffix: suffix for file names
        :type suffix: string
        """
                
        if rank == 0:
            d, q, s, minor, inter, major, obj_centers, obj_masses, succeeded = self.getShapeCatLocalBase(xyz.base, masses.base, idx_cat, MIN_NUMBER_PTCS, D_LOGSTART, D_LOGEND, D_BINS, M_TOL, N_WALL, N_MIN)
            getShapeProfs(VIZ_DEST, self.SNAP, D_LOGSTART, D_LOGEND, D_BINS, self.start_time, obj_masses, obj_centers, d, q, s, major, self.MASS_UNIT, suffix = suffix)
            del d; del q; del s; del minor; del inter; del major; del obj_centers; del obj_masses; del succeeded
            
    def plotLocalTHistBase(self, float[:,:] xyz, float[:] masses, idx_cat, int MIN_NUMBER_PTCS, float D_LOGSTART, float D_LOGEND, int D_BINS, float M_TOL, int N_WALL, int N_MIN, str VIZ_DEST, int HIST_NB_BINS, float frac_r200, str suffix = ''):
        """ Plot a local-shape triaxiality histogram at a specified ellipsoidal depth of ``frac_r200``
        
        :param xyz: positions of all simulation particles
        :type xyz: (N2,3) floats, N2 >> N1
        :param masses: masses of all simulation particles
        :type masses: (N2,) floats
        :param idx_cat: each entry of the list is a list containing indices of particles belonging to an object
        :type idx_cat: list of length N1
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
        :param VIZ_DEST: visualization folder
        :type VIZ_DEST: string
        :param HIST_NB_BINS: number of histogram bins
        :type HIST_NB_BINS: int
        :param frac_r200: depth of objects to plot triaxiality, in units of R200
        :type frac_r200: float
        :param suffix: suffix for file names
        :type suffix: string
        """
                
        if rank == 0:
            d, q, s, minor, inter, major, obj_centers, obj_masses, succeeded = self.getShapeCatLocalBase(xyz.base, masses.base, idx_cat, MIN_NUMBER_PTCS, D_LOGSTART, D_LOGEND, D_BINS, M_TOL, N_WALL, N_MIN)
            getLocalTHist(VIZ_DEST, self.SNAP, D_LOGSTART, D_LOGEND, D_BINS, self.start_time, obj_masses, obj_centers, d, q, s, major, HIST_NB_BINS, frac_r200, self.MASS_UNIT, suffix = suffix)
            del d; del q; del s; del minor; del inter; del major; del obj_centers; del obj_masses; del succeeded
    
    def plotGlobalTHistBase(self, float[:,:] xyz, float[:] masses, idx_cat, int MIN_NUMBER_PTCS, float M_TOL, int N_WALL, int N_MIN, str VIZ_DEST, int HIST_NB_BINS, str suffix = ''):
        """ Plot a global-shape triaxiality histogram
                
        :param xyz: positions of all simulation particles
        :type xyz: (N2,3) floats, N2 >> N1
        :param masses: masses of all simulation particles
        :type masses: (N2,) floats
        :param idx_cat: each entry of the list is a list containing indices of particles belonging to an object
        :type idx_cat: list of length N1
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
        :param VIZ_DEST: visualization folder
        :type VIZ_DEST: string
        :param HIST_NB_BINS: number of histogram bins
        :type HIST_NB_BINS: int
        :param suffix: suffix for file names
        :type suffix: string
        """
                
        if rank == 0:
            d, q, s, minor, inter, major, obj_centers, obj_masses = self.getShapeCatGlobalBase(xyz.base, masses.base, idx_cat, MIN_NUMBER_PTCS, M_TOL, N_WALL, N_MIN)
            getGlobalTHist(VIZ_DEST, self.SNAP, self.start_time, obj_masses, obj_centers, d, q, s, major, HIST_NB_BINS, self.MASS_UNIT, suffix = suffix)
            del d; del q; del s; del minor; del inter; del major; del obj_centers; del obj_masses
    
    def getDensProfsBestFitsBase(self, float[:,:] dens_profs, float[:] ROverR200, idx_cat, float[:] r200s, int MIN_NUMBER_PTCS, method = 'einasto'):
        """ Get best-fit results for density profile fitting
        
        :param dens_profs: density profiles to be fit, in units of M_sun*h^2/(Mpc)**3
        :type dens_profs: (N3, r_res) floats
        :param ROverR200: normalized radii at which ``dens_profs`` are defined
        :type ROverR200: (r_res,) floats
        :param idx_cat: each entry of the list is a list containing indices of particles belonging to an object
        :type idx_cat: list of length N1
        :param r200s: R_200 radii of the parent halos
        :type r200s: (N1,) floats
        :param MIN_NUMBER_PTCS: minimum number of particles for object to qualify for morphology calculation
        :type MIN_NUMBER_PTCS: int
        :param method: string describing density profile model assumed for fitting
        :type method: string, either `einasto`, `alpha_beta_gamma`, `hernquist`, `nfw`
        :return: best-fits for each object, and normalized radii used to calculate best-fits
        :rtype: (N3, n) floats, where n is the number of free parameters in the model ``method``,
            and (r_res,) floats"""
        
        if rank == 0:
            r200s = np.float32([r200s[i] for i in range(len(idx_cat)) if len(idx_cat[i]) >= MIN_NUMBER_PTCS])
            best_fits = fitDensProfHelper(dens_profs.base, ROverR200.base, r200s.base, method)
            del r200s
            return best_fits
        else:
            return None
        
    def getConcentrationsBase(self, float[:,:] dens_profs, float[:] ROverR200, idx_cat, float[:] r200s, int MIN_NUMBER_PTCS, method = 'einasto'):
        """ Get best-fit concentration values of objects from density profile fitting
        
        :param dens_profs: density profiles to be fit, in units of M_sun*h^2/(Mpc)**3
        :type dens_profs: (N3, r_res) floats
        :param ROverR200: normalized radii at which ``dens_profs`` are defined
        :type ROverR200: (r_res,) floats
        :param idx_cat: each entry of the list is a list containing indices of particles belonging to an object
        :type idx_cat: list of length N1
        :param r200s: R_200 radii of the parent halos
        :type r200s: (N1,) floats
        :param MIN_NUMBER_PTCS: minimum number of particles for object to qualify for morphology calculation
        :type MIN_NUMBER_PTCS: int
        :param method: string describing density profile model assumed for fitting
        :type method: string, either `einasto`, `alpha_beta_gamma`, `hernquist`, `nfw`
        :return: best-fit concentration for each object
        :rtype: (N3,) floats"""
                
        if rank == 0:
            r200s = np.float32([r200s[i] for i in range(len(idx_cat)) if len(idx_cat[i]) >= MIN_NUMBER_PTCS])
            best_fits = fitDensProfHelper(dens_profs.base, ROverR200.base, r200s.base, method)
            if method == 'einasto':
                cs = r200s/best_fits[:,-2]
            else:
                cs = r200s/best_fits[:,-1]
            del r200s
            return cs
        else:
            return None
        
    def getDensProfsDirectBinningBase(self, float[:,:] xyz, float[:] masses, idx_cat, int MIN_NUMBER_PTCS, float[:] ROverR200):
        """ Get direct-binning-based density profiles
        
        :param xyz: positions of all simulation particles
        :type xyz: (N2,3) floats, N2 >> N1
        :param masses: masses of all simulation particles
        :type masses: (N2,) floats
        :param idx_cat: each entry of the list is a list containing indices of particles belonging to an object
        :type idx_cat: list of length N1
        :param MIN_NUMBER_PTCS: minimum number of particles for object to qualify for morphology calculation
        :type MIN_NUMBER_PTCS: int
        :param ROverR200: normalized radii at which ``dens_profs`` are defined
        :type ROverR200: (r_res,) floats
        :return: density profiles
        :rtype: (N2, r_res) floats"""
                
        if rank == 0:
            obj_keep = np.int32([1 if len(x) >= MIN_NUMBER_PTCS else 0 for x in idx_cat])
            dens_profs = calcDensProfsDirectBinning(xyz.base, obj_keep, masses.base, self.r200.base, ROverR200.base, idx_cat, MIN_NUMBER_PTCS, self.L_BOX, self.CENTER)
            del obj_keep
            return dens_profs*self.MASS_UNIT
        else:
            return None
        
    def getDensProfsKernelBasedBase(self, float[:,:] xyz, float[:] masses, idx_cat, int MIN_NUMBER_PTCS, float[:] ROverR200):
        """ Get kernel-based density profiles
        
        :param xyz: positions of all simulation particles
        :type xyz: (N2,3) floats, N2 >> N1
        :param masses: masses of all simulation particles
        :type masses: (N2,) floats
        :param idx_cat: each entry of the list is a list containing indices of particles belonging to an object
        :type idx_cat: list of length N1
        :param MIN_NUMBER_PTCS: minimum number of particles for object to qualify for morphology calculation
        :type MIN_NUMBER_PTCS: int
        :param ROverR200: normalized radii at which ``dens_profs`` are defined
        :type ROverR200: (r_res,) floats
        :return: density profiles
        :rtype: (N2, r_res) floats"""
                
        if rank == 0:
            obj_keep = np.int32([1 if len(x) >= MIN_NUMBER_PTCS else 0 for x in idx_cat])
            dens_profs = calcDensProfsKernelBased(xyz.base, obj_keep, masses.base, self.r200.base, ROverR200.base, idx_cat, MIN_NUMBER_PTCS, self.L_BOX, self.CENTER)
            del obj_keep
            return dens_profs*self.MASS_UNIT
        else:
            return None
        
    def getObjInfoLocalBase(self, float[:,:] xyz, float[:] masses, idx_cat, int MIN_NUMBER_PTCS, float D_LOGSTART, float D_LOGEND, int D_BINS, float M_TOL, int N_WALL, int N_MIN, obj_type):
        """ Print basic info about the objects used for local shape estimation such as number of converged objects
        
        :param xyz: positions of all simulation particles
        :type xyz: (N2,3) floats, N2 >> N1
        :param masses: masses of all simulation particles
        :type masses: (N2,) floats
        :param idx_cat: each entry of the list is a list containing indices of particles belonging to an object
        :type idx_cat: list of length N1
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
        :param obj_type: either 'dm', 'gx' or 'unspecified', depending on what catalogue we are looking at
        :type obj_type: string"""
        obj_pass = np.sum(np.int32([1 if len(x) >= MIN_NUMBER_PTCS else 0 for x in idx_cat]))
        idx_cat_local = self.getIdxCatLocalBase(xyz.base, masses.base, idx_cat, MIN_NUMBER_PTCS, D_LOGSTART, D_LOGEND, D_BINS, M_TOL, N_WALL, N_MIN)
        print_status(rank,self.start_time,'Snap {0}'.format(self.SNAP))
        print_status(rank,self.start_time,'Number of non-empty objects: {0}'.format(np.sum(np.int32([1 if x != [] else 0 for x in idx_cat]))))
        print_status(rank,self.start_time,'Object type: {0}'.format(obj_type))
        print_status(rank,self.start_time,'Number of objects with sufficient resolution: {0}'.format(np.sum(np.int32([1 if len(x) >= MIN_NUMBER_PTCS else 0 for x in idx_cat]))))
        print_status(rank,self.start_time,'Number of objects for which shape profiles were determined: {0}'.format(np.sum(np.int32([1 if x != [] else 0 for x in idx_cat_local]))))
        del obj_pass; del idx_cat_local

    def getObjInfoGlobalBase(self, idx_cat, int MIN_NUMBER_PTCS, obj_type):
        """ Print basic info about the objects used for global shape estimation such as number of converged objects
        
        :param idx_cat: each entry of the list is a list containing indices of particles belonging to an object
        :type idx_cat: list of length N1
        :param MIN_NUMBER_PTCS: minimum number of particles for object to qualify for morphology calculation
        :type MIN_NUMBER_PTCS: int
        :param obj_type: either 'dm', 'gx' or 'unspecified', depending on what catalogue we are looking at
        :type obj_type: string"""
        obj_pass = np.sum(np.int32([1 if len(x) >= MIN_NUMBER_PTCS else 0 for x in idx_cat]))
        idx_cat_global = self.getIdxCatGlobalBase(idx_cat, MIN_NUMBER_PTCS)
        print_status(rank,self.start_time,'Snap {0}'.format(self.SNAP))
        print_status(rank,self.start_time,'Number of non-empty objects: {0}'.format(np.sum(np.int32([1 if x != [] else 0 for x in idx_cat]))))
        print_status(rank,self.start_time,'Object type: {0}'.format(obj_type))
        print_status(rank,self.start_time,'Number of objects with sufficient resolution: {0}'.format(np.sum(np.int32([1 if len(x) >= MIN_NUMBER_PTCS else 0 for x in idx_cat]))))
        print_status(rank,self.start_time,'Equal to the number of objects for which global shapes were determined: {0}'.format(np.sum(np.int32([1 if x != [] else 0 for x in idx_cat_global]))))
        del obj_pass; del idx_cat_global
    
    def getObjInfoVelLocalBase(self, float[:,:] xyz, float[:,:] velxyz, float[:] masses, idx_cat, int MIN_NUMBER_PTCS, float D_LOGSTART, float D_LOGEND, int D_BINS, float M_TOL, int N_WALL, int N_MIN, obj_type):
        """ Print basic info about the objects used for local velocity shape estimation such as number of converged objects
        
        :param xyz: positions of all simulation particles
        :type xyz: (N2,3) floats, N2 >> N1
        :param velxyz: velocity array
        :type velxyz: (N2,3) floats
        :param masses: masses of all simulation particles
        :type masses: (N2,) floats
        :param idx_cat: each entry of the list is a list containing indices of particles belonging to an object
        :type idx_cat: list of length N1
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
        :param obj_type: either 'dm', 'gx' or 'unspecified', depending on what catalogue we are looking at
        :type obj_type: string"""
        obj_pass = np.sum(np.int32([1 if len(x) >= MIN_NUMBER_PTCS else 0 for x in idx_cat]))
        idx_cat_local = self.getIdxCatVelLocalBase(xyz.base, velxyz.base, masses.base, idx_cat, MIN_NUMBER_PTCS, D_LOGSTART, D_LOGEND, D_BINS, M_TOL, N_WALL, N_MIN)
        print_status(rank,self.start_time,'Snap {0}'.format(self.SNAP))
        print_status(rank,self.start_time,'Number of non-empty objects: {0}'.format(np.sum(np.int32([1 if x != [] else 0 for x in idx_cat]))))
        print_status(rank,self.start_time,'Object type: {0}'.format(obj_type))
        print_status(rank,self.start_time,'Number of objects with sufficient resolution: {0}'.format(np.sum(np.int32([1 if len(x) >= MIN_NUMBER_PTCS else 0 for x in idx_cat]))))
        print_status(rank,self.start_time,'Number of objects for which velocity shape profiles were determined: {0}'.format(np.sum(np.int32([1 if x != [] else 0 for x in idx_cat_local]))))
        del obj_pass; del idx_cat_local
    
    def getObjInfoVelGlobalBase(self, idx_cat, int MIN_NUMBER_PTCS, obj_type):
        """ Print basic info about the objects used for global velocity shape estimation such as number of converged objects
        
        :param idx_cat: each entry of the list is a list containing indices of particles belonging to an object
        :type idx_cat: list of length N1
        :param MIN_NUMBER_PTCS: minimum number of particles for object to qualify for morphology calculation
        :type MIN_NUMBER_PTCS: int
        :param obj_type: either 'dm', 'gx' or 'unspecified', depending on what catalogue we are looking at
        :type obj_type: string"""
        obj_pass = np.sum(np.int32([1 if len(x) >= MIN_NUMBER_PTCS else 0 for x in idx_cat]))
        idx_cat_global = self.getIdxCatGlobalBase(idx_cat, MIN_NUMBER_PTCS)
        print_status(rank,self.start_time,'Snap {0}'.format(self.SNAP))
        print_status(rank,self.start_time,'Number of non-empty objects: {0}'.format(np.sum(np.int32([1 if x != [] else 0 for x in idx_cat]))))
        print_status(rank,self.start_time,'Object type: {0}'.format(obj_type))
        print_status(rank,self.start_time,'Number of objects with sufficient resolution: {0}'.format(np.sum(np.int32([1 if len(x) >= MIN_NUMBER_PTCS else 0 for x in idx_cat]))))
        print_status(rank,self.start_time,'Equal to the number of objects for which velocity global shapes were determined: {0}'.format(np.sum(np.int32([1 if x != [] else 0 for x in idx_cat_global]))))
        del obj_pass; del idx_cat_global

cdef class DensProfs(CosmicBase):
    """ Class for density profile calculations
    
    Its public methods are ``getIdxCat()``, ``getMassesCenters()``, ``getDensProfsDirectBinning()``,
    ``getDensProfsKernelBased()``, ``getDensProfsBestFits()``, ``getConcentrations()``, 
    ``plotDensProfs()``."""
    
    cdef float[:,:] xyz
    cdef float[:] masses
    cdef object idx_cat
    
    def __init__(self, float[:,:] xyz, float[:] masses, idx_cat, float[:] r200, str SNAP, float L_BOX, int MIN_NUMBER_PTCS, str CENTER, double start_time):
        """
        :param xyz: positions of all simulation particles
        :type xyz: (N2,3) floats, N2 >> N1
        :param masses: masses of all simulation particles
        :type masses: (N2,) floats
        :param idx_cat: each entry of the list is a list containing indices of particles belonging to an object
        :type idx_cat: list of length N1
        :param r200: R_200 radii of the parent halos
        :type r200: (N1,) floats
        :param SNAP: snapshot identifier, e.g. '024'
        :type SNAP: string
        :param L_BOX: simulation box side length
        :type L_BOX: float, units: Mpc/h
        :param MIN_NUMBER_PTCS: minimum number of particles for object to qualify for morphology calculation
        :type MIN_NUMBER_PTCS: int
        :param CENTER: shape quantities will be calculated with respect to CENTER = 'mode' (point of highest density)
            or 'com' (center of mass) of each halo
        :type CENTER: str
        :param start_time: time of start of object initialization
        :type start_time: float"""
        super().__init__(SNAP, L_BOX, MIN_NUMBER_PTCS, CENTER, start_time)
        assert xyz.shape[0] == masses.shape[0], "xyz.shape[0] must be equal to masses.shape[0]"
        self.xyz = xyz.base
        self.masses = masses.base
        self.idx_cat = idx_cat
        self.r200 = r200.base
       
    def getIdxCat(self):
        """ Fetch catalogue
        
        :return cat: list of indices defining the objects
        :rtype: list of length N1, each consisting of a list of int indices"""
        print_status(rank,self.start_time,'Starting getIdxCat() with snap {0}'.format(self.SNAP))
        return self.idx_cat
    
    def getMassesCenters(self):
        """ Calculate total mass and centers of objects
        
        :return centers, m: centers and masses
        :rtype: (N,3) and (N,) floats"""
        print_status(rank,self.start_time,'Starting getMassesCenters() with snap {0}'.format(self.SNAP))
        if rank == 0:
            centers, ms = self.getMassesCentersBase(self.xyz.base, self.masses.base, self.idx_cat, self.MIN_NUMBER_PTCS)
            return centers, ms
        else:
            return None, None
    
    def getDensProfsDirectBinning(self, ROverR200):
        """ Get direct-binning-based density profiles
        
        :param ROverR200: normalized radii at which ``dens_profs`` are defined
        :type ROverR200: (r_res,) floats
        :return: density profiles
        :rtype: (N2, r_res) floats"""
        print_status(rank,self.start_time,'Starting getDensProfsDirectBinning() with snap {0}'.format(self.SNAP))
        if rank == 0:
            dens_profs = self.getDensProfsDirectBinningBase(self.xyz.base, self.masses.base, self.idx_cat, self.MIN_NUMBER_PTCS, np.float32(ROverR200))
            return dens_profs
        else:
            return None

    def getDensProfsKernelBased(self, ROverR200):
        """ Get kernel-based density profiles
        
        :param ROverR200: normalized radii at which ``dens_profs`` are defined
        :type ROverR200: (r_res,) floats
        :return: density profiles
        :rtype: (N2, r_res) floats"""
        print_status(rank,self.start_time,'Starting getDensProfsKernelBased() with snap {0}'.format(self.SNAP))
        if rank == 0:
            dens_profs = self.getDensProfsKernelBasedBase(self.xyz.base, self.masses.base, self.idx_cat, self.MIN_NUMBER_PTCS, np.float32(ROverR200))
            return dens_profs
        else:
            return None
        
    def getDensProfsBestFits(self, dens_profs, ROverR200, method):
        """ Get best-fit results for density profile fitting
        
        :param dens_profs: density profiles to be fit, in units of M_sun*h^2/(Mpc)**3
        :type dens_profs: (N3, r_res) floats
        :param ROverR200: normalized radii at which ``dens_profs`` are defined
        :type ROverR200: (r_res,) floats
        :param method: string describing density profile model assumed for fitting
        :type method: string, either `einasto`, `alpha_beta_gamma`, `hernquist`, `nfw`
        :return: best-fits for each object
        :rtype: (N3, n) floats, where n is the number of free parameters in the model ``method``"""
        print_status(rank,self.start_time,'Starting getDensProfsBestFits() with snap {0}'.format(self.SNAP))
        if rank == 0:
            best_fits = self.getDensProfsBestFitsBase(np.float32(dens_profs), np.float32(ROverR200), self.idx_cat, self.r200.base, self.MIN_NUMBER_PTCS, method)
            return best_fits
        else:
            return None
        
    def getConcentrations(self, dens_profs, ROverR200, method):
        """ Get best-fit concentration values of objects from density profile fitting
        
        :param dens_profs: density profiles to be fit, in units of M_sun*h^2/(Mpc)**3
        :type dens_profs: (N3, r_res) floats
        :param ROverR200: normalized radii at which ``dens_profs`` are defined
        :type ROverR200: (r_res,) floats
        :param method: string describing density profile model assumed for fitting
        :type method: string, either `einasto`, `alpha_beta_gamma`, `hernquist`, `nfw`
        :return: best-fit concentration for each object
        :rtype: (N3,) floats"""
        print_status(rank,self.start_time,'Starting getConcentrations() with snap {0}'.format(self.SNAP))
        if rank == 0:
            cs = self.getConcentrationsBase(np.float32(dens_profs), np.float32(ROverR200), self.idx_cat, self.r200.base, self.MIN_NUMBER_PTCS, method)
            return cs
        else:
            return None
        
    def plotDensProfs(self, dens_profs, ROverR200, dens_profs_fit, ROverR200_fit, method, VIZ_DEST):
        """ Draws some simplistic density profiles
        
        :param dens_profs: estimated density profiles, in units of M_sun*h^2/(Mpc)**3
        :type dens_profs: (N2, r_res) floats
        :param ROverR200: radii at which ``dens_profs`` are defined
        :type ROverR200: (r_res,) floats
        :param dens_profs_fit: density profiles to be fit, in units of M_sun*h^2/(Mpc)**3
        :type dens_profs_fit: (N2, r_res2) floats
        :param ROverR200_fit: radii at which best-fits shall be calculated
        :type ROverR200_fit: (r_res2,) floats
        :param method: string describing density profile model assumed for fitting
        :type method: string, either `einasto`, `alpha_beta_gamma`, `hernquist`, `nfw`
        :param VIZ_DEST: visualization folder
        :type VIZ_DEST: string
        :param obj_type: either 'dm' or 'gx', depending on what catalogue we are looking at
        :type obj_type: string
        """
        print_status(rank,self.start_time,'Starting plotDensProfs() with snap {0}'.format(self.SNAP))
        
        if rank == 0:
            suffix = '_'
            obj_centers, obj_masses = self.getMassesCenters()
            getDensProfs(VIZ_DEST, self.SNAP, self.idx_cat, self.r200.base, dens_profs_fit, ROverR200_fit, dens_profs, np.float32(ROverR200), obj_masses, obj_centers, method, self.start_time, self.MASS_UNIT, suffix = suffix)
            del obj_centers; del obj_masses; del ROverR200_fit; del dens_profs; del ROverR200
      
cdef class DensShapeProfs(DensProfs):
    """ Class for density profile and shape profile calculations
    
    Its public methods are ``getIdxCatLocal()``, ``getIdxCatGlobal()``, 
    ``getShapeCatLocal()``, ``getShapeCatGlobal()``, ``vizLocalShapes()``, 
    ``vizGlobalShapes()``, ``plotGlobalEpsHist()``, ``plotLocalEpsHist()``.
    ``vizGlobalShapes()``, ``plotGlobalEpsHist()``, ``plotLocalEpsHist()``.
    ``plotGlobalTHist()``, ``plotLocalTHist()``, ``dumpShapeCatLocal()``,
    ``dumpShapeCatGlobal()``, ``getObjInfoLocal()``, ``getObjInfoGlobal()``."""
    
    cdef int D_LOGSTART
    cdef int D_LOGEND
    cdef int D_BINS
    cdef float M_TOL
    cdef int N_WALL
    cdef int N_MIN
    
    def __init__(self, float[:,:] xyz, float[:] masses, idx_cat, float[:] r200, str SNAP, float L_BOX, int MIN_NUMBER_PTCS, int D_LOGSTART, int D_LOGEND, int D_BINS, float M_TOL, int N_WALL, int N_MIN, str CENTER, double start_time):
        """
        :param xyz: positions of all simulation particles
        :type xyz: (N2,3) floats, N2 >> N1
        :param masses: masses of all simulation particles
        :type masses: (N2,) floats
        :param idx_cat: each entry of the list is a list containing indices of particles belonging to an object
        :type idx_cat: list of length N1
        :param r200: R_200 radii of the parent halos
        :type r200: (N1,) floats
        :param SNAP: snapshot identifier, e.g. '024'
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
        super().__init__(xyz.base, masses.base, idx_cat, r200.base, SNAP, L_BOX, MIN_NUMBER_PTCS, CENTER, start_time)
        self.D_LOGSTART = D_LOGSTART
        self.D_LOGEND = D_LOGEND
        self.D_BINS = D_BINS
        self.M_TOL = M_TOL
        self.N_WALL = N_WALL
        self.N_MIN = N_MIN
        
    def getIdxCatLocal(self):
        """ Fetch local shape index catalogue
        
        :return idx_cat_local: list of indices defining the objects
        :type idx_cat_local: list of length N1, each consisting of a list of int indices"""
        print_status(rank,self.start_time,'Starting getIdxCatLocal() with snap {0}'.format(self.SNAP))
        
        if rank == 0:
            idx_cat_local = self.getIdxCatLocalBase(self.xyz.base, self.masses.base, self.idx_cat, self.MIN_NUMBER_PTCS, self.D_LOGSTART, self.D_LOGEND, self.D_BINS, self.M_TOL, self.N_WALL, self.N_MIN)
            return idx_cat_local
        else:
            return None
    
    def getIdxCatGlobal(self):
        """ Fetch global shape index catalogue
        
        :return idx_cat_global: list of indices defining the objects
        :type idx_cat_global: list of length N1, each consisting of a list of int indices"""
        print_status(rank,self.start_time,'Starting getIdxCatGlobal() with snap {0}'.format(self.SNAP))
        
        if rank == 0:
            idx_cat_global = self.getIdxCatGlobalBase(self.idx_cat, self.MIN_NUMBER_PTCS)
            return idx_cat_global
        else:
            return None
        
    def getShapeCatLocal(self):
        """ Get all relevant local shape data
        
        :return: d, q, s, minor, inter, major, obj_center, obj_m, succeeded (list of indices of converged objects)
        :rtype: 3 x (number_of_objs, D_BINS+1) float arrays, 
            3 x (number_of_objs, D_BINS+1, 3) float arrays, 
            (number_of_objs,3) float array, (number_of_objs,) float array,
            list of length N3
        """
        print_status(rank,self.start_time,'Starting getShapeCatLocal() with snap {0}'.format(self.SNAP))
        if rank == 0:
            d, q, s, minor, inter, major, obj_centers, obj_masses, succeeded = self.getShapeCatLocalBase(self.xyz.base, self.masses.base, self.idx_cat, self.MIN_NUMBER_PTCS, self.D_LOGSTART, self.D_LOGEND, self.D_BINS, self.M_TOL, self.N_WALL, self.N_MIN)
            return d, q, s, minor, inter, major, obj_centers, obj_masses, succeeded
        else:
            return None, None, None, None, None, None, None, None, None
    
    def getShapeCatGlobal(self):
        """ Get all relevant global shape data
        
        :return: d, q, s, minor, inter, major, obj_center, obj_m
        :rtype: 3 x (number_of_objs, D_BINS+1) float arrays, 
            3 x (number_of_objs, D_BINS+1, 3) float arrays, 
            (number_of_objs,3) float array, (number_of_objs,) float array,
        """
        print_status(rank,self.start_time,'Starting getShapeCatGlobal() with snap {0}'.format(self.SNAP))
        if rank == 0:
            d, q, s, minor, inter, major, obj_centers, obj_masses = self.getShapeCatGlobalBase(self.xyz.base, self.masses.base, self.idx_cat, self.MIN_NUMBER_PTCS, self.M_TOL, self.N_WALL, self.N_MIN)
            return d, q, s, minor, inter, major, obj_centers, obj_masses
        else:
            return None, None, None, None, None, None, None, None
    
    def vizLocalShapes(self, obj_numbers, VIZ_DEST):
        """ Visualize local shape of objects with numbers ``obj_numbers``
        
        :param obj_numbers: list of object indices for which to visualize local shapes
        :type obj_numbers: list of int
        :param VIZ_DEST: visualization folder
        :type VIZ_DEST: strings"""
        print_status(rank,self.start_time,'Starting vizLocalShapes() with snap {0}'.format(self.SNAP))
        
        if rank == 0:
            # Retrieve shape information
            d, q, s, minor, inter, major, centers, obj_masses, succeeded = self.getShapeCatLocalBase(self.xyz.base, self.masses.base, self.idx_cat, self.MIN_NUMBER_PTCS, self.D_LOGSTART, self.D_LOGEND, self.D_BINS, self.M_TOL, self.N_WALL, self.N_MIN)
            del obj_masses; del succeeded
            idx_cat_local = self.getIdxCatLocal()
            
            # Viz all objects under 'obj_numbers'
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
                    obj = np.zeros((len(idx_cat_local[obj_number]),3), dtype = np.float32)
                    masses_obj = np.zeros((len(idx_cat_local[obj_number]),), dtype = np.float32)
                    for idx, ptc in enumerate(idx_cat_local[obj_number]):
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
                    fig.savefig("{}/LocalObj{}_{}.pdf".format(VIZ_DEST, obj_number, self.SNAP), bbox_inches='tight')
        
    def vizGlobalShapes(self, obj_numbers, VIZ_DEST):
        """ Visualize global shape of objects with numbers ``obj_numbers``
        
        :param obj_numbers: list of object indices for which to visualize global shapes
        :type obj_numbers: list of ints
        :param VIZ_DEST: visualization folder
        :type VIZ_DEST: string"""
        print_status(rank,self.start_time,'Starting vizGlobalShapes() with snap {0}'.format(self.SNAP))

        if rank == 0:
            # Retrieve shape information
            d, q, s, minor, inter, major, centers, obj_masses = self.getShapeCatGlobalBase(self.xyz.base, self.masses.base, self.idx_cat, self.MIN_NUMBER_PTCS, self.M_TOL, self.N_WALL, self.N_MIN)
            del obj_masses
            idx_cat_global = self.getIdxCatGlobal()
            
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
                    obj = np.zeros((len(idx_cat_global[obj_number]),3), dtype = np.float32)
                    masses_obj = np.zeros((len(idx_cat_global[obj_number]),), dtype = np.float32)
                    for idx, ptc in enumerate(idx_cat_global[obj_number]):
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
                    ax.quiver(*center, major_obj[0][0], major_obj[0][1], major_obj[0][2], length=d_obj[0], color='m', label= "Major")
                    ax.quiver(*center, inter_obj[0][0], inter_obj[0][1], inter_obj[0][2], length=q_obj[0]*d_obj[0], color='c', label = "Intermediate")
                    ax.quiver(*center, minor_obj[0][0], minor_obj[0][1], minor_obj[0][2], length=s_obj[0]*d_obj[0], color='y', label = "Minor")
                    fontP = FontProperties()
                    fontP.set_size('xx-small')
                    plt.legend(bbox_to_anchor=(0.95, 1), loc='upper right', prop=fontP)  
                    plt.xlabel(r"x (Mpc/h)")
                    plt.ylabel(r"y (Mpc/h)")
                    ax.set_zlabel(r"z (Mpc/h)")
                    ax.set_box_aspect([1,1,1])
                    set_axes_equal(ax)
                    fig.savefig("{}/GlobalObj{}_{}.pdf".format(VIZ_DEST, obj_number, self.SNAP), bbox_inches='tight')
    
    def plotGlobalEpsHist(self, HIST_NB_BINS, VIZ_DEST):
        """ Plot global ellipticity histogram
        
        :param HIST_NB_BINS: number of histogram bins
        :type HIST_NB_BINS: int
        :param VIZ_DEST: visualization folder
        :type VIZ_DEST: string"""
        print_status(rank,self.start_time,'Starting plotGlobalEpsHist() with snap {0}'.format(self.SNAP))
        
        if rank == 0:
            suffix = '_'
            idx_cat_global = self.getIdxCatGlobal()
            getGlobalEpsHist(idx_cat_global, self.xyz.base, self.masses.base, self.L_BOX, self.CENTER, VIZ_DEST, self.SNAP, suffix = suffix, HIST_NB_BINS = HIST_NB_BINS)
            del idx_cat_global

    def plotLocalEpsHist(self, frac_r200, HIST_NB_BINS, VIZ_DEST):
        """ Plot local ellipticity histogram at depth ``frac_r200``
        
        :param frac_r200: depth of objects to plot ellipticity, in units of R200
        :type frac_r200: float
        :param HIST_NB_BINS: number of histogram bins
        :type HIST_NB_BINS: int
        :param VIZ_DEST: visualization folder
        :type VIZ_DEST: string"""
        print_status(rank,self.start_time,'Starting plotLocalEpsHist() with snap {0}'.format(self.SNAP))
        
        if rank == 0:
            suffix = '_'
            idx_cat_local = self.getIdxCatLocal()
            getLocalEpsHist(idx_cat_local, self.xyz.base, self.masses.base, self.r200.base, self.L_BOX, self.CENTER, VIZ_DEST, self.SNAP, frac_r200, suffix = suffix, HIST_NB_BINS = HIST_NB_BINS)
            del idx_cat_local
    
    def plotLocalTHist(self, HIST_NB_BINS, VIZ_DEST, frac_r200):
        """ Plot local triaxiality histogram at depth ``frac_r200``
        
        :param HIST_NB_BINS: number of histogram bins
        :type HIST_NB_BINS: int
        :param VIZ_DEST: visualization folder
        :type VIZ_DEST: string
        :param frac_r200: depth of objects to plot triaxiality, in units of R200
        :type frac_r200: float"""
        print_status(rank,self.start_time,'Starting plotLocalTHist() with snap {0}'.format(self.SNAP))
        if rank == 0:
            suffix = '_'
            self.plotLocalTHistBase(self.xyz.base, self.masses.base, self.idx_cat, self.MIN_NUMBER_PTCS, self.D_LOGSTART, self.D_LOGEND, self.D_BINS, self.M_TOL, self.N_WALL, self.N_MIN, VIZ_DEST, HIST_NB_BINS, frac_r200, suffix = suffix)
    
    def plotGlobalTHist(self, HIST_NB_BINS, VIZ_DEST):
        """ Plot global triaxiality histogram
        
        :param HIST_NB_BINS: number of histogram bins
        :type HIST_NB_BINS: int
        :param VIZ_DEST: visualization folder
        :type VIZ_DEST: string"""
        print_status(rank,self.start_time,'Starting plotGlobalTHist() with snap {0}'.format(self.SNAP))
        if rank == 0:
            suffix = '_'
            self.plotGlobalTHistBase(self.xyz.base, self.masses.base, self.idx_cat, self.MIN_NUMBER_PTCS, self.M_TOL, self.N_WALL, self.N_MIN, VIZ_DEST, HIST_NB_BINS, suffix = suffix)
    
    def plotShapeProfs(self, VIZ_DEST):
        """ Draws shape profiles, also mass bin-decomposed ones
        
        :param VIZ_DEST: visualization folder
        :type VIZ_DEST: string"""
        print_status(rank,self.start_time,'Starting plotShapeProfs() with snap {0}'.format(self.SNAP))
        if rank == 0:
            suffix = '_'
            self.plotShapeProfsBase(self.xyz.base, self.masses.base, self.idx_cat, self.MIN_NUMBER_PTCS, self.D_LOGSTART, self.D_LOGEND, self.D_BINS, self.M_TOL, self.N_WALL, self.N_MIN, VIZ_DEST, suffix = suffix)
    
    def dumpShapeCatLocal(self, CAT_DEST):
        """ Dumps all relevant local shape data into ``CAT_DEST``
        
        :param CAT_DEST: catalogue folder
        :type CAT_DEST: string"""
        print_status(rank,self.start_time,'Starting dumpShapeCatLocal() with snap {0}'.format(self.SNAP))
        if rank == 0:
            suffix = '_'
            self.dumpShapeCatLocalBase(self.xyz.base, self.masses.base, self.idx_cat, self.MIN_NUMBER_PTCS, self.D_LOGSTART, self.D_LOGEND, self.D_BINS, self.M_TOL, self.N_WALL, self.N_MIN, CAT_DEST, suffix = suffix)
    
    def dumpShapeCatGlobal(self, CAT_DEST):
        """ Dumps all relevant global shape data into ``CAT_DEST``
        
        :param CAT_DEST: catalogue folder
        :type CAT_DEST: string"""
        print_status(rank,self.start_time,'Starting dumpShapeCatGlobal() with snap {0}'.format(self.SNAP))
        if rank == 0:
            suffix = '_'
            self.dumpShapeCatGlobalBase(self.xyz.base, self.masses.base, self.idx_cat, self.MIN_NUMBER_PTCS, self.M_TOL, self.N_WALL, self.N_MIN, CAT_DEST, suffix = suffix)
    
    def getObjInfoLocal(self):
        """ Print basic info about the objects used for local shape estimation such as number of converged objects"""
        print_status(rank,self.start_time,'Starting getObjInfoLocal() with snap {0}'.format(self.SNAP))
        obj_type = 'unspecified'
        self.getObjInfoLocalBase(self.xyz.base, self.masses.base, self.idx_cat, self.MIN_NUMBER_PTCS, self.D_LOGSTART, self.D_LOGEND, self.D_BINS, self.M_TOL, self.N_WALL, self.N_MIN, obj_type)

    def getObjInfoGlobal(self):
        """ Print basic info about the objects used for global shape estimation such as number of converged objects"""
        print_status(rank,self.start_time,'Starting getObjInfoGlobal() with snap {0}'.format(self.SNAP))
        obj_type = 'unspecified'
        self.getObjInfoGlobalBase(self.idx_cat, self.MIN_NUMBER_PTCS, obj_type)
    
cdef class DensProfsHDF5(CosmicBase):
    """ Class for density profile calculations for Gadget-style HDF5 data
    
    Its public methods are ``getXYZMasses()``, ``getVelXYZ()``, 
    ``getIdxCat()``, ``getMassesCenters()``, ``getDensProfsDirectBinning()``,
    ``getDensProfsKernelBased()``, ``getDensProfsBestFits()``, ``getConcentrations()``, 
    ``plotDensProfs()``."""
    
    cdef str HDF5_SNAP_DEST
    cdef str HDF5_GROUP_DEST
    cdef int MIN_NUMBER_STAR_PTCS
    cdef int SNAP_MAX
    cdef bint WANT_RVIR
    
    def __init__(self, str HDF5_SNAP_DEST, str HDF5_GROUP_DEST, str SNAP, int SNAP_MAX, float L_BOX, int MIN_NUMBER_PTCS, int MIN_NUMBER_STAR_PTCS, str CENTER, bint WANT_RVIR, double start_time):
        """
        :param HDF5_SNAP_DEST: where we can find the snapshot
        :type HDF5_SNAP_DEST: string
        :param HDF5_GROUP_DEST: where we can find the group files
        :type HDF5_GROUP_DEST: string
        :param SNAP: e.g. '024'
        :type SNAP: string
        :param SNAP_MAX: e.g. 16
        :type SNAP_MAX: int
        :param SNAP: snapshot identifier, e.g. '024'
        :type SNAP: string
        :param L_BOX: simulation box side length
        :type L_BOX: float, units: Mpc/h
        :param MIN_NUMBER_PTCS: minimum number of DM particles for halo to qualify for morphology calculation
        :type MIN_NUMBER_PTCS: int
        :param MIN_NUMBER_STAR_PTCS: minimum number of star particles for galaxy to qualify for morphology calculation
        :type MIN_NUMBER_STAR_PTCS: int
        :param CENTER: shape quantities will be calculated with respect to CENTER = 'mode' (point of highest density)
            or 'com' (center of mass) of each halo
        :type CENTER: str
        :param WANT_RVIR: Whether or not we want quantities (e.g. D_LOGSTART) expressed 
            with respect to the virial radius R_vir or the overdensity radius R_200
        :type WANT_RVIR: boolean
        :param start_time: time of start of object initialization
        :type start_time: float"""
        super().__init__(SNAP, L_BOX, MIN_NUMBER_PTCS, CENTER, start_time)
        self.HDF5_SNAP_DEST = HDF5_SNAP_DEST
        self.HDF5_GROUP_DEST = HDF5_GROUP_DEST
        self.MIN_NUMBER_STAR_PTCS = MIN_NUMBER_STAR_PTCS
        self.SNAP_MAX = SNAP_MAX
        self.WANT_RVIR = WANT_RVIR
        
    def getXYZMasses(self, obj_type = 'dm'):
        """ Retrieve positions and masses of DM/gx
        
        :param obj_type: either 'dm' or 'gx', depending on what catalogue we are looking at
        :type obj_type: string
        :return xyz, masses, MIN_NUMBER_PTCS: positions, masses, and minimum number of particles
        :rtype: (N2,3) floats, (N2,) floats, int"""
        print_status(rank,self.start_time,'Starting getXYZMasses() with snap {0} for obj_type {1}'.format(self.SNAP, obj_type))
        if obj_type == 'dm':
            xyz, masses, velxyz = getHDF5DMData(self.HDF5_SNAP_DEST, self.SNAP_MAX, self.SNAP)
            del velxyz
            MIN_NUMBER_PTCS = self.MIN_NUMBER_PTCS
        else:
            xyz, fof_com, sh_com, nb_shs, masses, velxyz, is_star = getHDF5GxData(self.HDF5_SNAP_DEST, self.HDF5_GROUP_DEST, self.SNAP_MAX, self.SNAP)
            del fof_com; del sh_com; del nb_shs; del velxyz; del is_star
            MIN_NUMBER_PTCS = self.MIN_NUMBER_STAR_PTCS
        if rank == 0:
            return xyz, masses, MIN_NUMBER_PTCS
        else:
            del xyz; del masses
            return None, None, None
        
    def getVelXYZ(self, obj_type = 'dm'):
        """ Retrieve velocities of DM/gx
        
        :param obj_type: either 'dm' or 'gx', depending on what catalogue we are looking at
        :type obj_type: string
        :return velxyz: velocity array
        :rtype: (N2,3) floats"""
        print_status(rank,self.start_time,'Starting getVelXYZ() with snap {0} for obj_type {1}'.format(self.SNAP, obj_type))
        if obj_type == 'dm':
            xyz, masses, velxyz = getHDF5DMData(self.HDF5_SNAP_DEST, self.SNAP_MAX, self.SNAP)
            del xyz; del masses
        else:
            xyz, fof_com, sh_com, nb_shs, masses, velxyz, is_star = getHDF5GxData(self.HDF5_SNAP_DEST, self.HDF5_GROUP_DEST, self.SNAP_MAX, self.SNAP)
            del xyz; del fof_com; del sh_com; del nb_shs; del masses; del is_star
        if rank == 0:
            return velxyz
        else:
            del velxyz
            return None
        
    def getIdxCat(self, obj_type = 'dm'):
        """ Fetch catalogue
        
        :param obj_type: either 'dm' or 'gx', depending on what catalogue we are looking at
        :type obj_type: string
        :return cat: list of indices defining the objects
        :rtype: list of length N1, each consisting of a list of int indices"""
        
        print_status(rank,self.start_time,'Starting getIdxCat() with snap {0} for obj_type {1}'.format(self.SNAP, obj_type))
        if rank != 0:
            return None
        if obj_type == 'dm':
            # Import hdf5 data
            print_status(rank,self.start_time,"Getting HDF5 raw data..")
            if rank == 0:
                nb_shs, sh_len, fof_dm_sizes, group_r200, halo_masses, fof_coms = getHDF5SHDMData(self.HDF5_GROUP_DEST, self.SNAP_MAX, self.SNAP, self.WANT_RVIR)
                del fof_coms
                print_status(rank, self.start_time, "Finished HDF5 raw data")
                
                # Construct catalogue
                print_status(rank, self.start_time, "Call calcCSHCat()")
                h_cat, h_r200, h_pass = calcCSHCat(np.array(nb_shs), np.array(sh_len), np.array(fof_dm_sizes), group_r200, halo_masses, self.MIN_NUMBER_PTCS)
                print_status(rank, self.start_time, "Finished calcCSHCat()")
                nb_shs_vec = np.array(nb_shs)
                h_cat_l = [[] for i in range(len(nb_shs))]
                corr = 0
                for i in range(len(nb_shs)):
                    if h_pass[i] == 1:
                        h_cat_l[i] = (np.ma.masked_where(h_cat[i-corr] == 0, h_cat[i-corr]).compressed()-1).tolist()
                    else:
                        corr += 1
                print_status(rank, self.start_time, "Constructed the CSH catalogue. The total number of halos with > 0 SHs is {0}, the total number of halos is {1}, the total number of SHs is {2}, the number of halos that have no SH is {3} and the total number of halos (CSH) that have sufficient resolution is {4}".format(nb_shs_vec[nb_shs_vec != 0].shape[0], len(nb_shs), len(sh_len), nb_shs_vec[nb_shs_vec == 0].shape[0], len([x for x in h_cat_l if x != []])))
                
                self.r200 = h_r200
                del nb_shs; del sh_len; del fof_dm_sizes; del group_r200; del h_cat; del halo_masses; del h_r200
                return h_cat_l
        else:
            # Import hdf5 data
            print_status(rank,self.start_time,"Getting HDF5 raw data..")
            if rank == 0:
                # Construct gx catalogue
                nb_shs, sh_len_gx, fof_gx_sizes = getHDF5SHGxData(self.HDF5_GROUP_DEST, self.SNAP_MAX, self.SNAP)
                print_status(rank, self.start_time, "Creating Gx CAT..")
                gx_cat, gx_pass = calcGxCat(np.array(nb_shs), np.array(sh_len_gx), np.array(fof_gx_sizes), self.MIN_NUMBER_STAR_PTCS)
                print_status(rank, self.start_time, "Finished calcGxCat()")
                gx_cat_l = [[] for i in range(len(nb_shs))]
                corr = 0
                for i in range(len(nb_shs)):
                    if gx_pass[i] == 1:
                        gx_cat_l[i] = (np.ma.masked_where(gx_cat[i-corr] == 0, gx_cat[i-corr]).compressed()-1).tolist()
                    else:
                        corr += 1
                print_status(rank, self.start_time, "Constructed the gx catalogue. The number of valid gxs (after discarding low-resolution ones) is {0}.".format(np.array([0 for x in gx_cat_l if x != []]).shape[0]))
                
                del nb_shs; del sh_len_gx; del fof_gx_sizes; del gx_cat
                return gx_cat_l
    
    def getMassesCenters(self, obj_type = 'dm'):
        """ Calculate total mass and centers of objects
        
        :param obj_type: either 'dm' or 'gx', depending on what catalogue we are looking at
        :type obj_type: string
        :return centers, m: centers and masses
        :rtype: (N,3) and (N,) floats"""
        print_status(rank,self.start_time,'Starting getMassesCenters() with snap {0} for obj_type {1}'.format(self.SNAP, obj_type))
        xyz, masses, MIN_NUMBER_PTCS = self.getXYZMasses(obj_type)
        if rank == 0:
            idx_cat = self.getIdxCat(obj_type)
            centers, ms = self.getMassesCentersBase(xyz, masses, idx_cat, MIN_NUMBER_PTCS)
            del xyz; del masses
            return centers, ms
        else:
            del xyz; del masses
            return None, None
    
    def getDensProfsDirectBinning(self, ROverR200, obj_type = 'dm'):
        """ Get direct-binning-based density profiles
        
        :param ROverR200: normalized radii at which ``dens_profs`` are defined
        :type ROverR200: (r_res,) floats
        :param obj_type: either 'dm' or 'gx', depending on what catalogue we are looking at
        :type obj_type: string
        :return: density profiles
        :rtype: (N2, r_res) floats"""
        print_status(rank,self.start_time,'Starting getDensProfsDirectBinning() with snap {0} for obj_type {1}'.format(self.SNAP, obj_type))
        xyz, masses, MIN_NUMBER_PTCS = self.getXYZMasses(obj_type)
        if rank == 0:
            idx_cat = self.getIdxCat(obj_type)
            dens_profs = self.getDensProfsDirectBinningBase(xyz, masses, idx_cat, MIN_NUMBER_PTCS, np.float32(ROverR200))
            del xyz; del masses; del idx_cat
            return dens_profs
        else:
            del xyz; del masses
            return None

    def getDensProfsKernelBased(self, ROverR200, obj_type = 'dm'):
        """ Get kernel-based density profiles
        
        :param ROverR200: normalized radii at which ``dens_profs`` are defined
        :type ROverR200: (r_res,) floats
        :param obj_type: either 'dm' or 'gx', depending on what catalogue we are looking at
        :type obj_type: string
        :return: density profiles
        :rtype: (N2, r_res) floats"""
        print_status(rank,self.start_time,'Starting getDensProfsKernelBased() with snap {0} for obj_type {1}'.format(self.SNAP, obj_type))
        xyz, masses, MIN_NUMBER_PTCS = self.getXYZMasses(obj_type)
        if rank == 0:
            idx_cat = self.getIdxCat(obj_type)
            dens_profs = self.getDensProfsKernelBasedBase(xyz, masses, idx_cat, MIN_NUMBER_PTCS, np.float32(ROverR200))
            del xyz; del masses; del idx_cat
            return dens_profs
        else:
            del xyz; del masses
            return None
        
    def getDensProfsBestFits(self, dens_profs, ROverR200, method, obj_type = 'dm'):
        """ Get best-fit results for density profile fitting
        
        :param dens_profs: density profiles to be fit, in units of M_sun*h^2/(Mpc)**3
        :type dens_profs: (N3, r_res) floats
        :param ROverR200: normalized radii at which ``dens_profs`` are defined
        :type ROverR200: (r_res,) floats
        :param method: string describing density profile model assumed for fitting
        :type method: string, either `einasto`, `alpha_beta_gamma`, `hernquist`, `nfw`
        :param obj_type: either 'dm' or 'gx', depending on what catalogue we are looking at
        :type obj_type: string
        :return: best-fits for each object
        :rtype: (N3, n) floats, where n is the number of free parameters in the model ``method``"""
        print_status(rank,self.start_time,'Starting getDensProfsBestFits() with snap {0} for obj_type {1}'.format(self.SNAP, obj_type))
        xyz, masses, MIN_NUMBER_PTCS = self.getXYZMasses(obj_type)
        if rank == 0:
            idx_cat = self.getIdxCat(obj_type)
            best_fits = self.getDensProfsBestFitsBase(np.float32(dens_profs), np.float32(ROverR200), idx_cat, self.r200.base, MIN_NUMBER_PTCS, method)
            del xyz; del masses; del idx_cat
            return best_fits
        else:
            del xyz; del masses
            return None
        
    def getConcentrations(self, dens_profs, ROverR200, method, obj_type = 'dm'):
        """ Get best-fit concentration values of objects from density profile fitting
        
        :param dens_profs: density profiles to be fit, in units of M_sun*h^2/(Mpc)**3
        :type dens_profs: (N3, r_res) floats
        :param ROverR200: normalized radii at which ``dens_profs`` are defined
        :type ROverR200: (r_res,) floats
        :param method: string describing density profile model assumed for fitting
        :type method: string, either `einasto`, `alpha_beta_gamma`, `hernquist`, `nfw`
        :param obj_type: either 'dm' or 'gx', depending on what catalogue we are looking at
        :type obj_type: string
        :return: best-fit concentration for each object
        :rtype: (N3,) floats"""
        print_status(rank,self.start_time,'Starting getConcentrations() with snap {0} for obj_type {1}'.format(self.SNAP, obj_type))
        xyz, masses, MIN_NUMBER_PTCS = self.getXYZMasses(obj_type)
        if rank == 0:
            idx_cat = self.getIdxCat(obj_type)
            cs = self.getConcentrationsBase(np.float32(dens_profs), np.float32(ROverR200), idx_cat, self.r200.base, MIN_NUMBER_PTCS, method)
            del xyz; del masses; del idx_cat
            return cs
        else:
            del xyz; del masses
            return None
        
    def plotDensProfs(self, dens_profs, ROverR200, dens_profs_fit, ROverR200_fit, method, VIZ_DEST, obj_type = 'dm'):
        """ Draws some simplistic density profiles
        
        :param dens_profs: estimated density profiles, in units of M_sun*h^2/(Mpc)**3
        :type dens_profs: (N2, r_res) floats
        :param ROverR200: radii at which ``dens_profs`` are defined
        :type ROverR200: (r_res,) floats
        :param dens_profs_fit: density profiles to be fit, in units of M_sun*h^2/(Mpc)**3
        :type dens_profs_fit: (N2, r_res2) floats
        :param ROverR200_fit: radii at which best-fits shall be calculated
        :type ROverR200_fit: (r_res2,) floats
        :param method: string describing density profile model assumed for fitting
        :type method: string, either `einasto`, `alpha_beta_gamma`, `hernquist`, `nfw`
        :param VIZ_DEST: visualization folder
        :type VIZ_DEST: string
        :param obj_type: either 'dm' or 'gx', depending on what catalogue we are looking at
        :type obj_type: string
        """
        print_status(rank,self.start_time,'Starting plotDensProfs() with snap {0}'.format(self.SNAP))
        
        if rank == 0:
            idx_cat = self.getIdxCat(obj_type)
            suffix = '_{}_'.format(obj_type)
            obj_centers, obj_masses = self.getMassesCenters(obj_type)
            getDensProfs(VIZ_DEST, self.SNAP, idx_cat, self.r200.base, dens_profs_fit, ROverR200_fit, dens_profs, ROverR200, obj_masses, obj_centers, method, self.start_time, self.MASS_UNIT, suffix = suffix)
            del obj_centers; del obj_masses; del ROverR200_fit; del dens_profs; del ROverR200

cdef class DensShapeProfsHDF5(DensProfsHDF5):
    """ Class for density profile and shape profile calculations for Gadget-style HDF5 data
    
    Its public methods are ``getIdxCatLocal()``, ``getIdxCatGlobal()``, 
    ``getShapeCatLocal()``, ``getShapeCatGlobal()``, ``vizLocalShapes()``, 
    ``vizGlobalShapes()``, ``plotGlobalEpsHist()``, ``plotLocalEpsHist()``.
    ``vizGlobalShapes()``, ``plotGlobalEpsHist()``, ``plotLocalEpsHist()``.
    ``plotGlobalTHist()``, ``plotLocalTHist()``, ``dumpShapeCatLocal()``,
    ``dumpShapeCatGlobal()``, ``dumpShapeCatVelLocal()``, ``dumpShapeCatVelGlobal()``, 
    ``getObjInfoLocal()``, ``getObjInfoGlobal()``, ``getObjInfoVelLocal()``, 
    ``getObjInfoVelGlobal()``."""
    
    cdef int D_LOGSTART
    cdef int D_LOGEND
    cdef int D_BINS
    cdef float M_TOL
    cdef int N_WALL
    cdef int N_MIN
    
    def __init__(self, str HDF5_SNAP_DEST, str HDF5_GROUP_DEST, str SNAP, int SNAP_MAX, float L_BOX, int MIN_NUMBER_PTCS, int MIN_NUMBER_STAR_PTCS, int D_LOGSTART, int D_LOGEND, int D_BINS, float M_TOL, int N_WALL, int N_MIN, str CENTER, bint WANT_RVIR, double start_time):
        
        super().__init__(HDF5_SNAP_DEST, HDF5_GROUP_DEST, SNAP, SNAP_MAX, L_BOX, MIN_NUMBER_PTCS, MIN_NUMBER_STAR_PTCS, CENTER, WANT_RVIR, start_time)
        self.D_LOGSTART = D_LOGSTART
        self.D_LOGEND = D_LOGEND
        self.D_BINS = D_BINS
        self.M_TOL = M_TOL
        self.N_WALL = N_WALL
        self.N_MIN = N_MIN
                
    def getIdxCatLocal(self, obj_type = 'dm'):
        """ Fetch local shape index catalogue
        
        :param obj_type: either 'dm' or 'gx', depending on what catalogue we are looking at
        :type obj_type: string
        :return idx_cat_local: list of indices defining the objects
        :type idx_cat_local: list of length N1, each consisting of a list of int indices"""
        print_status(rank, self.start_time,'Starting getIdxCatLocal() with snap {0}'.format(self.SNAP))
        
        xyz, masses, MIN_NUMBER_PTCS = self.getXYZMasses(obj_type)
        
        if rank == 0:
            idx_cat = self.getIdxCat(obj_type)
            idx_cat_local = self.getIdxCatLocalBase(xyz, masses, idx_cat, MIN_NUMBER_PTCS, self.D_LOGSTART, self.D_LOGEND, self.D_BINS, self.M_TOL, self.N_WALL, self.N_MIN)
            del xyz; del masses; del idx_cat
            return idx_cat_local
        else:
            del xyz; del masses
            return None
    
    def getIdxCatGlobal(self, obj_type = 'dm'):
        """ Fetch global shape index catalogue
        
        :param obj_type: either 'dm' or 'gx', depending on what catalogue we are looking at
        :type obj_type: string
        :return idx_cat_global: list of indices defining the objects
        :type idx_cat_global: list of length N1, each consisting of a list of int indices"""
        print_status(rank,self.start_time,'Starting getIdxCatGlobal() with snap {0}'.format(self.SNAP))
        
        if obj_type == 'dm':
            MIN_NUMBER_PTCS = self.MIN_NUMBER_PTCS
        else:
            MIN_NUMBER_PTCS = self.MIN_NUMBER_STAR_PTCS
        if rank == 0:
            idx_cat = self.getIdxCat(obj_type)
            idx_cat_global = self.getIdxCatGlobalBase(idx_cat, MIN_NUMBER_PTCS)
            del idx_cat
            return idx_cat_global
        else:
            return None
        
    def getShapeCatLocal(self, obj_type = 'dm'):
        """ Get all relevant local shape data
        
        :param obj_type: either 'dm' or 'gx', depending on what catalogue we are looking at
        :type obj_type: string
        :return: d, q, s, minor, inter, major, obj_center, obj_m, succeeded (list of indices of converged objects)
        :rtype: 3 x (number_of_objs, D_BINS+1) float arrays, 
            3 x (number_of_objs, D_BINS+1, 3) float arrays, 
            (number_of_objs,3) float array, (number_of_objs,) float array,
            list of length N3
        """
        print_status(rank,self.start_time,'Starting getShapeCatLocal() with snap {0}'.format(self.SNAP))
        xyz, masses, MIN_NUMBER_PTCS = self.getXYZMasses(obj_type)
            
        if rank == 0:
            idx_cat = self.getIdxCat(obj_type)
            d, q, s, minor, inter, major, obj_centers, obj_masses, succeeded = self.getShapeCatLocalBase(xyz, masses, idx_cat, MIN_NUMBER_PTCS, self.D_LOGSTART, self.D_LOGEND, self.D_BINS, self.M_TOL, self.N_WALL, self.N_MIN)
            del xyz; del masses; del idx_cat
            return d, q, s, minor, inter, major, obj_centers, obj_masses, succeeded
        else:
            del xyz; del masses
            return None, None, None, None, None, None, None, None, None
    
    def getShapeCatGlobal(self, obj_type = 'dm'):
        """ Get all relevant global shape data
        
        :param obj_type: either 'dm' or 'gx', depending on what catalogue we are looking at
        :type obj_type: string
        :return: d, q, s, minor, inter, major, obj_center, obj_m
        :rtype: 3 x (number_of_objs, D_BINS+1) float arrays, 
            3 x (number_of_objs, D_BINS+1, 3) float arrays, 
            (number_of_objs,3) float array, (number_of_objs,) float array,
        """
        print_status(rank,self.start_time,'Starting getShapeCatGlobal() with snap {0}'.format(self.SNAP))
        xyz, masses, MIN_NUMBER_PTCS = self.getXYZMasses(obj_type)
        
        if rank == 0:
            idx_cat = self.getIdxCat(obj_type)
            d, q, s, minor, inter, major, obj_centers, obj_masses = self.getShapeCatGlobalBase(xyz, masses, idx_cat, MIN_NUMBER_PTCS, self.M_TOL, self.N_WALL, self.N_MIN)
            del xyz; del masses; del idx_cat
            return d, q, s, minor, inter, major, obj_centers, obj_masses
        else:
            del xyz; del masses
            return None, None, None, None, None, None, None, None
        
    def getShapeCatVelLocal(self, obj_type = 'dm'):
        """ Get all relevant local velocity shape data
        
        :param obj_type: either 'dm' or 'gx', depending on what catalogue we are looking at
        :type obj_type: string
        :return: d, q, s, minor, inter, major, obj_center, obj_m, succeeded (list of indices of converged objects)
        :rtype: 3 x (number_of_objs, D_BINS+1) float arrays, 
            3 x (number_of_objs, D_BINS+1, 3) float arrays, 
            (number_of_objs,3) float array, (number_of_objs,) float array,
            list of length N3
        """
        print_status(rank,self.start_time,'Starting getShapeCatVelLocal() with snap {0}'.format(self.SNAP))
        xyz, masses, MIN_NUMBER_PTCS = self.getXYZMasses(obj_type)
        velxyz = self.getVelXYZ(obj_type)
        if rank == 0:
            idx_cat = self.getIdxCat(obj_type)
            d, q, s, minor, inter, major, obj_centers, obj_masses, succeeded = self.getShapeCatVelLocalBase(xyz, velxyz, masses, idx_cat, MIN_NUMBER_PTCS, self.D_LOGSTART, self.D_LOGEND, self.D_BINS, self.M_TOL, self.N_WALL, self.N_MIN)
            del xyz; del velxyz; del masses; del idx_cat
            return d, q, s, minor, inter, major, obj_centers, obj_masses, succeeded
        else:
            del xyz; del velxyz; del masses
            return None, None, None, None, None, None, None, None, None
    
    def getShapeCatVelGlobal(self, obj_type = 'dm'):
        """ Get all relevant global velocity shape data
        
        :param obj_type: either 'dm' or 'gx', depending on what catalogue we are looking at
        :type obj_type: string
        :return: d, q, s, minor, inter, major, obj_center, obj_m
        :rtype: 3 x (number_of_objs, D_BINS+1) float arrays, 
            3 x (number_of_objs, D_BINS+1, 3) float arrays, 
            (number_of_objs,3) float array, (number_of_objs,) float array,
        """
        print_status(rank,self.start_time,'Starting getShapeCatVelGlobal() with snap {0}'.format(self.SNAP))
        xyz, masses, MIN_NUMBER_PTCS = self.getXYZMasses(obj_type)
        velxyz = self.getVelXYZ(obj_type)
        if rank == 0:
            idx_cat = self.getIdxCat(obj_type)
            d, q, s, minor, inter, major, obj_centers, obj_masses = self.getShapeCatVelGlobalBase(xyz, velxyz, masses, idx_cat, MIN_NUMBER_PTCS, self.M_TOL, self.N_WALL, self.N_MIN, self.CENTER, self.SAFE)
            del xyz; del velxyz; del masses; del idx_cat
            return d, q, s, minor, inter, major, obj_centers, obj_masses
        else:
            del xyz; del velxyz; del masses
            return None, None, None, None, None, None, None, None
    
    def vizLocalShapes(self, obj_numbers, VIZ_DEST, obj_type = 'dm'):
        """ Visualize local shape of objects with numbers ``obj_numbers``
        
        :param obj_numbers: list of object indices for which to visualize local shapes
        :type obj_numbers: list of int
        :param VIZ_DEST: visualization folder
        :type VIZ_DEST: strings
        :param obj_type: either 'dm' or 'gx', depending on what catalogue we are looking at
        :type obj_type: string"""
        print_status(rank,self.start_time,'Starting vizLocalShapes() with snap {0}'.format(self.SNAP))
        
        xyz, masses, MIN_NUMBER_PTCS = self.getXYZMasses(obj_type)
        suffix = '_{}_'.format(obj_type)
        
        if rank == 0:
            idx_cat = self.getIdxCat(obj_type)
            # Retrieve shape information
            d, q, s, minor, inter, major, centers, obj_m, succeeded = self.getShapeCatLocalBase(xyz, masses, idx_cat, MIN_NUMBER_PTCS, self.D_LOGSTART, self.D_LOGEND, self.D_BINS, self.M_TOL, self.N_WALL, self.N_MIN)
            del obj_m; del succeeded
            idx_cat_local = self.getIdxCatLocal(obj_type)
            
            # Viz all objects under 'obj_numbers'
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
                    obj = np.zeros((len(idx_cat_local[obj_number]),3), dtype = np.float32)
                    masses_obj = np.zeros((len(idx_cat_local[obj_number]),), dtype = np.float32)
                    for idx, ptc in enumerate(idx_cat_local[obj_number]):
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
                    fig.savefig("{}/LocalObj{}{}{}.pdf".format(VIZ_DEST, obj_number, suffix, self.SNAP), bbox_inches='tight')
        
    def vizGlobalShapes(self, obj_numbers, VIZ_DEST, obj_type = 'dm'):
        """ Visualize global shape of objects with numbers ``obj_numbers``
        
        :param obj_numbers: list of object indices for which to visualize global shapes
        :type obj_numbers: list of ints
        :param VIZ_DEST: visualization folder
        :type VIZ_DEST: string
        :param obj_type: either 'dm' or 'gx', depending on what catalogue we are looking at
        :type obj_type: string"""
        print_status(rank,self.start_time,'Starting vizGlobalShapes() with snap {0}'.format(self.SNAP))

        xyz, masses, MIN_NUMBER_PTCS = self.getXYZMasses(obj_type)
        suffix = '_{}_'.format(obj_type)
        
        if rank == 0:
            idx_cat = self.getIdxCat(obj_type)
            # Retrieve shape information
            d, q, s, minor, inter, major, centers, obj_m = self.getShapeCatGlobalBase(xyz, masses, idx_cat, MIN_NUMBER_PTCS, self.M_TOL, self.N_WALL, self.N_MIN)
            del obj_m
            idx_cat_global = self.getIdxCatGlobal(obj_type)
            
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
                    obj = np.zeros((len(idx_cat_global[obj_number]),3), dtype = np.float32)
                    masses_obj = np.zeros((len(idx_cat_global[obj_number]),), dtype = np.float32)
                    for idx, ptc in enumerate(idx_cat_global[obj_number]):
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
                    ax.quiver(*center, major_obj[0][0], major_obj[0][1], major_obj[0][2], length=d_obj[0], color='m', label= "Major")
                    ax.quiver(*center, inter_obj[0][0], inter_obj[0][1], inter_obj[0][2], length=q_obj[0]*d_obj[0], color='c', label = "Intermediate")
                    ax.quiver(*center, minor_obj[0][0], minor_obj[0][1], minor_obj[0][2], length=s_obj[0]*d_obj[0], color='y', label = "Minor")
                    fontP = FontProperties()
                    fontP.set_size('xx-small')
                    plt.legend(bbox_to_anchor=(0.95, 1), loc='upper right', prop=fontP)  
                    plt.xlabel(r"x (Mpc/h)")
                    plt.ylabel(r"y (Mpc/h)")
                    ax.set_zlabel(r"z (Mpc/h)")
                    ax.set_box_aspect([1,1,1])
                    set_axes_equal(ax)
                    fig.savefig("{}/GlobalObj{}{}{}.pdf".format(VIZ_DEST, obj_number, suffix, self.SNAP), bbox_inches='tight')
    
    def plotGlobalEpsHist(self, HIST_NB_BINS, VIZ_DEST, obj_type = 'dm'):
        """ Plot global ellipticity histogram
        
        :param HIST_NB_BINS: number of histogram bins
        :type HIST_NB_BINS: int
        :param VIZ_DEST: visualization folder
        :type VIZ_DEST: string
        :param obj_type: either 'dm' or 'gx', depending on what catalogue we are looking at
        :type obj_type: string"""
        print_status(rank,self.start_time,'Starting plotGlobalEpsHist() with snap {0}'.format(self.SNAP))
        
        xyz, masses, MIN_NUMBER_PTCS = self.getXYZMasses(obj_type)
        suffix = '_{}_'.format(obj_type)
        
        if rank == 0:
            idx_cat_global = self.getIdxCatGlobal(obj_type)
            getGlobalEpsHist(idx_cat_global, xyz, masses, self.L_BOX, self.CENTER, VIZ_DEST, self.SNAP, suffix = suffix, HIST_NB_BINS = HIST_NB_BINS)
            del xyz; del masses; del idx_cat_global
        else:
            del xyz; del masses

    def plotLocalEpsHist(self, frac_r200, HIST_NB_BINS, VIZ_DEST, obj_type = 'dm'):
        """ Plot local ellipticity histogram at depth ``frac_r200``
        
        :param frac_r200: depth of objects to plot ellipticity, in units of R200
        :type frac_r200: float
        :param HIST_NB_BINS: number of histogram bins
        :type HIST_NB_BINS: int
        :param VIZ_DEST: visualization folder
        :type VIZ_DEST: string
        :param obj_type: either 'dm' or 'gx', depending on what catalogue we are looking at
        :type obj_type: string"""
        print_status(rank,self.start_time,'Starting plotLocalEpsHist() with snap {0}'.format(self.SNAP))
        
        xyz, masses, MIN_NUMBER_PTCS = self.getXYZMasses(obj_type)
        suffix = '_{}_'.format(obj_type)
            
        if rank == 0:
            idx_cat_local = self.getIdxCatLocal(obj_type)
            getLocalEpsHist(idx_cat_local, xyz, masses, self.r200.base, self.L_BOX, self.CENTER, VIZ_DEST, self.SNAP, frac_r200, suffix = suffix, HIST_NB_BINS = HIST_NB_BINS)
            del xyz; del masses; del idx_cat_local
        else:
            del xyz; del masses
    
    def plotLocalTHist(self, HIST_NB_BINS, VIZ_DEST, frac_r200, obj_type = 'dm'):
        """ Plot local triaxiality histogram at depth ``frac_r200``
        
        :param HIST_NB_BINS: number of histogram bins
        :type HIST_NB_BINS: int
        :param VIZ_DEST: visualization folder
        :type VIZ_DEST: string
        :param frac_r200: depth of objects to plot triaxiality, in units of R200
        :type frac_r200: float
        :param obj_type: either 'dm' or 'gx', depending on what catalogue we are looking at
        :type obj_type: string"""
        print_status(rank,self.start_time,'Starting plotLocalTHist() with snap {0}'.format(self.SNAP))
        
        xyz, masses, MIN_NUMBER_PTCS = self.getXYZMasses(obj_type)
        suffix = '_{}_'.format(obj_type)
            
        if rank == 0:
            idx_cat = self.getIdxCat(obj_type)
            self.plotLocalTHistBase(xyz, masses, idx_cat, MIN_NUMBER_PTCS, self.D_LOGSTART, self.D_LOGEND, self.D_BINS, self.M_TOL, self.N_WALL, self.N_MIN, VIZ_DEST, HIST_NB_BINS, frac_r200, suffix = suffix)
            del xyz; del masses; del idx_cat
        else:
            del xyz; del masses
    
    def plotGlobalTHist(self, HIST_NB_BINS, VIZ_DEST, obj_type = 'dm'):
        """ Plot global triaxiality histogram
        
        :param HIST_NB_BINS: number of histogram bins
        :type HIST_NB_BINS: int
        :param VIZ_DEST: visualization folder
        :type VIZ_DEST: string
        :param obj_type: either 'dm' or 'gx', depending on what catalogue we are looking at
        :type obj_type: string"""
        print_status(rank,self.start_time,'Starting plotGlobalTHist() with snap {0}'.format(self.SNAP))
        
        xyz, masses, MIN_NUMBER_PTCS = self.getXYZMasses(obj_type)
        suffix = '_{}_'.format(obj_type)
            
        if rank == 0:
            idx_cat = self.getIdxCat(obj_type)
            self.plotGlobalTHistBase(xyz, masses, idx_cat, MIN_NUMBER_PTCS, self.M_TOL, self.N_WALL, self.N_MIN, VIZ_DEST, HIST_NB_BINS, suffix = suffix)
            del xyz; del masses; del idx_cat
        else:
            del xyz; del masses
        
    def plotShapeProfs(self, VIZ_DEST, obj_type = 'dm'):
        """ Draws shape profiles, also mass bin-decomposed ones
        
        :param VIZ_DEST: visualization folder
        :type VIZ_DEST: string
        :param obj_type: either 'dm' or 'gx', depending on what catalogue we are looking at
        :type obj_type: string"""
        print_status(rank,self.start_time,'Starting plotShapeProfs() with snap {0}'.format(self.SNAP))
        
        xyz, masses, MIN_NUMBER_PTCS = self.getXYZMasses(obj_type)
        suffix = '_{}_'.format(obj_type)
        
        if rank == 0:
            idx_cat = self.getIdxCat(obj_type)
            self.plotShapeProfsBase(xyz, masses, idx_cat, MIN_NUMBER_PTCS, self.D_LOGSTART, self.D_LOGEND, self.D_BINS, self.M_TOL, self.N_WALL, self.N_MIN, VIZ_DEST, suffix = suffix)
            del xyz; del masses; del idx_cat
        else:
            del xyz; del masses

    def dumpShapeCatLocal(self, CAT_DEST, obj_type = 'dm'):
        """ Dumps all relevant local shape data into ``CAT_DEST``
        
        :param CAT_DEST: catalogue folder
        :type CAT_DEST: string
        :param obj_type: either 'dm' or 'gx', depending on what catalogue we are looking at
        :type obj_type: string"""
        print_status(rank,self.start_time,'Starting dumpShapeCatLocal() with snap {0}'.format(self.SNAP))
        
        xyz, masses, MIN_NUMBER_PTCS = self.getXYZMasses(obj_type)
        suffix = '_{}_'.format(obj_type)
        
        if rank == 0:
            idx_cat = self.getIdxCat(obj_type)
            self.dumpShapeCatLocalBase(xyz, masses, idx_cat, MIN_NUMBER_PTCS, self.D_LOGSTART, self.D_LOGEND, self.D_BINS, self.M_TOL, self.N_WALL, self.N_MIN, CAT_DEST, suffix = suffix)
            del xyz; del masses; del idx_cat
        else:
            del xyz; del masses

    def dumpShapeCatGlobal(self, CAT_DEST, obj_type = 'dm'):
        """ Dumps all relevant global shape data into ``CAT_DEST``
        
        :param CAT_DEST: catalogue folder
        :type CAT_DEST: string
        :param obj_type: either 'dm' or 'gx', depending on what catalogue we are looking at
        :type obj_type: string"""
        print_status(rank,self.start_time,'Starting dumpShapeCatGlobal() with snap {0}'.format(self.SNAP))
        
        xyz, masses, MIN_NUMBER_PTCS = self.getXYZMasses(obj_type)
        suffix = '_{}_'.format(obj_type)
        
        if rank == 0:
            idx_cat = self.getIdxCat(obj_type)
            self.dumpShapeCatGlobalBase(xyz, masses, idx_cat, MIN_NUMBER_PTCS, self.M_TOL, self.N_WALL, self.N_MIN, CAT_DEST, suffix = suffix)
            del xyz; del masses; del idx_cat
        else:
            del xyz; del masses

    def dumpShapeVelCatLocal(self, CAT_DEST, obj_type = 'dm'):
        """ Dumps all relevant local velocity shape data into ``CAT_DEST``
        
        :param CAT_DEST: catalogue folder
        :type CAT_DEST: string
        :param obj_type: either 'dm' or 'gx', depending on what catalogue we are looking at
        :type obj_type: string"""
        print_status(rank,self.start_time,'Starting dumpShapeVelCatLocal() with snap {0}'.format(self.SNAP))
        
        xyz, masses, MIN_NUMBER_PTCS = self.getXYZMasses(obj_type)
        suffix = '_v{}_'.format(obj_type)
        
        if rank == 0:
            idx_cat = self.getIdxCat(obj_type)
            self.dumpShapeCatLocalBase(xyz, masses, idx_cat, MIN_NUMBER_PTCS, self.D_LOGSTART, self.D_LOGEND, self.D_BINS, self.M_TOL, self.N_WALL, self.N_MIN, CAT_DEST, suffix = suffix)
            del xyz; del masses; del idx_cat
        else:
            del xyz; del masses

    def dumpShapeVelCatGlobal(self, CAT_DEST, obj_type = 'dm'):
        """ Dumps all relevant global velocity shape data into ``CAT_DEST``
        
        :param CAT_DEST: catalogue folder
        :type CAT_DEST: string
        :param obj_type: either 'dm' or 'gx', depending on what catalogue we are looking at
        :type obj_type: string"""
        print_status(rank,self.start_time,'Starting dumpShapeVelCatGlobal() with snap {0}'.format(self.SNAP))
        
        xyz, masses, MIN_NUMBER_PTCS = self.getXYZMasses(obj_type)
        suffix = '_v{}_'.format(obj_type)
        
        if rank == 0:
            idx_cat = self.getIdxCat(obj_type)
            self.dumpShapeCatGlobalBase(xyz, masses, idx_cat, MIN_NUMBER_PTCS, self.M_TOL, self.N_WALL, self.N_MIN, CAT_DEST, suffix = suffix)
            del xyz; del masses; del idx_cat
        else:
            del xyz; del masses

    def getObjInfoLocal(self, obj_type = 'dm'):
        """ Print basic info about the objects used for local shape estimation such as number of converged objects"""
        print_status(rank,self.start_time,'Starting getObjInfoLocal() with snap {0}'.format(self.SNAP))
        
        xyz, masses, MIN_NUMBER_PTCS = self.getXYZMasses(obj_type)
        if rank == 0:
            idx_cat = self.getIdxCat(obj_type)
            self.getObjInfoLocalBase(xyz, masses, idx_cat, MIN_NUMBER_PTCS, self.D_LOGSTART, self.D_LOGEND, self.D_BINS, self.M_TOL, self.N_WALL, self.N_MIN, obj_type)
            del idx_cat
        del xyz; del masses

    def getObjInfoGlobal(self, obj_type = 'dm'):
        """ Print basic info about the objects used for global shape estimation such as number of converged objects"""
        print_status(rank,self.start_time,'Starting getObjInfoGlobal() with snap {0}'.format(self.SNAP))
        
        xyz, masses, MIN_NUMBER_PTCS = self.getXYZMasses(obj_type)
        if rank == 0:
            idx_cat = self.getIdxCat(obj_type)
            self.getObjInfoGlobalBase(idx_cat, MIN_NUMBER_PTCS, obj_type)
            del idx_cat
        del xyz; del masses

    def getObjInfoVelLocal(self, obj_type = 'dm'):
        """ Print basic info about the objects used for local velocity shape estimation such as number of converged objects"""
        print_status(rank,self.start_time,'Starting getObjInfoVelLocal() with snap {0}'.format(self.SNAP))
        
        xyz, masses, MIN_NUMBER_PTCS = self.getXYZMasses(obj_type)
        velxyz = self.getVelXYZ(obj_type)
        if rank == 0:
            idx_cat = self.getIdxCat(obj_type)
            self.getObjInfoVelLocalBase(xyz, masses, velxyz, idx_cat, MIN_NUMBER_PTCS, self.D_LOGSTART, self.D_LOGEND, self.D_BINS, self.M_TOL, self.N_WALL, self.N_MIN, obj_type)
            del idx_cat
        del xyz; del velxyz; del masses
    
    def getObjInfoVelGlobal(self, obj_type = 'dm'):
        """ Print basic info about the objects used for global velocity shape estimation such as number of converged objects"""
        print_status(rank,self.start_time,'Starting getObjInfoVelGlobal() with snap {0}'.format(self.SNAP))
        
        xyz, masses, MIN_NUMBER_PTCS = self.getXYZMasses(obj_type)
        velxyz = self.getVelXYZ(obj_type)
        if rank == 0:
            idx_cat = self.getIdxCat(obj_type)
            self.getObjInfoVelGlobalBase(idx_cat, MIN_NUMBER_PTCS, obj_type)
            del idx_cat
        del xyz; del velxyz; del masses