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
        return
        
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
        return
    
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
        return
        
    def getIdxCatGlobalBase(self, idx_cat, int MIN_NUMBER_PTCS):
        """ Calculate index catalogue corresponding to objects whose global shapes can be determined
        
        :param idx_cat: each entry of the list is a list containing indices of particles belonging to an object
        :type idx_cat: list of length N1
        :param MIN_NUMBER_PTCS: minimum number of particles for object to qualify for morphology calculation
        :type MIN_NUMBER_PTCS: int
        :return idx_cat_global: each entry of the list is a list containing indices of particles belonging to an 
            object whose global shape calculation converged, empty entry if not converged
        :rtype: list of length N1"""
        return
        
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
        return
        
    def getIdxCatVelGlobalBase(self, idx_cat, int MIN_NUMBER_PTCS):
        """ Calculate index catalogue corresponding to objects whose global velocity shapes can be determined
        
        :param idx_cat: each entry of the list is a list containing indices of particles belonging to an object
        :type idx_cat: list of length N1
        :param MIN_NUMBER_PTCS: minimum number of particles for object to qualify for morphology calculation
        :type MIN_NUMBER_PTCS: int
        :return idx_cat_global: each entry of the list is a list containing indices of particles belonging to an 
            object whose global velocity shape calculation converged, empty entry if not converged
        :rtype: list of length N1"""
        return
    
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
        return
    
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
        return
        
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
        return
    
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
        return
    
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
        return
        
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
        return
    
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
        return
    
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
        
        return
    
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
        return
    
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
        return
    
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
        return
    
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
        :return: best-fits for each object
        :rtype: (N3, n) floats, where n is the number of free parameters in the model ``method``"""
        return
        
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
        return
        
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
        return
        
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
        return
        
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
        return

    def getObjInfoGlobalBase(self, idx_cat, int MIN_NUMBER_PTCS, obj_type):
        """ Print basic info about the objects used for global shape estimation such as number of converged objects
        
        :param idx_cat: each entry of the list is a list containing indices of particles belonging to an object
        :type idx_cat: list of length N1
        :param MIN_NUMBER_PTCS: minimum number of particles for object to qualify for morphology calculation
        :type MIN_NUMBER_PTCS: int
        :param obj_type: either 'dm', 'gx' or 'unspecified', depending on what catalogue we are looking at
        :type obj_type: string"""
        return
    
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
        return
    
    def getObjInfoVelGlobalBase(self, idx_cat, int MIN_NUMBER_PTCS, obj_type):
        """ Print basic info about the objects used for global velocity shape estimation such as number of converged objects
        
        :param idx_cat: each entry of the list is a list containing indices of particles belonging to an object
        :type idx_cat: list of length N1
        :param MIN_NUMBER_PTCS: minimum number of particles for object to qualify for morphology calculation
        :type MIN_NUMBER_PTCS: int
        :param obj_type: either 'dm', 'gx' or 'unspecified', depending on what catalogue we are looking at
        :type obj_type: string"""
        return

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
        return
    
    def getMassesCenters(self):
        """ Calculate total mass and centers of objects
        
        :return centers, m: centers and masses
        :rtype: (N,3) and (N,) floats"""
        return
    
    def getDensProfsDirectBinning(self, ROverR200):
        """ Get direct-binning-based density profiles
        
        :param ROverR200: normalized radii at which ``dens_profs`` are defined
        :type ROverR200: (r_res,) floats
        :return: density profiles
        :rtype: (N2, r_res) floats"""
        return

    def getDensProfsKernelBased(self, ROverR200):
        """ Get kernel-based density profiles
        
        :param ROverR200: normalized radii at which ``dens_profs`` are defined
        :type ROverR200: (r_res,) floats
        :return: density profiles
        :rtype: (N2, r_res) floats"""
        return
        
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
        return
        
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
        return
        
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
        return
      
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
        return
    
    def getIdxCatGlobal(self):
        """ Fetch global shape index catalogue
        
        :return idx_cat_global: list of indices defining the objects
        :type idx_cat_global: list of length N1, each consisting of a list of int indices"""
        return
        
    def getShapeCatLocal(self):
        """ Get all relevant local shape data
        
        :return: d, q, s, minor, inter, major, obj_center, obj_m, succeeded (list of indices of converged objects)
        :rtype: 3 x (number_of_objs, D_BINS+1) float arrays, 
            3 x (number_of_objs, D_BINS+1, 3) float arrays, 
            (number_of_objs,3) float array, (number_of_objs,) float array,
            list of length N3
        """
        return
    
    def getShapeCatGlobal(self):
        """ Get all relevant global shape data
        
        :return: d, q, s, minor, inter, major, obj_center, obj_m
        :rtype: 3 x (number_of_objs, D_BINS+1) float arrays, 
            3 x (number_of_objs, D_BINS+1, 3) float arrays, 
            (number_of_objs,3) float array, (number_of_objs,) float array,
        """
        return
    
    def vizLocalShapes(self, obj_numbers, VIZ_DEST):
        """ Visualize local shape of objects with numbers ``obj_numbers``
        
        :param obj_numbers: list of object indices for which to visualize local shapes
        :type obj_numbers: list of int
        :param VIZ_DEST: visualization folder
        :type VIZ_DEST: strings"""
        return
        
    def vizGlobalShapes(self, obj_numbers, VIZ_DEST):
        """ Visualize global shape of objects with numbers ``obj_numbers``
        
        :param obj_numbers: list of object indices for which to visualize global shapes
        :type obj_numbers: list of ints
        :param VIZ_DEST: visualization folder
        :type VIZ_DEST: string"""
        return
    
    def plotGlobalEpsHist(self, HIST_NB_BINS, VIZ_DEST):
        """ Plot global ellipticity histogram
        
        :param HIST_NB_BINS: number of histogram bins
        :type HIST_NB_BINS: int
        :param VIZ_DEST: visualization folder
        :type VIZ_DEST: string"""
        return

    def plotLocalEpsHist(self, frac_r200, HIST_NB_BINS, VIZ_DEST):
        """ Plot local ellipticity histogram at depth ``frac_r200``
        
        :param frac_r200: depth of objects to plot ellipticity, in units of R200
        :type frac_r200: float
        :param HIST_NB_BINS: number of histogram bins
        :type HIST_NB_BINS: int
        :param VIZ_DEST: visualization folder
        :type VIZ_DEST: string"""
        return
    
    def plotLocalTHist(self, HIST_NB_BINS, VIZ_DEST, frac_r200):
        """ Plot local triaxiality histogram at depth ``frac_r200``
        
        :param HIST_NB_BINS: number of histogram bins
        :type HIST_NB_BINS: int
        :param VIZ_DEST: visualization folder
        :type VIZ_DEST: string
        :param frac_r200: depth of objects to plot triaxiality, in units of R200
        :type frac_r200: float"""
        return
    
    def plotGlobalTHist(self, HIST_NB_BINS, VIZ_DEST):
        """ Plot global triaxiality histogram
        
        :param HIST_NB_BINS: number of histogram bins
        :type HIST_NB_BINS: int
        :param VIZ_DEST: visualization folder
        :type VIZ_DEST: string"""
        return
    
    def plotShapeProfs(self, VIZ_DEST):
        """ Draws shape profiles, also mass bin-decomposed ones
        
        :param VIZ_DEST: visualization folder
        :type VIZ_DEST: string"""
        return
    
    def dumpShapeCatLocal(self, CAT_DEST):
        """ Dumps all relevant local shape data into ``CAT_DEST``
        
        :param CAT_DEST: catalogue folder
        :type CAT_DEST: string"""
        return
    
    def dumpShapeCatGlobal(self, CAT_DEST):
        """ Dumps all relevant global shape data into ``CAT_DEST``
        
        :param CAT_DEST: catalogue folder
        :type CAT_DEST: string"""
        return
    
    def getObjInfoLocal(self):
        """ Print basic info about the objects used for local shape estimation such as number of converged objects"""
        return
    
    def getObjInfoGlobal(self):
        """ Print basic info about the objects used for global shape estimation such as number of converged objects"""
        return
    
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
        return
        
    def getVelXYZ(self, obj_type = 'dm'):
        """ Retrieve velocities of DM/gx
        
        :param obj_type: either 'dm' or 'gx', depending on what catalogue we are looking at
        :type obj_type: string
        :return velxyz: velocity array
        :rtype: (N2,3) floats"""
        return
        
    def getIdxCat(self, obj_type = 'dm'):
        """ Fetch catalogue
        
        :param obj_type: either 'dm' or 'gx', depending on what catalogue we are looking at
        :type obj_type: string
        :return cat: list of indices defining the objects
        :rtype: list of length N1, each consisting of a list of int indices"""
        
        return
    
    def getMassesCenters(self, obj_type = 'dm'):
        """ Calculate total mass and centers of objects
        
        :param obj_type: either 'dm' or 'gx', depending on what catalogue we are looking at
        :type obj_type: string
        :return centers, m: centers and masses
        :rtype: (N,3) and (N,) floats"""
        return
    
    def getDensProfsDirectBinning(self, ROverR200, obj_type = 'dm'):
        """ Get direct-binning-based density profiles
        
        :param ROverR200: normalized radii at which ``dens_profs`` are defined
        :type ROverR200: (r_res,) floats
        :param obj_type: either 'dm' or 'gx', depending on what catalogue we are looking at
        :type obj_type: string
        :return: density profiles
        :rtype: (N2, r_res) floats"""
        return

    def getDensProfsKernelBased(self, ROverR200, obj_type = 'dm'):
        """ Get kernel-based density profiles
        
        :param ROverR200: normalized radii at which ``dens_profs`` are defined
        :type ROverR200: (r_res,) floats
        :param obj_type: either 'dm' or 'gx', depending on what catalogue we are looking at
        :type obj_type: string
        :return: density profiles
        :rtype: (N2, r_res) floats"""
        return
        
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
        return
        
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
        return
        
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
        return

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
        return
    
    def getIdxCatGlobal(self, obj_type = 'dm'):
        """ Fetch global shape index catalogue
        
        :param obj_type: either 'dm' or 'gx', depending on what catalogue we are looking at
        :type obj_type: string
        :return idx_cat_global: list of indices defining the objects
        :type idx_cat_global: list of length N1, each consisting of a list of int indices"""
        return
    
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
        return
    
    def getShapeCatGlobal(self, obj_type = 'dm'):
        """ Get all relevant global shape data
        
        :param obj_type: either 'dm' or 'gx', depending on what catalogue we are looking at
        :type obj_type: string
        :return: d, q, s, minor, inter, major, obj_center, obj_m
        :rtype: 3 x (number_of_objs, D_BINS+1) float arrays, 
            3 x (number_of_objs, D_BINS+1, 3) float arrays, 
            (number_of_objs,3) float array, (number_of_objs,) float array,
        """
        return
        
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
        return
    
    def getShapeCatVelGlobal(self, obj_type = 'dm'):
        """ Get all relevant global velocity shape data
        
        :param obj_type: either 'dm' or 'gx', depending on what catalogue we are looking at
        :type obj_type: string
        :return: d, q, s, minor, inter, major, obj_center, obj_m
        :rtype: 3 x (number_of_objs, D_BINS+1) float arrays, 
            3 x (number_of_objs, D_BINS+1, 3) float arrays, 
            (number_of_objs,3) float array, (number_of_objs,) float array,
        """
        return
    
    def vizLocalShapes(self, obj_numbers, VIZ_DEST, obj_type = 'dm'):
        """ Visualize local shape of objects with numbers ``obj_numbers``
        
        :param obj_numbers: list of object indices for which to visualize local shapes
        :type obj_numbers: list of int
        :param VIZ_DEST: visualization folder
        :type VIZ_DEST: strings
        :param obj_type: either 'dm' or 'gx', depending on what catalogue we are looking at
        :type obj_type: string"""
        return
        
    def vizGlobalShapes(self, obj_numbers, VIZ_DEST, obj_type = 'dm'):
        """ Visualize global shape of objects with numbers ``obj_numbers``
        
        :param obj_numbers: list of object indices for which to visualize global shapes
        :type obj_numbers: list of ints
        :param VIZ_DEST: visualization folder
        :type VIZ_DEST: string
        :param obj_type: either 'dm' or 'gx', depending on what catalogue we are looking at
        :type obj_type: string"""
        return
    
    def plotGlobalEpsHist(self, HIST_NB_BINS, VIZ_DEST, obj_type = 'dm'):
        """ Plot global ellipticity histogram
        
        :param HIST_NB_BINS: number of histogram bins
        :type HIST_NB_BINS: int
        :param VIZ_DEST: visualization folder
        :type VIZ_DEST: string
        :param obj_type: either 'dm' or 'gx', depending on what catalogue we are looking at
        :type obj_type: string"""
        return
    
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
        return
    
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
        return
    
    def plotGlobalTHist(self, HIST_NB_BINS, VIZ_DEST, obj_type = 'dm'):
        """ Plot global triaxiality histogram
        
        :param HIST_NB_BINS: number of histogram bins
        :type HIST_NB_BINS: int
        :param VIZ_DEST: visualization folder
        :type VIZ_DEST: string
        :param obj_type: either 'dm' or 'gx', depending on what catalogue we are looking at
        :type obj_type: string"""
        return
        
    def plotShapeProfs(self, VIZ_DEST, obj_type = 'dm'):
        """ Draws shape profiles, also mass bin-decomposed ones
        
        :param VIZ_DEST: visualization folder
        :type VIZ_DEST: string
        :param obj_type: either 'dm' or 'gx', depending on what catalogue we are looking at
        :type obj_type: string"""
        return

    def dumpShapeCatLocal(self, CAT_DEST, obj_type = 'dm'):
        """ Dumps all relevant local shape data into ``CAT_DEST``
        
        :param CAT_DEST: catalogue folder
        :type CAT_DEST: string
        :param obj_type: either 'dm' or 'gx', depending on what catalogue we are looking at
        :type obj_type: string"""
        return

    def dumpShapeCatGlobal(self, CAT_DEST, obj_type = 'dm'):
        """ Dumps all relevant global shape data into ``CAT_DEST``
        
        :param CAT_DEST: catalogue folder
        :type CAT_DEST: string
        :param obj_type: either 'dm' or 'gx', depending on what catalogue we are looking at
        :type obj_type: string"""
        return

    def dumpShapeVelCatLocal(self, CAT_DEST, obj_type = 'dm'):
        """ Dumps all relevant local velocity shape data into ``CAT_DEST``
        
        :param CAT_DEST: catalogue folder
        :type CAT_DEST: string
        :param obj_type: either 'dm' or 'gx', depending on what catalogue we are looking at
        :type obj_type: string"""
        return

    def dumpShapeVelCatGlobal(self, CAT_DEST, obj_type = 'dm'):
        """ Dumps all relevant global velocity shape data into ``CAT_DEST``
        
        :param CAT_DEST: catalogue folder
        :type CAT_DEST: string
        :param obj_type: either 'dm' or 'gx', depending on what catalogue we are looking at
        :type obj_type: string"""
        return

    def getObjInfoLocal(self, obj_type = 'dm'):
        """ Print basic info about the objects used for local shape estimation such as number of converged objects"""
        return

    def getObjInfoGlobal(self, obj_type = 'dm'):
        """ Print basic info about the objects used for global shape estimation such as number of converged objects"""
        return

    def getObjInfoVelLocal(self, obj_type = 'dm'):
        """ Print basic info about the objects used for local velocity shape estimation such as number of converged objects"""
        return
    
    def getObjInfoVelGlobal(self, obj_type = 'dm'):
        """ Print basic info about the objects used for global velocity shape estimation such as number of converged objects"""
        return