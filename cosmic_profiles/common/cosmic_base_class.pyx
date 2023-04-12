#cython: language_level=3
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
cimport cython
from cosmic_profiles.common.python_routines import print_status, isValidSelection, getSubSetIdxCat
from cosmic_profiles.common import config
from cosmic_profiles.shape_profs.shape_profs_tools import getLocalTHist, getGlobalTHist, getShapeProfs
from cosmic_profiles.dens_profs.dens_profs_tools import fitDensProfHelper, drawDensProfs
from cosmic_profiles.shape_profs.shape_profs_algos import calcMorphLocal, calcMorphGlobal, calcMorphLocalVelDisp, calcMorphGlobalVelDisp
from cosmic_profiles.dens_profs.dens_profs_algos import calcMassesCenters, calcDensProfsSphDirectBinning, calcDensProfsEllDirectBinning, calcDensProfsKernelBased
import time
import subprocess
import sys
import inspect
import os
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

@cython.embedsignature(True)
cdef class CosmicBase:
    """ Parent class governing high-level cosmic shape calculations
    
    Its public methods are ``_getMassesCentersBase()``, 
    ``_getShapeCatLocalBase()``, ``_getShapeCatGlobalBase()``, ``_getShapeCatVelLocalBase()``, 
    ``_getShapeCatVelGlobalBase()``, ``_dumpShapeCatLocalBase()``, ``_dumpShapeCatGlobalBase()``,
    ``_dumpShapeCatVelLocalBase()``, ``_dumpShapeCatVelGlobalBase()``, ``_plotShapeProfsBase()``,
    ``_plotLocalTHistBase()``, ``_plotGlobalTHistBase()``, ``_getDensProfsBestFitsBase()``,
    ``_getConcentrationsBase()``, ``_getDensProfsSphDirectBinningBase()``, ``_getDensProfsEllDirectBinningBase()``,
    ``_getDensProfsKernelBasedBase()``, ``_getObjInfoBase()``"""
    
    def __init__(self, str SNAP, float L_BOX, int MIN_NUMBER_PTCS, str CENTER, str VIZ_DEST, str CAT_DEST, str SUFFIX):
        """
        :param SNAP: snapshot identifier, e.g. '024'
        :type SNAP: string
        :param L_BOX: simulation box side length in config.InUnitLength_in_cm
        :type L_BOX: float
        :param MIN_NUMBER_PTCS: minimum number of particles for object to qualify for morphology calculation
        :type MIN_NUMBER_PTCS: int
        :param CENTER: shape quantities will be calculated with respect to CENTER = 'mode' (point of highest density)
            or 'com' (center of mass) of each halo
        :type CENTER: str
        :param VIZ_DEST: visualization folder
        :type VIZ_DEST: string
        :param CAT_DEST: catalogue destination
        :type CAT_DEST: string
        :param SUFFIX: either '_dm_' or '_gx_' or '' (latter for CosmicProfsDirect)
        :type SUFFIX: string"""
        self.SNAP = SNAP
        l_curr_over_target = config.InUnitLength_in_cm/3.085678e24
        self.L_BOX = L_BOX # self.L_BOX must be in units of 3.085678e24 cm (Mpc/h)
        self.CENTER = CENTER
        self.VIZ_DEST = VIZ_DEST
        self.CAT_DEST = CAT_DEST
        self.MIN_NUMBER_PTCS = MIN_NUMBER_PTCS
        self.start_time = time.time()
        self.SAFE = 6 # in units of 3.085678e24 cm
        self.MASS_UNIT = 1e10
        self.r200 = None
        self.SUFFIX = SUFFIX
        
    def _getMassesCentersBase(self, float[:,:] xyz, float[:] masses, int[:] idx_cat, int[:] obj_size):
        """ Calculate total mass and centers of objects
        
        :param xyz: positions of all simulation particles in Mpc/h
        :type xyz: (N2,3) floats, N2 >> N1
        :param masses: masses of all simulation particles in 10^10*M_sun/h
        :type masses: (N2,) floats
        :param idx_cat: contains indices of particles belonging to an object
        :type idx_cat: (N3) integers
        :param obj_size: indicates how many particles are in each object
        :type obj_size: (N1,) integers
        :return centers, m: centers in Mpc/h and masses in 10^10*M_sun/h
        :rtype: (N,3) and (N,) floats"""
        centers, m = calcMassesCenters(xyz.base, masses.base, idx_cat.base, obj_size.base, self.L_BOX, self.CENTER)
        return centers, m
    
    def _getShapeCatLocalBase(self, float[:,:] xyz, float[:] masses, float[:] r200, int[:] idx_cat, int[:] obj_size, float D_LOGSTART, float D_LOGEND, int D_BINS, float IT_TOL, int IT_WALL, int IT_MIN, bint reduced, bint shell_based, str suffix):
        """ Get all relevant local shape data
        
        :param xyz: positions of all simulation particles in Mpc/h
        :type xyz: (N2,3) floats, N2 >> N1
        :param masses: masses of all simulation particles in 10^10*M_sun/h
        :type masses: (N2,) floats
        :param r200: each entry gives the R_200 radius of the parent halo in Mpc/h
        :type r200: (N1,) floats
        :param idx_cat: contains indices of particles belonging to an object
        :type idx_cat: (N3) integers
        :param obj_size: indicates how many particles are in each object
        :type obj_size: (N1,) integers
        :param D_LOGSTART: logarithm of minimum ellipsoidal radius of interest, in units of R200 of parent halo
        :type D_LOGSTART: int
        :param D_LOGEND: logarithm of maximum ellipsoidal radius of interest, in units of R200 of parent halo
        :type D_LOGEND: int
        :param D_BINS: number of ellipsoidal radii of interest minus 1 (i.e. number of bins)
        :type D_BINS: int
        :param IT_TOL: convergence tolerance, eigenvalue fractions must differ by less than ``IT_TOL``
            for iteration to stop
        :type IT_TOL: float
        :param IT_WALL: maximum permissible number of iterations
        :type IT_WALL: float
        :param IT_MIN: minimum number of particles (DM or star particle) in any iteration; 
            if undercut, shape is unclassified
        :type IT_MIN: int
        :param reduced: whether or not reduced shape tensor (1/r^2 factor)
        :type reduced: boolean
        :param shell_based: whether shell-based or ellipsoid-based algorithm should be run
        :type shell_based: boolean
        :param suffix: suffix for file names
        :type suffix: string
        :return: d in Mpc/h, q, s, minor, inter, major, obj_centers in units of Mpc/h,
            obj_masses in units of 10^10*M_sun/h
        :rtype: 3 x (number_of_objs, D_BINS+1) float arrays, 
            3 x (number_of_objs, D_BINS+1, 3) float arrays, 
            (number_of_objs,3) float array, (number_of_objs,) float array
        """
        if rank == 0:
            
            if os.path.exists('{0}/d_local{1}{2}.txt'.format(self.CAT_DEST, suffix, self.SNAP)):
                d = np.loadtxt('{0}/d_local{1}{2}.txt'.format(self.CAT_DEST, suffix, self.SNAP))
                q = np.loadtxt('{0}/q_local{1}{2}.txt'.format(self.CAT_DEST, suffix, self.SNAP))
                s = np.loadtxt('{0}/s_local{1}{2}.txt'.format(self.CAT_DEST, suffix, self.SNAP))
                minor = np.loadtxt('{0}/minor_local{1}{2}.txt'.format(self.CAT_DEST, suffix, self.SNAP))
                inter = np.loadtxt('{0}/inter_local{1}{2}.txt'.format(self.CAT_DEST, suffix, self.SNAP))
                major = np.loadtxt('{0}/major_local{1}{2}.txt'.format(self.CAT_DEST, suffix, self.SNAP))
                obj_masses = np.loadtxt('{0}/m_local{1}{2}.txt'.format(self.CAT_DEST, suffix, self.SNAP))
                obj_centers = np.loadtxt('{0}/centers_local{1}{2}.txt'.format(self.CAT_DEST, suffix, self.SNAP))
                
                l_curr_over_target = 3.085678e24/config.OutUnitLength_in_cm
                m_curr_over_target = 1.989e43/config.OutUnitMass_in_g
                d = d/l_curr_over_target
                obj_centers = obj_centers/l_curr_over_target
                obj_masses = obj_masses/m_curr_over_target
                if d.shape[0] != 0 and d.ndim == 2:
                    minor = minor.reshape(minor.shape[0], -1, 3)
                    inter = inter.reshape(inter.shape[0], -1, 3)
                    major = major.reshape(major.shape[0], -1, 3)
                elif d.ndim == 1:
                    minor = minor.reshape(1, -1, 3)
                    inter = inter.reshape(1, -1, 3)
                    major = major.reshape(1, -1, 3)
                    obj_masses = np.array([obj_masses])
                    obj_centers = obj_centers.reshape(1, 3)
                    d = d.reshape(1, -1)
                    q = q.reshape(1, -1)
                    s = s.reshape(1, -1)
                else:
                    minor = np.array([])
                    inter = np.array([])
                    major = np.array([])
            
            else:
                d, q, s, minor, inter, major, obj_centers, obj_masses = calcMorphLocal(xyz.base, masses.base, r200.base, idx_cat.base, obj_size.base, self.L_BOX, D_LOGSTART, D_LOGEND, D_BINS, IT_TOL, IT_WALL, IT_MIN, self.CENTER, reduced, shell_based)
            return d, q, s, minor, inter, major, obj_centers, obj_masses
        else:
            return None, None, None, None, None, None, None, None
    
    def _getShapeCatGlobalBase(self, float[:,:] xyz, float[:] masses, float[:] r200, int[:] idx_cat, int[:] obj_size, float IT_TOL, int IT_WALL, int IT_MIN, bint reduced, str suffix):
        """ Get all relevant global shape data
        
        :param xyz: positions of all simulation particles in Mpc/h
        :type xyz: (N2,3) floats, N2 >> N1
        :param masses: masses of all simulation particles in 10^10*M_sun/h
        :type masses: (N2,) floats
        :param r200: each entry gives the R_200 radius of the parent halo in Mpc/h
        :type r200: (N1,) floats
        :param idx_cat: contains indices of particles belonging to an object
        :type idx_cat: (N3) integers
        :param obj_size: indicates how many particles are in each object
        :type obj_size: (N1,) integers
        :param IT_TOL: convergence tolerance, eigenvalue fractions must differ by less than ``IT_TOL``
            for iteration to stop
        :type IT_TOL: float
        :param IT_WALL: maximum permissible number of iterations
        :type IT_WALL: float
        :param IT_MIN: minimum number of particles (DM or star particle) in any iteration; 
            if undercut, shape is unclassified
        :type IT_MIN: int
        :param reduced: whether or not reduced shape tensor (1/r^2 factor)
        :type reduced: boolean
        :param suffix: suffix for file names
        :type suffix: string
        :return: d in Mpc/h, q, s, minor, inter, major, obj_centers in units of Mpc/h,
            obj_masses in units of 10^10*M_sun/h
        :rtype: 3 x (number_of_objs,) float arrays, 
            3 x (number_of_objs, 3) float arrays, 
            (number_of_objs, 3) float array, (number_of_objs,) float array"""
        if rank == 0:
            
            if os.path.exists('{0}/d_global{1}{2}.txt'.format(self.CAT_DEST, suffix, self.SNAP)):
                
                d = np.loadtxt('{0}/d_global{1}{2}.txt'.format(self.CAT_DEST, suffix, self.SNAP))
                q = np.loadtxt('{0}/q_global{1}{2}.txt'.format(self.CAT_DEST, suffix, self.SNAP))
                s = np.loadtxt('{0}/s_global{1}{2}.txt'.format(self.CAT_DEST, suffix, self.SNAP))
                minor = np.loadtxt('{0}/minor_global{1}{2}.txt'.format(self.CAT_DEST, suffix, self.SNAP))
                inter = np.loadtxt('{0}/inter_global{1}{2}.txt'.format(self.CAT_DEST, suffix, self.SNAP))
                major = np.loadtxt('{0}/major_global{1}{2}.txt'.format(self.CAT_DEST, suffix, self.SNAP))
                obj_masses = np.loadtxt('{0}/m_global{1}{2}.txt'.format(self.CAT_DEST, suffix, self.SNAP))
                obj_centers = np.loadtxt('{0}/centers_global{1}{2}.txt'.format(self.CAT_DEST, suffix, self.SNAP))
                
                l_curr_over_target = 3.085678e24/config.OutUnitLength_in_cm
                m_curr_over_target = 1.989e43/config.OutUnitMass_in_g
                d = d/l_curr_over_target
                obj_centers = obj_centers/l_curr_over_target
                obj_masses = obj_masses/m_curr_over_target
                if len(d.shape) != 0 and d.shape[0] != 0 and d.ndim == 1:
                    minor = minor.reshape(minor.shape[0], -1, 3)
                    inter = inter.reshape(inter.shape[0], -1, 3)
                    major = major.reshape(major.shape[0], -1, 3)
                    d = d.reshape(-1, 1)
                    q = q.reshape(-1, 1)
                    s = s.reshape(-1, 1)
                elif d.ndim == 0: # 1 object. Even in case of 0 objects, ndim will be 1
                    minor = minor.reshape(1, -1, 3)
                    inter = inter.reshape(1, -1, 3)
                    major = major.reshape(1, -1, 3)
                    obj_masses = np.array([obj_masses])
                    obj_centers = obj_centers.reshape(1, 3)
                    d = d.reshape(1, 1)
                    q = q.reshape(1, 1)
                    s = s.reshape(1, 1)
                else:
                    assert d.size == 0
                    minor = np.array([])
                    inter = np.array([])
                    major = np.array([])
            
            else:
                d, q, s, minor, inter, major, obj_centers, obj_masses = calcMorphGlobal(xyz.base, masses.base, r200.base, idx_cat.base, obj_size.base, self.L_BOX, IT_TOL, IT_WALL, IT_MIN, self.CENTER, self.SAFE, reduced)
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
            return d, q, s, minor, inter, major, obj_centers, obj_masses
        else:
            return None, None, None, None, None, None, None, None
        
    def _getShapeCatVelLocalBase(self, float[:,:] xyz, float[:,:] velxyz, float[:] masses, float[:] r200, int[:] idx_cat, int[:] obj_size, float D_LOGSTART, float D_LOGEND, int D_BINS, float IT_TOL, int IT_WALL, int IT_MIN, bint reduced, bint shell_based, str suffix):
        """ Get all relevant local velocity shape data
        
        :param xyz: positions of all simulation particles in Mpc/h
        :type xyz: (N2,3) floats, N2 >> N1
        :param velxyz: velocity array in km/s
        :type velxyz: (N2,3) floats
        :param masses: masses of all simulation particles in 10^10*M_sun/h
        :type masses: (N2,) floats
        :param r200: each entry gives the R_200 radius of the parent halo in Mpc/h
        :type r200: (N1,) floats
        :param idx_cat: contains indices of particles belonging to an object
        :type idx_cat: (N3) integers
        :param obj_size: indicates how many particles are in each object
        :type obj_size: (N1,) integers
        :param D_LOGSTART: logarithm of minimum ellipsoidal radius of interest, in units of R200 of parent halo
        :type D_LOGSTART: int
        :param D_LOGEND: logarithm of maximum ellipsoidal radius of interest, in units of R200 of parent halo
        :type D_LOGEND: int
        :param D_BINS: number of ellipsoidal radii of interest minus 1 (i.e. number of bins)
        :type D_BINS: int
        :param IT_TOL: convergence tolerance, eigenvalue fractions must differ by less than ``IT_TOL``
            for iteration to stop
        :type IT_TOL: float
        :param IT_WALL: maximum permissible number of iterations
        :type IT_WALL: float
        :param IT_MIN: minimum number of particles (DM or star particle) in any iteration; 
            if undercut, shape is unclassified
        :type IT_MIN: int
        :param reduced: whether or not reduced shape tensor (1/r^2 factor)
        :type reduced: boolean
        :param shell_based: whether shell-based or ellipsoid-based algorithm should be run
        :type shell_based: boolean
        :param suffix: suffix for file names
        :type suffix: string
        :return: d in Mpc/h, q, s, minor, inter, major, obj_centers in units of Mpc/h,
            obj_masses in units of 10^10*M_sun/h
        :rtype: 3 x (number_of_objs, D_BINS+1) float arrays, 
            3 x (number_of_objs, D_BINS+1, 3) float arrays, 
            (number_of_objs,3) float array, (number_of_objs,) float array
        """
        if rank == 0:
            
            if os.path.exists('{0}/d_local{1}{2}.txt'.format(self.CAT_DEST, suffix, self.SNAP)):
                d = np.loadtxt('{0}/d_local{1}{2}.txt'.format(self.CAT_DEST, suffix, self.SNAP))
                q = np.loadtxt('{0}/q_local{1}{2}.txt'.format(self.CAT_DEST, suffix, self.SNAP))
                s = np.loadtxt('{0}/s_local{1}{2}.txt'.format(self.CAT_DEST, suffix, self.SNAP))
                minor = np.loadtxt('{0}/minor_local{1}{2}.txt'.format(self.CAT_DEST, suffix, self.SNAP))
                inter = np.loadtxt('{0}/inter_local{1}{2}.txt'.format(self.CAT_DEST, suffix, self.SNAP))
                major = np.loadtxt('{0}/major_local{1}{2}.txt'.format(self.CAT_DEST, suffix, self.SNAP))
                obj_masses = np.loadtxt('{0}/m_local{1}{2}.txt'.format(self.CAT_DEST, suffix, self.SNAP))
                obj_centers = np.loadtxt('{0}/centers_local{1}{2}.txt'.format(self.CAT_DEST, suffix, self.SNAP))
                
                l_curr_over_target = 3.085678e24/config.OutUnitLength_in_cm
                m_curr_over_target = 1.989e43/config.OutUnitMass_in_g
                d = d/l_curr_over_target
                obj_centers = obj_centers/l_curr_over_target
                obj_masses = obj_masses/m_curr_over_target
                if d.shape[0] != 0 and d.ndim == 2:
                    minor = minor.reshape(minor.shape[0], -1, 3)
                    inter = inter.reshape(inter.shape[0], -1, 3)
                    major = major.reshape(major.shape[0], -1, 3)
                elif d.ndim == 1:
                    minor = minor.reshape(1, -1, 3)
                    inter = inter.reshape(1, -1, 3)
                    major = major.reshape(1, -1, 3)
                    obj_masses = np.array([obj_masses])
                    obj_centers = obj_centers.reshape(1, 3)
                    d = d.reshape(1, -1)
                    q = q.reshape(1, -1)
                    s = s.reshape(1, -1)
                else:
                    minor = np.array([])
                    inter = np.array([])
                    major = np.array([])
            
            else:
                d, q, s, minor, inter, major, obj_centers, obj_masses = calcMorphLocalVelDisp(xyz.base, velxyz.base, masses.base, r200.base, idx_cat.base, obj_size.base, self.L_BOX, D_LOGSTART, D_LOGEND, D_BINS, IT_TOL, IT_WALL, IT_MIN, self.CENTER, reduced, shell_based)
                print_status(rank, self.start_time, "Finished calcMorphLocalVelDisp()")
            return d, q, s, minor, inter, major, obj_centers, obj_masses
        else:
            return None, None, None, None, None, None, None, None
    
    def _getShapeCatVelGlobalBase(self, float[:,:] xyz, float[:,:] velxyz, float[:] masses, float[:] r200, int[:] idx_cat, int[:] obj_size, float IT_TOL, int IT_WALL, int IT_MIN, bint reduced, str suffix):
        """ Get all relevant global velocity shape data
        
        :param xyz: positions of all simulation particles in Mpc/h
        :type xyz: (N2,3) floats, N2 >> N1
        :param velxyz: velocity array in km/s
        :type velxyz: (N2,3) floats
        :param masses: masses of all simulation particles in 10^10*M_sun/h
        :type masses: (N2,) floats
        :param r200: each entry gives the R_200 radius of the parent halo in Mpc/h
        :type r200: (N1,) floats
        :param idx_cat: contains indices of particles belonging to an object
        :type idx_cat: (N3) integers
        :param obj_size: indicates how many particles are in each object
        :type obj_size: (N1,) integers
        :param IT_TOL: convergence tolerance, eigenvalue fractions must differ by less than ``IT_TOL``
            for iteration to stop
        :type IT_TOL: float
        :param IT_WALL: maximum permissible number of iterations
        :type IT_WALL: float
        :param IT_MIN: minimum number of particles (DM or star particle) in any iteration; 
            if undercut, shape is unclassified
        :type IT_MIN: int
        :param reduced: whether or not reduced shape tensor (1/r^2 factor)
        :type reduced: boolean
        :param suffix: suffix for file names
        :type suffix: string
        :return: d in Mpc/h, q, s, minor, inter, major, obj_centers in units of Mpc/h,
            obj_masses in units of 10^10*M_sun/h
        :rtype: 3 x (number_of_objs,) float arrays, 
            3 x (number_of_objs, 3) float arrays, 
            (number_of_objs, 3) float array, (number_of_objs,) float array"""
        if rank == 0:
            
            if os.path.exists('{0}/d_global{1}{2}.txt'.format(self.CAT_DEST, suffix, self.SNAP)):
                
                d = np.loadtxt('{0}/d_global{1}{2}.txt'.format(self.CAT_DEST, suffix, self.SNAP))
                q = np.loadtxt('{0}/q_global{1}{2}.txt'.format(self.CAT_DEST, suffix, self.SNAP))
                s = np.loadtxt('{0}/s_global{1}{2}.txt'.format(self.CAT_DEST, suffix, self.SNAP))
                minor = np.loadtxt('{0}/minor_global{1}{2}.txt'.format(self.CAT_DEST, suffix, self.SNAP))
                inter = np.loadtxt('{0}/inter_global{1}{2}.txt'.format(self.CAT_DEST, suffix, self.SNAP))
                major = np.loadtxt('{0}/major_global{1}{2}.txt'.format(self.CAT_DEST, suffix, self.SNAP))
                obj_masses = np.loadtxt('{0}/m_global{1}{2}.txt'.format(self.CAT_DEST, suffix, self.SNAP))
                obj_centers = np.loadtxt('{0}/centers_global{1}{2}.txt'.format(self.CAT_DEST, suffix, self.SNAP))
                
                l_curr_over_target = 3.085678e24/config.OutUnitLength_in_cm
                m_curr_over_target = 1.989e43/config.OutUnitMass_in_g
                d = d/l_curr_over_target
                obj_centers = obj_centers/l_curr_over_target
                obj_masses = obj_masses/m_curr_over_target
                if len(d.shape) != 0 and d.shape[0] != 0 and d.ndim == 1:
                    minor = minor.reshape(minor.shape[0], -1, 3)
                    inter = inter.reshape(inter.shape[0], -1, 3)
                    major = major.reshape(major.shape[0], -1, 3)
                    d = d.reshape(-1, 1)
                    q = q.reshape(-1, 1)
                    s = s.reshape(-1, 1)
                elif d.ndim == 0: # 1 object. Even in case of 0 objects, ndim will be 1
                    minor = minor.reshape(1, -1, 3)
                    inter = inter.reshape(1, -1, 3)
                    major = major.reshape(1, -1, 3)
                    obj_masses = np.array([obj_masses])
                    obj_centers = obj_centers.reshape(1, 3)
                    d = d.reshape(1, 1)
                    q = q.reshape(1, 1)
                    s = s.reshape(1, 1)
                else:
                    assert d.size == 0
                    minor = np.array([])
                    inter = np.array([])
                    major = np.array([])
            
            else:
                d, q, s, minor, inter, major, obj_centers, obj_masses = calcMorphGlobalVelDisp(xyz.base, velxyz.base, masses.base, r200.base, idx_cat.base, obj_size.base, self.L_BOX, IT_TOL, IT_WALL, IT_MIN, self.CENTER, self.SAFE, reduced)
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
            return d, q, s, minor, inter, major, obj_centers, obj_masses
        else:
            return None, None, None, None, None, None, None, None
    
    def _dumpShapeCatLocalBase(self, float[:,:] xyz, float[:] masses, float[:] r200, int[:] idx_cat, int[:] obj_size, float D_LOGSTART, float D_LOGEND, int D_BINS, float IT_TOL, int IT_WALL, int IT_MIN, str suffix, bint reduced, bint shell_based):
        """ Dumps all relevant local shape data into ``CAT_DEST``
        
        :param xyz: positions of all simulation particles in Mpc/h
        :type xyz: (N2,3) floats, N2 >> N1
        :param masses: masses of all simulation particles in 10^10*M_sun/h
        :type masses: (N2,) floats
        :param r200: each entry gives the R_200 radius of the parent halo in Mpc/h
        :type r200: (N1,) floats
        :param idx_cat: contains indices of particles belonging to an object
        :type idx_cat: (N3) integers
        :param obj_size: indicates how many particles are in each object
        :type obj_size: (N1,) integers
        :param D_LOGSTART: logarithm of minimum ellipsoidal radius of interest, in units of R200 of parent halo
        :type D_LOGSTART: int
        :param D_LOGEND: logarithm of maximum ellipsoidal radius of interest, in units of R200 of parent halo
        :type D_LOGEND: int
        :param D_BINS: number of ellipsoidal radii of interest minus 1 (i.e. number of bins)
        :type D_BINS: int
        :param IT_TOL: convergence tolerance, eigenvalue fractions must differ by less than ``IT_TOL``
            for iteration to stop
        :type IT_TOL: float
        :param IT_WALL: maximum permissible number of iterations
        :type IT_WALL: float
        :param IT_MIN: minimum number of particles (DM or star particle) in any iteration; 
            if undercut, shape is unclassified
        :type IT_MIN: int
        :param suffix: suffix for file names
        :type suffix: string
        :param reduced: whether or not reduced shape tensor (1/r^2 factor)
        :type reduced: boolean
        :param shell_based: whether shell-based or ellipsoid-based algorithm should be run
        :type shell_based: boolean
        """
        if rank == 0:
            d, q, s, minor, inter, major, obj_centers, obj_masses = self._getShapeCatLocalBase(xyz.base, masses.base, r200.base, idx_cat.base, obj_size.base, D_LOGSTART, D_LOGEND, D_BINS, IT_TOL, IT_WALL, IT_MIN, reduced, shell_based, suffix)
            del idx_cat
            l_curr_over_target = 3.085678e24/config.OutUnitLength_in_cm
            m_curr_over_target = 1.989e43/config.OutUnitMass_in_g
            d = d*l_curr_over_target
            obj_centers = obj_centers*l_curr_over_target
            obj_masses = obj_masses*m_curr_over_target
            if d.shape[0] != 0:
                minor = minor.reshape(minor.shape[0], -1)
                inter = inter.reshape(inter.shape[0], -1)
                major = major.reshape(major.shape[0], -1)
            else:
                minor = np.array([])
                inter = np.array([])
                major = np.array([])
            
            # Create CAT_DEST if not available
            subprocess.call(['mkdir', '-p', '{}'.format(self.CAT_DEST)], cwd=os.path.join(currentdir))
            np.savetxt('{0}/d_local{1}{2}.txt'.format(self.CAT_DEST, suffix, self.SNAP), d, fmt='%1.7e')
            np.savetxt('{0}/q_local{1}{2}.txt'.format(self.CAT_DEST, suffix, self.SNAP), q, fmt='%1.7e')
            np.savetxt('{0}/s_local{1}{2}.txt'.format(self.CAT_DEST, suffix, self.SNAP), s, fmt='%1.7e')
            np.savetxt('{0}/minor_local{1}{2}.txt'.format(self.CAT_DEST, suffix, self.SNAP), minor, fmt='%1.7e')
            np.savetxt('{0}/inter_local{1}{2}.txt'.format(self.CAT_DEST, suffix, self.SNAP), inter, fmt='%1.7e')
            np.savetxt('{0}/major_local{1}{2}.txt'.format(self.CAT_DEST, suffix, self.SNAP), major, fmt='%1.7e')
            np.savetxt('{0}/m_local{1}{2}.txt'.format(self.CAT_DEST, suffix, self.SNAP), obj_masses, fmt='%1.7e')
            np.savetxt('{0}/centers_local{1}{2}.txt'.format(self.CAT_DEST, suffix, self.SNAP), obj_centers, fmt='%1.7e')
            del d; del q; del s; del minor; del inter; del major; del obj_centers; del obj_masses
        
    def _dumpShapeCatGlobalBase(self, float[:,:] xyz, float[:] masses, float[:] r200, int[:] idx_cat, int[:] obj_size, float IT_TOL, int IT_WALL, int IT_MIN, str suffix, bint reduced):
        """ Dumps all relevant global shape data into ``CAT_DEST``
        
        :param xyz: positions of all simulation particles in Mpc/h
        :type xyz: (N2,3) floats, N2 >> N1
        :param masses: masses of all simulation particles in 10^10*M_sun/h
        :type masses: (N2,) floats
        :param r200: each entry gives the R_200 radius of the parent halo in Mpc/h
        :type r200: (N1,) floats
        :param idx_cat: contains indices of particles belonging to an object
        :type idx_cat: (N3) integers
        :param obj_size: indicates how many particles are in each object
        :type obj_size: (N1,) integers
        :param IT_TOL: convergence tolerance, eigenvalue fractions must differ by less than ``IT_TOL``
            for iteration to stop
        :type IT_TOL: float
        :param IT_WALL: maximum permissible number of iterations
        :type IT_WALL: float
        :param IT_MIN: minimum number of particles (DM or star particle) in any iteration; 
            if undercut, shape is unclassified
        :type IT_MIN: int
        :param suffix: suffix for file names
        :type suffix: string
        :param reduced: whether or not reduced shape tensor (1/r^2 factor)
        :type reduced: boolean"""
        if rank == 0:
            d, q, s, minor, inter, major, obj_centers, obj_masses = self._getShapeCatGlobalBase(xyz.base, masses.base, r200.base, idx_cat.base, obj_size.base, IT_TOL, IT_WALL, IT_MIN, reduced, suffix)
            l_curr_over_target = 3.085678e24/config.OutUnitLength_in_cm
            m_curr_over_target = 1.989e43/config.OutUnitMass_in_g
            d = d*l_curr_over_target
            obj_centers = obj_centers*l_curr_over_target
            obj_masses = obj_masses*m_curr_over_target
            if d.shape[0] != 0:
                minor = minor.reshape(minor.shape[0], -1)
                inter = inter.reshape(inter.shape[0], -1)
                major = major.reshape(major.shape[0], -1)
            else:
                minor = np.array([])
                inter = np.array([])
                major = np.array([])
            
            # Create CAT_DEST if not available
            subprocess.call(['mkdir', '-p', '{}'.format(self.CAT_DEST)], cwd=os.path.join(currentdir))
            np.savetxt('{0}/d_global{1}{2}.txt'.format(self.CAT_DEST, suffix, self.SNAP), d, fmt='%1.7e')
            np.savetxt('{0}/q_global{1}{2}.txt'.format(self.CAT_DEST, suffix, self.SNAP), q, fmt='%1.7e')
            np.savetxt('{0}/s_global{1}{2}.txt'.format(self.CAT_DEST, suffix, self.SNAP), s, fmt='%1.7e')
            np.savetxt('{0}/minor_global{1}{2}.txt'.format(self.CAT_DEST, suffix, self.SNAP), minor, fmt='%1.7e')
            np.savetxt('{0}/inter_global{1}{2}.txt'.format(self.CAT_DEST, suffix, self.SNAP), inter, fmt='%1.7e')
            np.savetxt('{0}/major_global{1}{2}.txt'.format(self.CAT_DEST, suffix, self.SNAP), major, fmt='%1.7e')
            np.savetxt('{0}/m_global{1}{2}.txt'.format(self.CAT_DEST, suffix, self.SNAP), obj_masses, fmt='%1.7e')
            np.savetxt('{0}/centers_global{1}{2}.txt'.format(self.CAT_DEST, suffix, self.SNAP), obj_centers, fmt='%1.7e')
            del d; del q; del s; del minor; del inter; del major; del obj_centers; del obj_masses
    
    def _dumpShapeVelCatLocalBase(self, float[:,:] xyz, float[:,:] velxyz, float[:] masses, float[:] r200, int[:] idx_cat, int[:] obj_size, float D_LOGSTART, float D_LOGEND, int D_BINS, float IT_TOL, int IT_WALL, int IT_MIN, str suffix, bint reduced, bint shell_based):
        """ Dumps all relevant local velocity shape data into ``CAT_DEST``
        
        :param xyz: positions of all simulation particles in Mpc/h
        :type xyz: (N2,3) floats, N2 >> N1
        :param velxyz: velocity array in km/s
        :type velxyz: (N2,3) floats
        :param masses: masses of all simulation particles in 10^10*M_sun/h
        :type masses: (N2,) floats
        :param r200: each entry gives the R_200 radius of the parent halo in Mpc/h
        :type r200: (N1,) floats
        :param idx_cat: contains indices of particles belonging to an object
        :type idx_cat: (N3) integers
        :param obj_size: indicates how many particles are in each object
        :type obj_size: (N1,) integers
        :param D_LOGSTART: logarithm of minimum ellipsoidal radius of interest, in units of R200 of parent halo
        :type D_LOGSTART: int
        :param D_LOGEND: logarithm of maximum ellipsoidal radius of interest, in units of R200 of parent halo
        :type D_LOGEND: int
        :param D_BINS: number of ellipsoidal radii of interest minus 1 (i.e. number of bins)
        :type D_BINS: int
        :param IT_TOL: convergence tolerance, eigenvalue fractions must differ by less than ``IT_TOL``
            for iteration to stop
        :type IT_TOL: float
        :param IT_WALL: maximum permissible number of iterations
        :type IT_WALL: float
        :param IT_MIN: minimum number of particles (DM or star particle) in any iteration; 
            if undercut, shape is unclassified
        :type IT_MIN: int
        :param suffix: suffix for file names
        :type suffix: string
        :param reduced: whether or not reduced shape tensor (1/r^2 factor)
        :type reduced: boolean
        :param shell_based: whether shell-based or ellipsoid-based algorithm should be run
        :type shell_based: boolean
        """
        if rank == 0:
            d, q, s, minor, inter, major, obj_centers, obj_masses = self._getShapeCatVelLocalBase(xyz.base, velxyz.base, masses.base, r200.base, idx_cat.base, obj_size.base, D_LOGSTART, D_LOGEND, D_BINS, IT_TOL, IT_WALL, IT_MIN, reduced, shell_based, suffix)
            l_curr_over_target = 3.085678e24/config.OutUnitLength_in_cm
            m_curr_over_target = 1.989e43/config.OutUnitMass_in_g
            d = d*l_curr_over_target
            obj_centers = obj_centers*l_curr_over_target
            obj_masses = obj_masses*m_curr_over_target
            if d.shape[0] != 0:
                minor = minor.reshape(minor.shape[0], -1)
                inter = inter.reshape(inter.shape[0], -1)
                major = major.reshape(major.shape[0], -1)
            else:
                minor = np.array([])
                inter = np.array([])
                major = np.array([])
            
            # Create CAT_DEST if not available
            subprocess.call(['mkdir', '-p', '{}'.format(self.CAT_DEST)], cwd=os.path.join(currentdir))
            np.savetxt('{0}/d_local{1}{2}.txt'.format(self.CAT_DEST, suffix, self.SNAP), d, fmt='%1.7e')
            np.savetxt('{0}/q_local{1}{2}.txt'.format(self.CAT_DEST, suffix, self.SNAP), q, fmt='%1.7e')
            np.savetxt('{0}/s_local{1}{2}.txt'.format(self.CAT_DEST, suffix, self.SNAP), s, fmt='%1.7e')
            np.savetxt('{0}/minor_local{1}{2}.txt'.format(self.CAT_DEST, suffix, self.SNAP), minor, fmt='%1.7e')
            np.savetxt('{0}/inter_local{1}{2}.txt'.format(self.CAT_DEST, suffix, self.SNAP), inter, fmt='%1.7e')
            np.savetxt('{0}/major_local{1}{2}.txt'.format(self.CAT_DEST, suffix, self.SNAP), major, fmt='%1.7e')
            np.savetxt('{0}/m_local{1}{2}.txt'.format(self.CAT_DEST, suffix, self.SNAP), obj_masses, fmt='%1.7e')
            np.savetxt('{0}/centers_local{1}{2}.txt'.format(self.CAT_DEST, suffix, self.SNAP), obj_centers, fmt='%1.7e')
            del d; del q; del s; del minor; del inter; del major; del obj_centers; del obj_masses
    
    def _dumpShapeVelCatGlobalBase(self, float[:,:] xyz, float[:,:] velxyz, float[:] masses, float[:] r200, int[:] idx_cat, int[:] obj_size, float IT_TOL, int IT_WALL, int IT_MIN, str suffix, bint reduced):
        """ Dumps all relevant global velocity shape data into ``CAT_DEST``
        
        :param xyz: positions of all simulation particles in Mpc/h
        :type xyz: (N2,3) floats, N2 >> N1
        :param velxyz: velocity array in km/s
        :type velxyz: (N2,3) floats
        :param masses: masses of all simulation particles in 10^10*M_sun/h
        :type masses: (N2,) floats
        :param r200: each entry gives the R_200 radius of the parent halo in Mpc/h
        :type r200: (N1,) floats
        :param idx_cat: contains indices of particles belonging to an object
        :type idx_cat: (N3) integers
        :param obj_size: indicates how many particles are in each object
        :type obj_size: (N1,) integers
        :param D_LOGSTART: logarithm of minimum ellipsoidal radius of interest, in units of R200 of parent halo
        :type D_LOGSTART: int
        :param D_LOGEND: logarithm of maximum ellipsoidal radius of interest, in units of R200 of parent halo
        :type D_LOGEND: int
        :param D_BINS: number of ellipsoidal radii of interest minus 1 (i.e. number of bins)
        :type D_BINS: int
        :param IT_TOL: convergence tolerance, eigenvalue fractions must differ by less than ``IT_TOL``
            for iteration to stop
        :type IT_TOL: float
        :param IT_WALL: maximum permissible number of iterations
        :type IT_WALL: float
        :param IT_MIN: minimum number of particles (DM or star particle) in any iteration; 
            if undercut, shape is unclassified
        :type IT_MIN: int
        :param suffix: suffix for file names
        :type suffix: string
        :param reduced: whether or not reduced shape tensor (1/r^2 factor)
        :type reduced: boolean
        """
        
        if rank == 0:
            d, q, s, minor, inter, major, obj_centers, obj_masses = self._getShapeCatVelGlobalBase(xyz.base, velxyz.base, masses.base, r200.base, idx_cat.base, obj_size.base, IT_TOL, IT_WALL, IT_MIN, reduced, suffix)
            l_curr_over_target = 3.085678e24/config.OutUnitLength_in_cm
            m_curr_over_target = 1.989e43/config.OutUnitMass_in_g
            d = d*l_curr_over_target
            obj_centers = obj_centers*l_curr_over_target
            obj_masses = obj_masses*m_curr_over_target
            if d.shape[0] != 0:
                minor = minor.reshape(minor.shape[0], -1)
                inter = inter.reshape(inter.shape[0], -1)
                major = major.reshape(major.shape[0], -1)
            else:
                minor = np.array([])
                inter = np.array([])
                major = np.array([])
            
            # Create CAT_DEST if not available
            subprocess.call(['mkdir', '-p', '{}'.format(self.CAT_DEST)], cwd=os.path.join(currentdir))
            np.savetxt('{0}/d_global{1}{2}.txt'.format(self.CAT_DEST, suffix, self.SNAP), d, fmt='%1.7e')
            np.savetxt('{0}/q_global{1}{2}.txt'.format(self.CAT_DEST, suffix, self.SNAP), q, fmt='%1.7e')
            np.savetxt('{0}/s_global{1}{2}.txt'.format(self.CAT_DEST, suffix, self.SNAP), s, fmt='%1.7e')
            np.savetxt('{0}/minor_global{1}{2}.txt'.format(self.CAT_DEST, suffix, self.SNAP), minor, fmt='%1.7e')
            np.savetxt('{0}/inter_global{1}{2}.txt'.format(self.CAT_DEST, suffix, self.SNAP), inter, fmt='%1.7e')
            np.savetxt('{0}/major_global{1}{2}.txt'.format(self.CAT_DEST, suffix, self.SNAP), major, fmt='%1.7e')
            np.savetxt('{0}/m_global{1}{2}.txt'.format(self.CAT_DEST, suffix, self.SNAP), obj_masses, fmt='%1.7e')
            np.savetxt('{0}/centers_global{1}{2}.txt'.format(self.CAT_DEST, suffix, self.SNAP), obj_centers, fmt='%1.7e')
            del d; del q; del s; del minor; del inter; del major; del obj_centers; del obj_masses
    
    def _plotShapeProfsBase(self, float[:,:] xyz, float[:] masses, float[:] r200, int[:] idx_cat, int[:] obj_size, float D_LOGSTART, float D_LOGEND, int D_BINS, float IT_TOL, int IT_WALL, int IT_MIN, bint reduced, bint shell_based, int nb_bins, str suffix = ''):
        """ Draws shape profiles, also mass bin-decomposed ones
        
        :param xyz: positions of all simulation particles in Mpc/h
        :type xyz: (N2,3) floats, N2 >> N1
        :param masses: masses of all simulation particles in 10^10*M_sun/h
        :type masses: (N2,) floats
        :param r200: each entry gives the R_200 radius of the parent halo in Mpc/h
        :type r200: (N1,) floats
        :param idx_cat: contains indices of particles belonging to an object
        :type idx_cat: (N3) integers
        :param obj_size: indicates how many particles are in each object
        :type obj_size: (N1,) integers
        :param D_LOGSTART: logarithm of minimum ellipsoidal radius of interest, in units of R200 of parent halo
        :type D_LOGSTART: int
        :param D_LOGEND: logarithm of maximum ellipsoidal radius of interest, in units of R200 of parent halo
        :type D_LOGEND: int
        :param D_BINS: number of ellipsoidal radii of interest minus 1 (i.e. number of bins)
        :type D_BINS: int
        :param IT_TOL: convergence tolerance, eigenvalue fractions must differ by less than ``IT_TOL``
            for iteration to stop
        :type IT_TOL: float
        :param IT_WALL: maximum permissible number of iterations
        :type IT_WALL: float
        :param IT_MIN: minimum number of particles (DM or star particle) in any iteration; 
            if undercut, shape is unclassified
        :type IT_MIN: int
        :param reduced: whether or not reduced shape tensor (1/r^2 factor)
        :type reduced: boolean
        :param shell_based: whether shell-based or ellipsoid-based algorithm should be run
        :type shell_based: boolean
        :param nb_bins: Number of mass bins to plot density profiles for
        :type nb_bins: int
        :param suffix: suffix for file names
        :type suffix: string
        """
                
        if rank == 0:
            d, q, s, minor, inter, major, obj_centers, obj_masses = self._getShapeCatLocalBase(xyz.base, masses.base, r200.base, idx_cat.base, obj_size.base, D_LOGSTART, D_LOGEND, D_BINS, IT_TOL, IT_WALL, IT_MIN, reduced, shell_based, suffix)
            getShapeProfs(self.VIZ_DEST, self.SNAP, D_LOGSTART, D_LOGEND, D_BINS, self.start_time, obj_masses, obj_centers, d, q, s, major, nb_bins, self.MASS_UNIT, suffix = suffix)
            del d; del q; del s; del minor; del inter; del major; del obj_centers; del obj_masses
            
    def _plotLocalTHistBase(self, float[:,:] xyz, float[:] masses, float[:] r200, int[:] idx_cat, int[:] obj_size, float D_LOGSTART, float D_LOGEND, int D_BINS, float IT_TOL, int IT_WALL, int IT_MIN, int HIST_NB_BINS, float frac_r200, bint reduced, bint shell_based, str suffix = ''):
        """ Plot a local-shape triaxiality histogram at a specified ellipsoidal depth of ``frac_r200``
        
        :param xyz: positions of all simulation particles in Mpc/h
        :type xyz: (N2,3) floats, N2 >> N1
        :param masses: masses of all simulation particles in 10^10*M_sun/h
        :type masses: (N2,) floats
        :param r200: each entry gives the R_200 radius of the parent halo in Mpc/h
        :type r200: (N1,) floats
        :param idx_cat: contains indices of particles belonging to an object
        :type idx_cat: (N3) integers
        :param obj_size: indicates how many particles are in each object
        :type obj_size: (N1,) integers
        :param D_LOGSTART: logarithm of minimum ellipsoidal radius of interest, in units of R200 of parent halo
        :type D_LOGSTART: int
        :param D_LOGEND: logarithm of maximum ellipsoidal radius of interest, in units of R200 of parent halo
        :type D_LOGEND: int
        :param D_BINS: number of ellipsoidal radii of interest minus 1 (i.e. number of bins)
        :type D_BINS: int
        :param IT_TOL: convergence tolerance, eigenvalue fractions must differ by less than ``IT_TOL``
            for iteration to stop
        :type IT_TOL: float
        :param IT_WALL: maximum permissible number of iterations
        :type IT_WALL: float
        :param IT_MIN: minimum number of particles (DM or star particle) in any iteration; 
            if undercut, shape is unclassified
        :type IT_MIN: int
        :param HIST_NB_BINS: number of histogram bins
        :type HIST_NB_BINS: int
        :param frac_r200: depth of objects to plot triaxiality, in units of R200
        :type frac_r200: float
        :param reduced: whether or not reduced shape tensor (1/r^2 factor)
        :type reduced: boolean
        :param shell_based: whether shell-based or ellipsoid-based algorithm should be run
        :type shell_based: boolean
        :param suffix: suffix for file names
        :type suffix: string
        """
                
        if rank == 0:
            d, q, s, minor, inter, major, obj_centers, obj_masses = self._getShapeCatLocalBase(xyz.base, masses.base, r200.base, idx_cat.base, obj_size.base, D_LOGSTART, D_LOGEND, D_BINS, IT_TOL, IT_WALL, IT_MIN, reduced, shell_based, suffix)
            getLocalTHist(self.VIZ_DEST, self.SNAP, D_LOGSTART, D_LOGEND, D_BINS, self.start_time, obj_masses, obj_centers, d, q, s, major, HIST_NB_BINS, frac_r200, self.MASS_UNIT, suffix = suffix)
            del d; del q; del s; del minor; del inter; del major; del obj_centers; del obj_masses
    
    def _plotGlobalTHistBase(self, float[:,:] xyz, float[:] masses, float[:] r200, int[:] idx_cat, int[:] obj_size, float IT_TOL, int IT_WALL, int IT_MIN, int HIST_NB_BINS, bint reduced, str suffix = ''):
        """ Plot a global-shape triaxiality histogram
                
        :param xyz: positions of all simulation particles in Mpc/h
        :type xyz: (N2,3) floats, N2 >> N1
        :param masses: masses of all simulation particles in 10^10*M_sun/h
        :type masses: (N2,) floats
        :param r200: each entry gives the R_200 radius of the parent halo in Mpc/h
        :type r200: (N1,) floats
        :param idx_cat: contains indices of particles belonging to an object
        :type idx_cat: (N3) integers
        :param obj_size: indicates how many particles are in each object
        :type obj_size: (N1,) integers
        :param IT_TOL: convergence tolerance, eigenvalue fractions must differ by less than ``IT_TOL``
            for iteration to stop
        :type IT_TOL: float
        :param IT_WALL: maximum permissible number of iterations
        :type IT_WALL: float
        :param IT_MIN: minimum number of particles (DM or star particle) in any iteration; 
            if undercut, shape is unclassified
        :type IT_MIN: int
        :param HIST_NB_BINS: number of histogram bins
        :type HIST_NB_BINS: int
        :param suffix: suffix for file names
        :type suffix: string
        """
                
        if rank == 0:
            d, q, s, minor, inter, major, obj_centers, obj_masses = self._getShapeCatGlobalBase(xyz.base, masses.base, r200.base, idx_cat.base, obj_size.base, IT_TOL, IT_WALL, IT_MIN, reduced, suffix)
            getGlobalTHist(self.VIZ_DEST, self.SNAP, self.start_time, obj_masses, obj_centers, d, q, s, major, HIST_NB_BINS, self.MASS_UNIT, suffix = suffix)
            del d; del q; del s; del minor; del inter; del major; del obj_centers; del obj_masses
    
    def _getDensProfsBestFitsBase(self, float[:,:] dens_profs, float[:] ROverR200, float[:] r200, str method = 'einasto'):
        """ Get best-fit results for density profile fitting
        
        :param dens_profs: density profiles to be fit, in units of M_sun*h^2/(Mpc)**3
        :type dens_profs: (N3, r_res) floats
        :param ROverR200: normalized radii at which ``dens_profs`` are defined
        :type ROverR200: (r_res,) floats
        :param r200: each entry gives the R_200 radius of the parent halo in Mpc/h
        :type r200: (N1,) floats
        :param method: string describing density profile model assumed for fitting
        :type method: string, either `einasto`, `alpha_beta_gamma`, `hernquist`, `nfw`
        :return: best-fits for each object
        :rtype: (N3, n) floats, where n is the number of free parameters in the model ``method``"""
        
        if rank == 0:
            best_fits = fitDensProfHelper(dens_profs.base, ROverR200.base, r200.base, method)
            return best_fits
        else:
            return None
        
    def _getConcentrationsBase(self, float[:,:] dens_profs, float[:] ROverR200, float[:] r200, str method = 'einasto'):
        """ Get best-fit concentration values of objects from density profile fitting
        
        :param dens_profs: density profiles to be fit, in units of M_sun*h^2/(Mpc)**3
        :type dens_profs: (N3, r_res) floats
        :param ROverR200: normalized radii at which ``dens_profs`` are defined
        :type ROverR200: (r_res,) floats
        :param r200: each entry gives the R_200 radius of the parent halo in Mpc/h
        :type r200: (N1,) floats
        :param method: string describing density profile model assumed for fitting
        :type method: string, either `einasto`, `alpha_beta_gamma`, `hernquist`, `nfw`
        :return: best-fit concentration for each object
        :rtype: (N3,) floats"""
                
        if rank == 0:
            best_fits = fitDensProfHelper(dens_profs.base, ROverR200.base, r200.base, method)
            cs = r200.base/best_fits[:,-1]
            return cs
        else:
            return None
        
    def _getDensProfsSphDirectBinningBase(self, float[:,:] xyz, float[:] masses, float[:] r200, int[:] idx_cat, int[:] obj_size, float[:] ROverR200):
        """ Get direct-binning-based spherically averaged density profiles
        
        :param xyz: positions of all simulation particles in Mpc/h
        :type xyz: (N2,3) floats, N2 >> N1
        :param masses: masses of all simulation particles in 10^10*M_sun/h
        :type masses: (N2,) floats
        :param r200: each entry gives the R_200 radius of the parent halo in Mpc/h
        :type r200: (N1,) floats
        :param idx_cat: contains indices of particles belonging to an object
        :type idx_cat: (N3) integers
        :param obj_size: indicates how many particles are in each object
        :type obj_size: (N1,) integers
        :param ROverR200: normalized radii at which ``dens_profs`` are defined
        :type ROverR200: (r_res,) floats
        :return: density profiles in M_sun*h^2/(Mpc)**3
        :rtype: (N2, r_res) floats"""
                
        if rank == 0:
            dens_profs = calcDensProfsSphDirectBinning(xyz.base, masses.base, r200.base, ROverR200.base, idx_cat.base, obj_size.base, self.L_BOX, self.CENTER)
            return dens_profs*self.MASS_UNIT
        else:
            return None
        
    def _getDensProfsEllDirectBinningBase(self, float[:,:] xyz, float[:] masses, float[:] r200, int[:] idx_cat, int[:] obj_size, float[:] ROverR200, float[:,:] a, float[:,:] b, float[:,:] c, float[:,:,:] major, float[:,:,:] inter, float[:,:,:] minor):
        """ Get direct-binning-based ellipsoidal shell-based density profiles
        
        :param xyz: positions of all simulation particles in Mpc/h
        :type xyz: (N2,3) floats, N2 >> N1
        :param masses: masses of all simulation particles in 10^10*M_sun/h
        :type masses: (N2,) floats
        :param r200: each entry gives the R_200 radius of the parent halo in Mpc/h
        :type r200: (N1,) floats
        :param idx_cat: contains indices of particles belonging to an object
        :type idx_cat: (N3) integers
        :param obj_size: indicates how many particles are in each object
        :type obj_size: (N1,) integers
        :param ROverR200: normalized radii at which ``dens_profs`` are defined
        :type ROverR200: (r_res,) floats
        :param a: major axis eigenvalues in Mpc/h
        :type a: (N1,D_BINS+1,) floats
        :param b: intermediate axis eigenvalues in Mpc/h
        :type b: (N1,D_BINS+1,) floats
        :param c: minor axis eigenvalues in Mpc/h
        :type c: (N1,D_BINS+1,) floats
        :param major: major axis eigenvectors
        :type major: (N1,D_BINS+1,3) floats
        :param inter: inter axis eigenvectors
        :type inter: (N1,D_BINS+1,3) floats
        :param minor: minor axis eigenvectors
        :type minor: (N1,D_BINS+1,3) floats
        :return: density profiles in M_sun*h^2/(Mpc)**3
        :rtype: (N2, r_res) floats"""
        
        if rank == 0:
            if ROverR200.shape[0] < 3:
                raise ValueError("Your ROverR200 array has fewer than 3 entries. Please modify.")
            dens_profs = calcDensProfsEllDirectBinning(xyz.base, masses.base, r200.base, ROverR200.base, a.base, b.base, c.base, major.base, inter.base, minor.base, idx_cat.base, obj_size.base, self.L_BOX, self.CENTER)
            return dens_profs*self.MASS_UNIT
        else:
            return None
        
    def _getDensProfsKernelBasedBase(self, float[:,:] xyz, float[:] masses, float[:] r200, int[:] idx_cat, int[:] obj_size, float[:] ROverR200):
        """ Get kernel-based density profiles
        
        :param xyz: positions of all simulation particles in Mpc/h
        :type xyz: (N2,3) floats, N2 >> N1
        :param masses: masses of all simulation particles in 10^10*M_sun/h
        :type masses: (N2,) floats
        :param r200: each entry gives the R_200 radius of the parent halo in Mpc/h
        :type r200: (N1,) floats
        :param idx_cat: contains indices of particles belonging to an object
        :type idx_cat: (N3) integers
        :param obj_size: indicates how many particles are in each object
        :type obj_size: (N1,) integers
        :param ROverR200: normalized radii at which ``dens_profs`` are defined
        :type ROverR200: (r_res,) floats
        :return: density profiles in M_sun*h^2/(Mpc)**3
        :rtype: (N2, r_res) floats"""
                
        if rank == 0:
            dens_profs = calcDensProfsKernelBased(xyz.base, masses.base, r200.base, ROverR200.base, idx_cat.base, obj_size.base, self.L_BOX, self.CENTER)
            return dens_profs*self.MASS_UNIT
        else:
            return None
        
    def _estDensProfsBase(self, float[:,:] xyz, float[:] masses, float[:] r200, int[:] idx_cat, int[:] obj_size, float[:] ROverR200, obj_numbers, bint direct_binning = True, bint spherical = True, bint reduced = False, bint shell_based = False, int D_LOGSTART = 0, int D_LOGEND = 0, int D_BINS = 0, float IT_TOL = 0.0, int IT_WALL = 0, int IT_MIN = 0):
        """ Estimate density profiles
        
        :param xyz: positions of all simulation particles in Mpc/h
        :type xyz: (N2,3) floats, N2 >> N1
        :param masses: masses of all simulation particles in 10^10*M_sun/h
        :type masses: (N2,) floats
        :param r200: each entry gives the R_200 radius of the parent halo in Mpc/h
        :type r200: (N1,) floats
        :param idx_cat: contains indices of particles belonging to an object
        :type idx_cat: (N3) integers
        :param obj_size: indicates how many particles are in each object
        :type obj_size: (N1,) integers
        :param ROverR200: normalized radii at which to-be-estimated density profiles are defined
        :type ROverR200: (r_res,) floats
        :param obj_numbers: list of object indices of interest
        :type obj_numbers: list of int
        :param direct_binning: whether or not direct binning approach or
            kernel-based approach should be used
        :type direct_binning: boolean
        :param spherical: whether or not spherical shell-based or ellipsoidal shell-based
            should be used, ignored if ``direct_binning`` = False
        :type spherical: boolean
        :param reduced: whether or not reduced shape tensor (1/r^2 factor) should be used,
            ignored if ``direct_binning`` = False
        :type reduced: boolean
        :param shell_based: whether shell-based or ellipsoid-based algorithm should be run,
            ignored if ``direct_binning`` = False
        :type shell_based: boolean
        :param D_LOGSTART: logarithm of minimum ellipsoidal radius of interest, in units of R200 of parent halo
        :type D_LOGSTART: int
        :param D_LOGEND: logarithm of maximum ellipsoidal radius of interest, in units of R200 of parent halo
        :type D_LOGEND: int
        :param D_BINS: number of ellipsoidal radii of interest minus 1 (i.e. number of bins)
        :type D_BINS: int
        :param IT_TOL: convergence tolerance, eigenvalue fractions must differ by less than ``IT_TOL``
            for iteration to stop
        :type IT_TOL: float
        :param IT_WALL: maximum permissible number of iterations
        :type IT_WALL: int
        :param IT_MIN: minimum number of particles (DM or star particle) in any iteration; 
            if undercut, shape is unclassified
        :type IT_MIN: int
        :return: density profiles in units of config.OutUnitMass_in_g/config.OutUnitLength_in_cm**3
        :rtype: (N2, r_res) floats"""
        print_status(rank,self.start_time,'Starting {} estDensProfs() with snap {}'.format('direct binning' if direct_binning == True else 'kernel based', self.SNAP))
        if type(obj_numbers) == list:
            obj_numbers = np.int32(obj_numbers)
        if rank == 0:
            idx_cat_len = len(obj_size)
            isValidSelection(obj_numbers, idx_cat_len)
            subset_idx_cat = getSubSetIdxCat(idx_cat, obj_size, obj_numbers)
            if direct_binning:
                suffix = '_'
                if spherical == False:
                    d, q, s, minor, inter, major, obj_centers, obj_masses = self._getShapeCatLocalBase(xyz.base, masses.base, r200.base[obj_numbers], subset_idx_cat, obj_size.base[obj_numbers], D_LOGSTART, D_LOGEND, D_BINS, IT_TOL, IT_WALL, IT_MIN, reduced, shell_based, suffix)
                    dens_profs = self._getDensProfsEllDirectBinningBase(xyz.base, masses.base, r200.base[obj_numbers], subset_idx_cat, obj_size.base[obj_numbers], ROverR200.base, d, d*q, d*s, major, inter, minor)
                else:
                    dens_profs = self._getDensProfsSphDirectBinningBase(xyz.base, masses.base, r200.base[obj_numbers], subset_idx_cat, obj_size.base[obj_numbers], ROverR200.base)
            else:
                dens_profs = self._getDensProfsKernelBasedBase(xyz.base, masses.base, r200.base[obj_numbers], subset_idx_cat, obj_size.base[obj_numbers], ROverR200.base)
            l_curr_over_target = 3.085678e24/config.OutUnitLength_in_cm
            m_curr_over_target = 1.989e33/config.OutUnitMass_in_g
            return dens_profs*m_curr_over_target*l_curr_over_target**(-3)
        else:
            return None
        
    def estDensProfs(self, ROverR200, obj_numbers, bint direct_binning = True, bint spherical = True, bint reduced = False, bint shell_based = False): # Public Method
        """ Estimate density profiles
        
        :param ROverR200: normalized radii at which to-be-estimated density profiles are defined
        :type ROverR200: (r_res,) floats
        :param obj_numbers: list of object indices of interest
        :type obj_numbers: list of int
        :param direct_binning: whether or not direct binning approach or
            kernel-based approach should be used
        :type direct_binning: boolean
        :param spherical: whether or not spherical shell-based or ellipsoidal shell-based
            should be used, ignored if ``direct_binning`` = False
        :type spherical: boolean
        :param reduced: whether or not reduced shape tensor (1/r^2 factor) should be used,
            ignored if ``direct_binning`` = False
        :type reduced: boolean
        :param shell_based: whether shell-based or ellipsoid-based algorithm should be run
        :type shell_based: boolean
        :return: density profiles in units of config.OutUnitMass_in_g/config.OutUnitLength_in_cm**3
        :rtype: (N2, r_res) floats"""
        print_status(rank,self.start_time,'Starting {} estDensProfs() with snap {}'.format('direct binning' if direct_binning == True else 'kernel based', self.SNAP))
        if type(obj_numbers) == list:
            obj_numbers = np.int32(obj_numbers)
        if rank == 0:
            nb_objects = len(self.obj_size.base)
            isValidSelection(obj_numbers, nb_objects)
            subset_idx_cat = getSubSetIdxCat(self.idx_cat.base, self.obj_size.base, obj_numbers)
            if direct_binning:
                suffix = '_'
                if spherical == False:
                    d, q, s, minor, inter, major, obj_centers, obj_masses = self._getShapeCatLocalBase(self.xyz.base, self.masses.base, self.r200.base[obj_numbers], subset_idx_cat, self.obj_size.base[obj_numbers], self.D_LOGSTART, self.D_LOGEND, self.D_BINS, self.IT_TOL, self.IT_WALL, self.IT_MIN, reduced, shell_based, suffix)
                    dens_profs = self._getDensProfsEllDirectBinningBase(self.xyz.base, self.masses.base, self.r200.base[obj_numbers], subset_idx_cat, self.obj_size.base[obj_numbers], np.float32(ROverR200), d, d*q, d*s, major, inter, minor)
                else:
                    dens_profs = self._getDensProfsSphDirectBinningBase(self.xyz.base, self.masses.base, self.r200.base[obj_numbers], subset_idx_cat, self.obj_size.base[obj_numbers], np.float32(ROverR200))
            else:
                dens_profs = self._getDensProfsKernelBasedBase(self.xyz.base, self.masses.base, self.r200.base[obj_numbers], subset_idx_cat, self.obj_size.base[obj_numbers], np.float32(ROverR200))
            m_curr_over_target = 1.989e33/config.OutUnitMass_in_g
            l_curr_over_target = 3.085678e24/config.OutUnitLength_in_cm
            return dens_profs*m_curr_over_target*l_curr_over_target**(-3)
        else:
            return None
        
    def _fitDensProfsBase(self, float[:] r200, int[:] obj_size, float[:,:] dens_profs, float[:] ROverR200, str method, obj_numbers):
        """ Get best-fit results for density profile fitting
        
        :param r200: each entry gives the R_200 radius of the parent halo in Mpc/h
        :type r200: (N1,) floats
        :param obj_size: indicates how many particles are in each object
        :type obj_size: (N1,) integers
        :param dens_profs: density profiles to be fit, in units of config.OutUnitMass_in_g/config.OutUnitLength_in_cm**3
        :type dens_profs: (N3, r_res) floats
        :param ROverR200: normalized radii at which ``dens_profs`` are defined
        :type ROverR200: (r_res,) floats
        :param method: string describing density profile model assumed for fitting
        :type method: string, either `einasto`, `alpha_beta_gamma`, `hernquist`, `nfw`
        :param obj_numbers: list of object indices of interest
        :type obj_numbers: list of int
        :return: best-fits for each object
        :rtype: (N3, n) floats, where n is the number of free parameters in the model ``method``"""
        print_status(rank,self.start_time,'Starting fitDensProfs() with snap {0}'.format(self.SNAP))
        if len(dens_profs) != len(obj_numbers):
            raise ValueError("The `obj_numbers` argument is inconsistent with the `dens_profs` handed over to the `fitDensProfs()` function. Please double-check and use the same `obj_numbers` as used for the density profile estimation!")
        if type(obj_numbers) == list:
            obj_numbers = np.int32(obj_numbers)
        l_curr_over_target = config.OutUnitLength_in_cm/3.085678e24
        m_curr_over_target = config.OutUnitMass_in_g/1.989e33
        dens_profs = dens_profs.base*m_curr_over_target*l_curr_over_target**(-3) # So that dens_profs is in M_sun*h^2/(Mpc)**3
        if rank == 0:
            idx_cat_len = len(obj_size)
            isValidSelection(obj_numbers, idx_cat_len)
            best_fits = self._getDensProfsBestFitsBase(dens_profs, ROverR200.base, r200.base[obj_numbers], method)
            l_curr_over_target = 3.085678e24/config.OutUnitLength_in_cm
            m_curr_over_target = 1.989e33/config.OutUnitMass_in_g
            best_fits[:,0] = best_fits[:,0]*m_curr_over_target*l_curr_over_target**(-3)
            if method == 'einasto':
                idx = 2
            elif method == 'alpha_beta_gamma':
                idx = 4
            else:
                idx = 1
            best_fits[:,idx] = best_fits[:,idx]*l_curr_over_target
            return best_fits
        else:
            return None
        
    def _estConcentrationsBase(self, float[:] r200, int[:] obj_size, float[:,:] dens_profs, float[:] ROverR200, str method, obj_numbers):
        """ Get best-fit concentration values of objects from density profile fitting
        
        :param r200: each entry gives the R_200 radius of the parent halo in Mpc/h
        :type r200: (N1,) floats
        :param obj_size: indicates how many particles are in each object
        :type obj_size: (N1,) integers
        :param dens_profs: density profiles whose concentrations are to be determined, 
            in units of config.OutUnitMass_in_g/config.OutUnitLength_in_cm**3
        :type dens_profs: (N3, r_res) floats
        :param ROverR200: normalized radii at which ``dens_profs`` are defined
        :type ROverR200: (r_res,) floats
        :param method: string describing density profile model assumed for fitting
        :type method: string, either `einasto`, `alpha_beta_gamma`, `hernquist`, `nfw`
        :param obj_numbers: list of object indices of interest
        :type obj_numbers: list of int
        :return: best-fit concentration for each object
        :rtype: (N3,) floats"""
        print_status(rank,self.start_time,'Starting estConcentrations() with snap {0}'.format(self.SNAP))
        if len(dens_profs) != len(obj_numbers):
            raise ValueError("The `obj_numbers` argument is inconsistent with the `dens_profs` handed over to the `estConcentrations()` function. Please double-check and use the same `obj_numbers` as used for the density profile estimation!")
        if type(obj_numbers) == list:
            obj_numbers = np.int32(obj_numbers)
        l_curr_over_target = config.OutUnitLength_in_cm/3.085678e24
        m_curr_over_target = config.OutUnitMass_in_g/1.989e33
        dens_profs = dens_profs.base*m_curr_over_target*l_curr_over_target**(-3) # So that dens_profs is in M_sun*h^2/(Mpc)**3
        if rank == 0:
            idx_cat_len = len(obj_size)
            isValidSelection(obj_numbers, idx_cat_len)
            cs = self._getConcentrationsBase(dens_profs, ROverR200.base, r200.base[obj_numbers], method)
            return cs
        else:
            return None
        
    def _plotDensProfsBase(self, float[:] r200, int[:] obj_size, float[:,:] dens_profs, float[:] ROverR200, float[:,:] dens_profs_fit, float[:] ROverR200_fit, str method, str suffix, int nb_bins, obj_numbers):
        """ Draws some simplistic density profiles
        
        :param r200: each entry gives the R_200 radius of the parent halo in Mpc/h
        :type r200: (N1,) floats
        :param obj_size: indicates how many particles are in each object
        :type obj_size: (N1,) integers
        :param dens_profs: estimated density profiles, in units of 
            config.OutUnitMass_in_g/config.OutUnitLength_in_cm**3
        :type dens_profs: (N2, r_res) floats
        :param ROverR200: radii at which ``dens_profs`` are defined
        :type ROverR200: (r_res,) floats
        :param dens_profs_fit: density profiles to be fit, in units of 
            config.OutUnitMass_in_g/config.OutUnitLength_in_cm**3
        :type dens_profs_fit: (N2, r_res2) floats
        :param ROverR200_fit: radii at which best-fits shall be calculated
        :type ROverR200_fit: (r_res2,) floats
        :param method: string describing density profile model assumed for fitting
        :type method: string, either `einasto`, `alpha_beta_gamma`, `hernquist`, `nfw`
        :param suffix: either '_dm_' or '_gx_' or '' (latter for CosmicProfsDirect)
        :type suffix: string
        :param nb_bins: Number of mass bins to plot density profiles for
        :type nb_bins: int
        :param obj_numbers: list of object indices of interest
        :type obj_numbers: list of int
        """
        print_status(rank,self.start_time,'Starting plotDensProfs() with snap {0}'.format(self.SNAP))
        if len(dens_profs) != len(obj_numbers):
            raise ValueError("The `obj_numbers` argument is inconsistent with the `dens_profs` handed over to the `plotDensProfs()` function. Please double-check and use the same `obj_numbers` as used for the density profile estimation!")
        if type(obj_numbers) == list:
            obj_numbers = np.int32(obj_numbers)
        l_curr_over_target = config.OutUnitLength_in_cm/3.085678e24
        m_curr_over_target = config.OutUnitMass_in_g/1.989e33
        dens_profs_ = dens_profs.base*m_curr_over_target*l_curr_over_target**(-3) # So that dens_profs is in M_sun*h^2/(Mpc)**3
        dens_profs_fit_ = dens_profs_fit.base*m_curr_over_target*l_curr_over_target**(-3) # So that dens_profs_fit is in M_sun*h^2/(Mpc)**3
        obj_centers, obj_masses = self._getMassesCenters(obj_numbers) # In units of Mpc/h and 10^10*M_sun*h^2/(Mpc)**3
        
        if rank == 0:
            idx_cat_len = len(obj_size)
            isValidSelection(obj_numbers, idx_cat_len)
            drawDensProfs(self.VIZ_DEST, self.SNAP, r200.base[obj_numbers], dens_profs_fit_, ROverR200_fit.base, dens_profs_, ROverR200.base, obj_masses, obj_centers, method, nb_bins, self.start_time, self.MASS_UNIT, suffix = suffix)
            del obj_centers; del obj_masses; del ROverR200_fit; del dens_profs; del ROverR200
        else:
            del obj_centers; del obj_masses
        
    def _getObjInfoBase(self, int[:] idx_cat, int[:] obj_size, str obj_type):
        """ Print basic info about the objects used for local shape estimation such as number of converged objects
        
        :param idx_cat: each entry of the list is a list containing indices of particles belonging to an object
        :type idx_cat: list of length N1
        :param obj_size: indicates how many particles are in each object
        :type obj_size: (N1,) integers
        :param obj_type: either 'dm', 'gx' or 'unspecified', depending on what catalogue we are looking at
        :type obj_type: string"""
        print_status(rank,self.start_time,'Basic Info. Object type: {0}'.format(obj_type))
        print_status(rank,self.start_time,'Snap {0}'.format(self.SNAP))
        print_status(rank,self.start_time,'Number of objects with sufficient resolution: {0}'.format(obj_size.shape[0]))