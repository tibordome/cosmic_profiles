#cython: language_level=3
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
cimport cython
from cosmic_profiles.common.python_routines import print_status
from cosmic_profiles.common import config
from cosmic_profiles.shape_profs.shape_profs_tools import getLocalTHist, getGlobalTHist, getShapeProfs
from cosmic_profiles.dens_profs.dens_profs_tools import fitDensProfHelper
from cosmic_profiles.shape_profs.shape_profs_algos import calcMorphLocal, calcMorphGlobal, calcMorphLocalVelDisp, calcMorphGlobalVelDisp
from cosmic_profiles.dens_profs.dens_profs_algos import calcMassesCenters, calcDensProfsSphDirectBinning, calcDensProfsEllDirectBinning, calcDensProfsKernelBased
import time
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
    
    def __init__(self, str SNAP, float L_BOX, int MIN_NUMBER_PTCS, str CENTER):
        """
        :param SNAP: snapshot identifier, e.g. '024'
        :type SNAP: string
        :param L_BOX: simulation box side length in config.InUnitLength_in_cm
        :type L_BOX: float
        :param MIN_NUMBER_PTCS: minimum number of particles for object to qualify for morphology calculation
        :type MIN_NUMBER_PTCS: int
        :param CENTER: shape quantities will be calculated with respect to CENTER = 'mode' (point of highest density)
            or 'com' (center of mass) of each halo
        :type CENTER: str"""
        self.SNAP = SNAP
        self.L_BOX = L_BOX*config.InUnitLength_in_cm/3.085678e24 # self.L_BOX will be in Mpc/h
        self.CENTER = CENTER
        self.MIN_NUMBER_PTCS = MIN_NUMBER_PTCS
        self.start_time = time.time()
        self.SAFE = 6
        self.MASS_UNIT = 1e10
        self.r200 = None
        
    def _getMassesCentersBase(self, float[:,:] xyz, float[:] masses, int[:,:] idx_cat, int[:] obj_size):
        """ Calculate total mass and centers of objects
        
        :param xyz: positions of all simulation particles in Mpc/h
        :type xyz: (N2,3) floats, N2 >> N1
        :param masses: masses of all simulation particles in 10^10*M_sun/h
        :type masses: (N2,) floats
        :param idx_cat: each row contains indices of particles belonging to an object
        :type idx_cat: (N1, N3) integers
        :param obj_size: indicates how many particles are in each object
        :type obj_size: (N1,) integers
        :return centers, m: centers in Mpc/h and masses in 10^10*M_sun/h
        :rtype: (N,3) and (N,) floats"""
        centers, m = calcMassesCenters(xyz.base, masses.base, idx_cat.base, obj_size.base, self.L_BOX, self.CENTER)
        return centers, m
    
    def _getShapeCatLocalBase(self, float[:,:] xyz, float[:] masses, float[:] r200, int[:,:] idx_cat, int[:] obj_size, float D_LOGSTART, float D_LOGEND, int D_BINS, float IT_TOL, int IT_WALL, int IT_MIN, bint reduced, bint shell_based):
        """ Get all relevant local shape data
        
        :param xyz: positions of all simulation particles in Mpc/h
        :type xyz: (N2,3) floats, N2 >> N1
        :param masses: masses of all simulation particles in 10^10*M_sun/h
        :type masses: (N2,) floats
        :param r200: each entry gives the R_200 radius of the parent halo in Mpc/h
        :type r200: (N1,) floats
        :param idx_cat: each row contains indices of particles belonging to an object
        :type idx_cat: (N1, N3) integers
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
        :return: d in Mpc/h, q, s, minor, inter, major, obj_centers in units of Mpc/h,
            obj_masses in units of 10^10*M_sun/h
        :rtype: 3 x (number_of_objs, D_BINS+1) float arrays, 
            3 x (number_of_objs, D_BINS+1, 3) float arrays, 
            (number_of_objs,3) float array, (number_of_objs,) float array
        """
        if rank == 0:
            d, q, s, minor, inter, major, obj_centers, obj_masses = calcMorphLocal(xyz.base, masses.base, r200.base, idx_cat.base, obj_size.base, self.L_BOX, D_LOGSTART, D_LOGEND, D_BINS, IT_TOL, IT_WALL, IT_MIN, self.CENTER, reduced, shell_based)
            return d, q, s, minor, inter, major, obj_centers, obj_masses
        else:
            return None, None, None, None, None, None, None, None
    
    def _getShapeCatGlobalBase(self, float[:,:] xyz, float[:] masses, float[:] r200, int[:,:] idx_cat, int[:] obj_size, float IT_TOL, int IT_WALL, int IT_MIN, bint reduced):
        """ Get all relevant global shape data
        
        :param xyz: positions of all simulation particles in Mpc/h
        :type xyz: (N2,3) floats, N2 >> N1
        :param masses: masses of all simulation particles in 10^10*M_sun/h
        :type masses: (N2,) floats
        :param r200: each entry gives the R_200 radius of the parent halo in Mpc/h
        :type r200: (N1,) floats
        :param idx_cat: each row contains indices of particles belonging to an object
        :type idx_cat: (N1, N3) integers
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
        :return: d in Mpc/h, q, s, minor, inter, major, obj_centers in units of Mpc/h,
            obj_masses in units of 10^10*M_sun/h
        :rtype: 3 x (number_of_objs,) float arrays, 
            3 x (number_of_objs, 3) float arrays, 
            (number_of_objs, 3) float array, (number_of_objs,) float array"""
        if rank == 0:
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
        
    def _getShapeCatVelLocalBase(self, float[:,:] xyz, float[:,:] velxyz, float[:] masses, float[:] r200, int[:,:] idx_cat, int[:] obj_size, float D_LOGSTART, float D_LOGEND, int D_BINS, float IT_TOL, int IT_WALL, int IT_MIN, bint reduced, bint shell_based):
        """ Get all relevant local velocity shape data
        
        :param xyz: positions of all simulation particles in Mpc/h
        :type xyz: (N2,3) floats, N2 >> N1
        :param velxyz: velocity array in km/s
        :type velxyz: (N2,3) floats
        :param masses: masses of all simulation particles in 10^10*M_sun/h
        :type masses: (N2,) floats
        :param r200: each entry gives the R_200 radius of the parent halo in Mpc/h
        :type r200: (N1,) floats
        :param idx_cat: each row contains indices of particles belonging to an object
        :type idx_cat: (N1, N3) integers
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
        :return: d in Mpc/h, q, s, minor, inter, major, obj_centers in units of Mpc/h,
            obj_masses in units of 10^10*M_sun/h
        :rtype: 3 x (number_of_objs, D_BINS+1) float arrays, 
            3 x (number_of_objs, D_BINS+1, 3) float arrays, 
            (number_of_objs,3) float array, (number_of_objs,) float array
        """
        if rank == 0:
            d, q, s, minor, inter, major, obj_centers, obj_masses = calcMorphLocalVelDisp(xyz.base, velxyz.base, masses.base, r200.base, idx_cat.base, obj_size.base, self.L_BOX, D_LOGSTART, D_LOGEND, D_BINS, IT_TOL, IT_WALL, IT_MIN, self.CENTER, reduced, shell_based)
            print_status(rank, self.start_time, "Finished calcMorphLocalVelDisp()")
            return d, q, s, minor, inter, major, obj_centers, obj_masses
        else:
            return None, None, None, None, None, None, None, None
    
    def _getShapeCatVelGlobalBase(self, float[:,:] xyz, float[:,:] velxyz, float[:] masses, float[:] r200, int[:,:] idx_cat, int[:] obj_size, float IT_TOL, int IT_WALL, int IT_MIN, bint reduced):
        """ Get all relevant global velocity shape data
        
        :param xyz: positions of all simulation particles in Mpc/h
        :type xyz: (N2,3) floats, N2 >> N1
        :param velxyz: velocity array in km/s
        :type velxyz: (N2,3) floats
        :param masses: masses of all simulation particles in 10^10*M_sun/h
        :type masses: (N2,) floats
        :param r200: each entry gives the R_200 radius of the parent halo in Mpc/h
        :type r200: (N1,) floats
        :param idx_cat: each row contains indices of particles belonging to an object
        :type idx_cat: (N1, N3) integers
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
        :return: d in Mpc/h, q, s, minor, inter, major, obj_centers in units of Mpc/h,
            obj_masses in units of 10^10*M_sun/h
        :rtype: 3 x (number_of_objs,) float arrays, 
            3 x (number_of_objs, 3) float arrays, 
            (number_of_objs, 3) float array, (number_of_objs,) float array"""
        if rank == 0:
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
    
    def _dumpShapeCatLocalBase(self, float[:,:] xyz, float[:] masses, float[:] r200, int[:,:] idx_cat, int[:] obj_size, float D_LOGSTART, float D_LOGEND, int D_BINS, float IT_TOL, int IT_WALL, int IT_MIN, str CAT_DEST, str suffix, bint reduced, bint shell_based):
        """ Dumps all relevant local shape data into ``CAT_DEST``
        
        :param xyz: positions of all simulation particles in Mpc/h
        :type xyz: (N2,3) floats, N2 >> N1
        :param masses: masses of all simulation particles in 10^10*M_sun/h
        :type masses: (N2,) floats
        :param r200: each entry gives the R_200 radius of the parent halo in Mpc/h
        :type r200: (N1,) floats
        :param idx_cat: each row contains indices of particles belonging to an object
        :type idx_cat: (N1, N3) integers
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
        :param CAT_DEST: catalogue destination
        :type CAT_DEST: string
        :param suffix: suffix for file names
        :type suffix: string
        :param reduced: whether or not reduced shape tensor (1/r^2 factor)
        :type reduced: boolean
        :param shell_based: whether shell-based or ellipsoid-based algorithm should be run
        :type shell_based: boolean
        """
        if rank == 0:
            d, q, s, minor, inter, major, obj_centers, obj_masses = self._getShapeCatLocalBase(xyz.base, masses.base, r200.base, idx_cat.base, obj_size.base, D_LOGSTART, D_LOGEND, D_BINS, IT_TOL, IT_WALL, IT_MIN, reduced, shell_based)
            d = d*3.085678e24/config.OutUnitLength_in_cm
            obj_centers = obj_centers*3.085678e24/config.OutUnitLength_in_cm
            obj_masses = obj_masses*self.MASS_UNIT*1.989e33/config.OutUnitMass_in_g
            if d.shape[0] != 0:
                minor = minor.reshape(minor.shape[0], -1)
                inter = inter.reshape(inter.shape[0], -1)
                major = major.reshape(major.shape[0], -1)
            else:
                minor = np.array([])
                inter = np.array([])
                major = np.array([])
            
            np.savetxt('{0}/d_local{1}{2}.txt'.format(CAT_DEST, suffix, self.SNAP), d, fmt='%1.7e')
            np.savetxt('{0}/q_local{1}{2}.txt'.format(CAT_DEST, suffix, self.SNAP), q, fmt='%1.7e')
            np.savetxt('{0}/s_local{1}{2}.txt'.format(CAT_DEST, suffix, self.SNAP), s, fmt='%1.7e')
            np.savetxt('{0}/minor_local{1}{2}.txt'.format(CAT_DEST, suffix, self.SNAP), minor, fmt='%1.7e')
            np.savetxt('{0}/inter_local{1}{2}.txt'.format(CAT_DEST, suffix, self.SNAP), inter, fmt='%1.7e')
            np.savetxt('{0}/major_local{1}{2}.txt'.format(CAT_DEST, suffix, self.SNAP), major, fmt='%1.7e')
            np.savetxt('{0}/m_local{1}{2}.txt'.format(CAT_DEST, suffix, self.SNAP), obj_masses, fmt='%1.7e')
            np.savetxt('{0}/centers_local{1}{2}.txt'.format(CAT_DEST, suffix, self.SNAP), obj_centers, fmt='%1.7e')
            del d; del q; del s; del minor; del inter; del major; del obj_centers; del obj_masses
        
    def _dumpShapeCatGlobalBase(self, float[:,:] xyz, float[:] masses, float[:] r200, int[:,:] idx_cat, int[:] obj_size, float IT_TOL, int IT_WALL, int IT_MIN, str CAT_DEST, str suffix, bint reduced):
        """ Dumps all relevant global shape data into ``CAT_DEST``
        
        :param xyz: positions of all simulation particles in Mpc/h
        :type xyz: (N2,3) floats, N2 >> N1
        :param masses: masses of all simulation particles in 10^10*M_sun/h
        :type masses: (N2,) floats
        :param r200: each entry gives the R_200 radius of the parent halo in Mpc/h
        :type r200: (N1,) floats
        :param idx_cat: each row contains indices of particles belonging to an object
        :type idx_cat: (N1, N3) integers
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
        :param CAT_DEST: catalogue destination
        :type CAT_DEST: string
        :param suffix: suffix for file names
        :type suffix: string
        :param reduced: whether or not reduced shape tensor (1/r^2 factor)
        :type reduced: boolean"""
        if rank == 0:
            d, q, s, minor, inter, major, obj_centers, obj_masses = self._getShapeCatGlobalBase(xyz.base, masses.base, r200.base, idx_cat.base, obj_size.base, IT_TOL, IT_WALL, IT_MIN, reduced)
            d = d*3.085678e24/config.OutUnitLength_in_cm
            obj_centers = obj_centers*3.085678e24/config.OutUnitLength_in_cm
            obj_masses = obj_masses*self.MASS_UNIT*1.989e33/config.OutUnitMass_in_g
            if d.shape[0] != 0:
                minor = minor.reshape(minor.shape[0], -1)
                inter = inter.reshape(inter.shape[0], -1)
                major = major.reshape(major.shape[0], -1)
            else:
                minor = np.array([])
                inter = np.array([])
                major = np.array([])
            
            np.savetxt('{0}/d_global{1}{2}.txt'.format(CAT_DEST, suffix, self.SNAP), d, fmt='%1.7e')
            np.savetxt('{0}/q_global{1}{2}.txt'.format(CAT_DEST, suffix, self.SNAP), q, fmt='%1.7e')
            np.savetxt('{0}/s_global{1}{2}.txt'.format(CAT_DEST, suffix, self.SNAP), s, fmt='%1.7e')
            np.savetxt('{0}/minor_global{1}{2}.txt'.format(CAT_DEST, suffix, self.SNAP), minor, fmt='%1.7e')
            np.savetxt('{0}/inter_global{1}{2}.txt'.format(CAT_DEST, suffix, self.SNAP), inter, fmt='%1.7e')
            np.savetxt('{0}/major_global{1}{2}.txt'.format(CAT_DEST, suffix, self.SNAP), major, fmt='%1.7e')
            np.savetxt('{0}/m_global{1}{2}.txt'.format(CAT_DEST, suffix, self.SNAP), obj_masses, fmt='%1.7e')
            np.savetxt('{0}/centers_global{1}{2}.txt'.format(CAT_DEST, suffix, self.SNAP), obj_centers, fmt='%1.7e')
            del d; del q; del s; del minor; del inter; del major; del obj_centers; del obj_masses
    
    def _dumpShapeVelCatLocalBase(self, float[:,:] xyz, float[:,:] velxyz, float[:] masses, float[:] r200, int[:,:] idx_cat, int[:] obj_size, float D_LOGSTART, float D_LOGEND, int D_BINS, float IT_TOL, int IT_WALL, int IT_MIN, str CAT_DEST, str suffix, bint reduced, bint shell_based):
        """ Dumps all relevant local velocity shape data into ``CAT_DEST``
        
        :param xyz: positions of all simulation particles in Mpc/h
        :type xyz: (N2,3) floats, N2 >> N1
        :param velxyz: velocity array in km/s
        :type velxyz: (N2,3) floats
        :param masses: masses of all simulation particles in 10^10*M_sun/h
        :type masses: (N2,) floats
        :param r200: each entry gives the R_200 radius of the parent halo in Mpc/h
        :type r200: (N1,) floats
        :param idx_cat: each row contains indices of particles belonging to an object
        :type idx_cat: (N1, N3) integers
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
        :param CAT_DEST: catalogue destination
        :type CAT_DEST: string
        :param suffix: suffix for file names
        :type suffix: string
        :param reduced: whether or not reduced shape tensor (1/r^2 factor)
        :type reduced: boolean
        :param shell_based: whether shell-based or ellipsoid-based algorithm should be run
        :type shell_based: boolean
        """
        if rank == 0:
            d, q, s, minor, inter, major, obj_centers, obj_masses = self._getShapeCatVelLocalBase(xyz.base, velxyz.base, masses.base, r200.base, idx_cat.base, obj_size.base, D_LOGSTART, D_LOGEND, D_BINS, IT_TOL, IT_WALL, IT_MIN, reduced, shell_based)
            d = d*3.085678e24/config.OutUnitLength_in_cm
            obj_centers = obj_centers*3.085678e24/config.OutUnitLength_in_cm
            obj_masses = obj_masses*self.MASS_UNIT*1.989e33/config.OutUnitMass_in_g
            if d.shape[0] != 0:
                minor = minor.reshape(minor.shape[0], -1)
                inter = inter.reshape(inter.shape[0], -1)
                major = major.reshape(major.shape[0], -1)
            else:
                minor = np.array([])
                inter = np.array([])
                major = np.array([])
            
            np.savetxt('{0}/d_local{1}{2}.txt'.format(CAT_DEST, suffix, self.SNAP), d, fmt='%1.7e')
            np.savetxt('{0}/q_local{1}{2}.txt'.format(CAT_DEST, suffix, self.SNAP), q, fmt='%1.7e')
            np.savetxt('{0}/s_local{1}{2}.txt'.format(CAT_DEST, suffix, self.SNAP), s, fmt='%1.7e')
            np.savetxt('{0}/minor_local{1}{2}.txt'.format(CAT_DEST, suffix, self.SNAP), minor, fmt='%1.7e')
            np.savetxt('{0}/inter_local{1}{2}.txt'.format(CAT_DEST, suffix, self.SNAP), inter, fmt='%1.7e')
            np.savetxt('{0}/major_local{1}{2}.txt'.format(CAT_DEST, suffix, self.SNAP), major, fmt='%1.7e')
            np.savetxt('{0}/m_local{1}{2}.txt'.format(CAT_DEST, suffix, self.SNAP), obj_masses, fmt='%1.7e')
            np.savetxt('{0}/centers_local{1}{2}.txt'.format(CAT_DEST, suffix, self.SNAP), obj_centers, fmt='%1.7e')
            del d; del q; del s; del minor; del inter; del major; del obj_centers; del obj_masses
    
    def _dumpShapeVelCatGlobalBase(self, float[:,:] xyz, float[:,:] velxyz, float[:] masses, float[:] r200, int[:,:] idx_cat, int[:] obj_size, float IT_TOL, int IT_WALL, int IT_MIN, str CAT_DEST, str suffix, bint reduced):
        """ Dumps all relevant global velocity shape data into ``CAT_DEST``
        
        :param xyz: positions of all simulation particles in Mpc/h
        :type xyz: (N2,3) floats, N2 >> N1
        :param velxyz: velocity array in km/s
        :type velxyz: (N2,3) floats
        :param masses: masses of all simulation particles in 10^10*M_sun/h
        :type masses: (N2,) floats
        :param r200: each entry gives the R_200 radius of the parent halo in Mpc/h
        :type r200: (N1,) floats
        :param idx_cat: each row contains indices of particles belonging to an object
        :type idx_cat: (N1, N3) integers
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
        :param CAT_DEST: catalogue destination
        :type CAT_DEST: string
        :param suffix: suffix for file names
        :type suffix: string
        :param reduced: whether or not reduced shape tensor (1/r^2 factor)
        :type reduced: boolean
        """
        
        if rank == 0:
            d, q, s, minor, inter, major, obj_centers, obj_masses = self._getShapeCatVelGlobalBase(xyz.base, velxyz.base, masses.base, r200.base, idx_cat.base, obj_size.base, IT_TOL, IT_WALL, IT_MIN, reduced)
            d = d*3.085678e24/config.OutUnitLength_in_cm
            obj_centers = obj_centers*3.085678e24/config.OutUnitLength_in_cm
            obj_masses = obj_masses*self.MASS_UNIT*1.989e33/config.OutUnitMass_in_g
            if d.shape[0] != 0:
                minor = minor.reshape(minor.shape[0], -1)
                inter = inter.reshape(inter.shape[0], -1)
                major = major.reshape(major.shape[0], -1)
            else:
                minor = np.array([])
                inter = np.array([])
                major = np.array([])
            
            np.savetxt('{0}/d_global{1}{2}.txt'.format(CAT_DEST, suffix, self.SNAP), d, fmt='%1.7e')
            np.savetxt('{0}/q_global{1}{2}.txt'.format(CAT_DEST, suffix, self.SNAP), q, fmt='%1.7e')
            np.savetxt('{0}/s_global{1}{2}.txt'.format(CAT_DEST, suffix, self.SNAP), s, fmt='%1.7e')
            np.savetxt('{0}/minor_global{1}{2}.txt'.format(CAT_DEST, suffix, self.SNAP), minor, fmt='%1.7e')
            np.savetxt('{0}/inter_global{1}{2}.txt'.format(CAT_DEST, suffix, self.SNAP), inter, fmt='%1.7e')
            np.savetxt('{0}/major_global{1}{2}.txt'.format(CAT_DEST, suffix, self.SNAP), major, fmt='%1.7e')
            np.savetxt('{0}/m_global{1}{2}.txt'.format(CAT_DEST, suffix, self.SNAP), obj_masses, fmt='%1.7e')
            np.savetxt('{0}/centers_global{1}{2}.txt'.format(CAT_DEST, suffix, self.SNAP), obj_centers, fmt='%1.7e')
            del d; del q; del s; del minor; del inter; del major; del obj_centers; del obj_masses
    
    def _plotShapeProfsBase(self, float[:,:] xyz, float[:] masses, float[:] r200, int[:,:] idx_cat, int[:] obj_size, float D_LOGSTART, float D_LOGEND, int D_BINS, float IT_TOL, int IT_WALL, int IT_MIN, str VIZ_DEST, bint reduced, bint shell_based, int nb_bins, str suffix = ''):
        """ Draws shape profiles, also mass bin-decomposed ones
        
        :param xyz: positions of all simulation particles in Mpc/h
        :type xyz: (N2,3) floats, N2 >> N1
        :param masses: masses of all simulation particles in 10^10*M_sun/h
        :type masses: (N2,) floats
        :param r200: each entry gives the R_200 radius of the parent halo in Mpc/h
        :type r200: (N1,) floats
        :param idx_cat: each row contains indices of particles belonging to an object
        :type idx_cat: (N1, N3) integers
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
        :param VIZ_DEST: visualization folder
        :type VIZ_DEST: string
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
            d, q, s, minor, inter, major, obj_centers, obj_masses = self._getShapeCatLocalBase(xyz.base, masses.base, r200.base, idx_cat.base, obj_size.base, D_LOGSTART, D_LOGEND, D_BINS, IT_TOL, IT_WALL, IT_MIN, reduced, shell_based)
            getShapeProfs(VIZ_DEST, self.SNAP, D_LOGSTART, D_LOGEND, D_BINS, self.start_time, obj_masses, obj_centers, d, q, s, major, nb_bins, self.MASS_UNIT, suffix = suffix)
            del d; del q; del s; del minor; del inter; del major; del obj_centers; del obj_masses
            
    def _plotLocalTHistBase(self, float[:,:] xyz, float[:] masses, float[:] r200, int[:,:] idx_cat, int[:] obj_size, float D_LOGSTART, float D_LOGEND, int D_BINS, float IT_TOL, int IT_WALL, int IT_MIN, str VIZ_DEST, int HIST_NB_BINS, float frac_r200, bint reduced, bint shell_based, str suffix = ''):
        """ Plot a local-shape triaxiality histogram at a specified ellipsoidal depth of ``frac_r200``
        
        :param xyz: positions of all simulation particles in Mpc/h
        :type xyz: (N2,3) floats, N2 >> N1
        :param masses: masses of all simulation particles in 10^10*M_sun/h
        :type masses: (N2,) floats
        :param r200: each entry gives the R_200 radius of the parent halo in Mpc/h
        :type r200: (N1,) floats
        :param idx_cat: each row contains indices of particles belonging to an object
        :type idx_cat: (N1, N3) integers
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
        :param VIZ_DEST: visualization folder
        :type VIZ_DEST: string
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
            d, q, s, minor, inter, major, obj_centers, obj_masses = self._getShapeCatLocalBase(xyz.base, masses.base, r200.base, idx_cat.base, obj_size.base, D_LOGSTART, D_LOGEND, D_BINS, IT_TOL, IT_WALL, IT_MIN, reduced, shell_based)
            getLocalTHist(VIZ_DEST, self.SNAP, D_LOGSTART, D_LOGEND, D_BINS, self.start_time, obj_masses, obj_centers, d, q, s, major, HIST_NB_BINS, frac_r200, self.MASS_UNIT, suffix = suffix)
            del d; del q; del s; del minor; del inter; del major; del obj_centers; del obj_masses
    
    def _plotGlobalTHistBase(self, float[:,:] xyz, float[:] masses, float[:] r200, int[:,:] idx_cat, int[:] obj_size, float IT_TOL, int IT_WALL, int IT_MIN, str VIZ_DEST, int HIST_NB_BINS, bint reduced, str suffix = ''):
        """ Plot a global-shape triaxiality histogram
                
        :param xyz: positions of all simulation particles in Mpc/h
        :type xyz: (N2,3) floats, N2 >> N1
        :param masses: masses of all simulation particles in 10^10*M_sun/h
        :type masses: (N2,) floats
        :param r200: each entry gives the R_200 radius of the parent halo in Mpc/h
        :type r200: (N1,) floats
        :param idx_cat: each row contains indices of particles belonging to an object
        :type idx_cat: (N1, N3) integers
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
        :param VIZ_DEST: visualization folder
        :type VIZ_DEST: string
        :param HIST_NB_BINS: number of histogram bins
        :type HIST_NB_BINS: int
        :param suffix: suffix for file names
        :type suffix: string
        """
                
        if rank == 0:
            d, q, s, minor, inter, major, obj_centers, obj_masses = self._getShapeCatGlobalBase(xyz.base, masses.base, r200.base, idx_cat.base, obj_size.base, IT_TOL, IT_WALL, IT_MIN, reduced)
            getGlobalTHist(VIZ_DEST, self.SNAP, self.start_time, obj_masses, obj_centers, d, q, s, major, HIST_NB_BINS, self.MASS_UNIT, suffix = suffix)
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
        
    def _getDensProfsSphDirectBinningBase(self, float[:,:] xyz, float[:] masses, float[:] r200, int[:,:] idx_cat, int[:] obj_size, float[:] ROverR200):
        """ Get direct-binning-based spherically averaged density profiles
        
        :param xyz: positions of all simulation particles in Mpc/h
        :type xyz: (N2,3) floats, N2 >> N1
        :param masses: masses of all simulation particles in 10^10*M_sun/h
        :type masses: (N2,) floats
        :param r200: each entry gives the R_200 radius of the parent halo in Mpc/h
        :type r200: (N1,) floats
        :param idx_cat: each row contains indices of particles belonging to an object
        :type idx_cat: (N1, N3) integers
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
        
    def _getDensProfsEllDirectBinningBase(self, float[:,:] xyz, float[:] masses, float[:] r200, int[:,:] idx_cat, int[:] obj_size, float[:] ROverR200, float[:,:] a, float[:,:] b, float[:,:] c, float[:,:,:] major, float[:,:,:] inter, float[:,:,:] minor):
        """ Get direct-binning-based ellipsoidal shell-based density profiles
        
        :param xyz: positions of all simulation particles in Mpc/h
        :type xyz: (N2,3) floats, N2 >> N1
        :param masses: masses of all simulation particles in 10^10*M_sun/h
        :type masses: (N2,) floats
        :param r200: each entry gives the R_200 radius of the parent halo in Mpc/h
        :type r200: (N1,) floats
        :param idx_cat: each row contains indices of particles belonging to an object
        :type idx_cat: (N1, N3) integers
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
        
    def _getDensProfsKernelBasedBase(self, float[:,:] xyz, float[:] masses, float[:] r200, int[:,:] idx_cat, int[:] obj_size, float[:] ROverR200):
        """ Get kernel-based density profiles
        
        :param xyz: positions of all simulation particles in Mpc/h
        :type xyz: (N2,3) floats, N2 >> N1
        :param masses: masses of all simulation particles in 10^10*M_sun/h
        :type masses: (N2,) floats
        :param r200: each entry gives the R_200 radius of the parent halo in Mpc/h
        :type r200: (N1,) floats
        :param idx_cat: each row contains indices of particles belonging to an object
        :type idx_cat: (N1, N3) integers
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
        
    def _getObjInfoBase(self, int[:,:] idx_cat, str obj_type):
        """ Print basic info about the objects used for local shape estimation such as number of converged objects
        
        :param idx_cat: each entry of the list is a list containing indices of particles belonging to an object
        :type idx_cat: list of length N1
        :param obj_type: either 'dm', 'gx' or 'unspecified', depending on what catalogue we are looking at
        :type obj_type: string"""
        print_status(rank,self.start_time,'Basic Info. Object type: {0}'.format(obj_type))
        print_status(rank,self.start_time,'Snap {0}'.format(self.SNAP))
        print_status(rank,self.start_time,'Number of objects with sufficient resolution: {0}'.format(idx_cat.shape[0]))