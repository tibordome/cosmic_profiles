#cython: language_level=3
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from common import config
cimport cython
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
        return
    
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
        return
    
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
        return
        
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
        return
    
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
        return
    
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
        return
    
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
        return
    
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
        return
    
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
        return
    
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
        return
            
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
        return
    
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
        return
    
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
        return
        
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
        return
        
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
        return
    
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
        return
        
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
        return
        
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
        return
        
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
        return
        
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
        return
        
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
        return
        
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
        return
        
    def _getObjInfoBase(self, int[:] idx_cat, int[:] obj_size, str obj_type):
        """ Print basic info about the objects used for local shape estimation such as number of converged objects
        
        :param idx_cat: each entry of the list is a list containing indices of particles belonging to an object
        :type idx_cat: list of length N1
        :param obj_size: indicates how many particles are in each object
        :type obj_size: (N1,) integers
        :param obj_type: either 'dm', 'gx' or 'unspecified', depending on what catalogue we are looking at
        :type obj_type: string"""
        return