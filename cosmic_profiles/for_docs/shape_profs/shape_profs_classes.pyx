#cython: language_level=3
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dens_profs.dens_profs_classes cimport DensProfs, DensProfsHDF5
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
      
cdef class DensShapeProfs(DensProfs):
    """ Class for density profile and shape profile calculations
    
    Its public methods are ``estDensProfs()``, ``getShapeCatLocal()``, ``getShapeCatGlobal()``, 
    ``vizLocalShapes()``, ``vizGlobalShapes()``, ``plotGlobalEpsHist()``, 
    ``plotLocalEpsHist()``, ``plotGlobalTHist()``, ``plotLocalTHist()``, 
    ``dumpShapeCatLocal()``, ``dumpShapeCatGlobal()``, ``getObjInfo()``."""
    
    cdef int D_LOGSTART
    cdef int D_LOGEND
    cdef int D_BINS
    cdef float IT_TOL
    cdef int IT_WALL
    cdef int IT_MIN
    
    def __init__(self, float[:,:] xyz, float[:] masses, idx_cat, float[:] r200, str SNAP, float L_BOX, int MIN_NUMBER_PTCS, int D_LOGSTART, int D_LOGEND, int D_BINS, float IT_TOL, int IT_WALL, int IT_MIN, str CENTER):
        """
        :param xyz: positions of all simulation particles in config.InUnitLength_in_cm
        :type xyz: (N2,3) floats, N2 >> N1
        :param masses: masses of all simulation particles in config.InUnitMass_in_g
        :type masses: (N2,) floats
        :param idx_cat: each entry of the list is a list containing indices of particles belonging to an object
        :type idx_cat: list of length N1
        :param r200: R_200 radii of the parent halos in config.InUnitLength_in_cm
        :type r200: (N1,) floats
        :param SNAP: snapshot identifier, e.g. '024'
        :type SNAP: string
        :param L_BOX: simulation box side length in config.InUnitLength_in_cm
        :type L_BOX: float
        :param MIN_NUMBER_PTCS: minimum number of particles for object to qualify for morphology calculation
        :type MIN_NUMBER_PTCS: int
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
        :param CENTER: shape quantities will be calculated with respect to CENTER = 'mode' (point of highest density)
            or 'com' (center of mass) of each halo
        :type CENTER: str"""
        super().__init__(xyz.base, masses.base, idx_cat, r200.base, SNAP, L_BOX, MIN_NUMBER_PTCS, CENTER)
        self.D_LOGSTART = D_LOGSTART
        self.D_LOGEND = D_LOGEND
        self.D_BINS = D_BINS
        self.IT_TOL = IT_TOL
        self.IT_WALL = IT_WALL
        self.IT_MIN = IT_MIN
        
    def estDensProfs(self, ROverR200, list select, bint direct_binning = True, bint spherical = True, bint reduced = False, bint shell_based = False):
        """ Estimate density profiles
        
        :param ROverR200: normalized radii at which to-be-estimated density profiles are defined
        :type ROverR200: (r_res,) floats
        :param select: index of first and last object to look at in the format [idx_first, idx_last]
        :type select: list containing two integers
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
        :return: density profiles in units of config.OutUnitMass_in_g/config.OutUnitLength_in_cm**3
        :rtype: (N2, r_res) floats"""
        return
        
    def getShapeCatLocal(self, list select, bint reduced = False, bint shell_based = False):
        """ Get all relevant local shape data
        
        :param select: index of first and last object to look at in the format [idx_first, idx_last]
        :type select: list containing two integers
        :param reduced: whether or not reduced shape tensor (1/r^2 factor)
        :type reduced: boolean
        :param shell_based: whether shell-based or ellipsoid-based algorithm should be run
        :type shell_based: boolean
        :return: d, q, s, minor, inter, major, obj_centers in units of config.OutUnitLength_in_cm,
            obj_masses in units of config.OutUnitMass_in_g
        :rtype: 3 x (number_of_objs, D_BINS+1) float arrays, 
            3 x (number_of_objs, D_BINS+1, 3) float arrays, 
            (number_of_objs,3) float array, (number_of_objs,) float array"""
        return
    
    def getShapeCatGlobal(self, list select, bint reduced = False):
        """ Get all relevant global shape data
        
        :param select: index of first and last object to look at in the format [idx_first, idx_last]
        :type select: list containing two integers
        :param reduced: whether or not reduced shape tensor (1/r^2 factor)
        :type reduced: boolean
        :return: d, q, s, minor, inter, major, obj_centers in units of config.OutUnitLength_in_cm,
            obj_masses in units of config.OutUnitMass_in_g
        :rtype: 3 x (number_of_objs,) float arrays, 
            3 x (number_of_objs, 3) float arrays, 
            (number_of_objs, 3) float array, (number_of_objs,) float array"""
        return
    
    def vizLocalShapes(self, obj_numbers, str VIZ_DEST, bint reduced = False, bint shell_based = False):
        """ Visualize local shape of objects with numbers ``obj_numbers``
        
        :param obj_numbers: list of object indices for which to visualize local shapes
        :type obj_numbers: list of int
        :param VIZ_DEST: visualization folder
        :type VIZ_DEST: strings
        :param reduced: whether or not reduced shape tensor (1/r^2 factor)
        :type reduced: boolean
        :param shell_based: whether shell-based or ellipsoid-based algorithm should be run
        :type shell_based: boolean"""
        return
        
    def vizGlobalShapes(self, obj_numbers, str VIZ_DEST, bint reduced = False):
        """ Visualize global shape of objects with numbers ``obj_numbers``
        
        :param obj_numbers: list of object indices for which to visualize global shapes
        :type obj_numbers: list of ints
        :param VIZ_DEST: visualization folder
        :type VIZ_DEST: string
        :param reduced: whether or not reduced shape tensor (1/r^2 factor)
        :type reduced: boolean"""
        return
    
    def plotGlobalEpsHist(self, HIST_NB_BINS, str VIZ_DEST, list select):
        """ Plot global ellipticity histogram
        
        :param HIST_NB_BINS: number of histogram bins
        :type HIST_NB_BINS: int
        :param VIZ_DEST: visualization folder
        :type VIZ_DEST: string
        :param select: index of first and last object to look at in the format [idx_first, idx_last]
        :type select: list containing two integers"""
        return
    
    def plotLocalEpsHist(self, frac_r200, HIST_NB_BINS, str VIZ_DEST, list select):
        """ Plot local ellipticity histogram at depth ``frac_r200``
        
        :param frac_r200: depth of objects to plot ellipticity, in units of R200
        :type frac_r200: float
        :param HIST_NB_BINS: number of histogram bins
        :type HIST_NB_BINS: int
        :param VIZ_DEST: visualization folder
        :type VIZ_DEST: string
        :param select: index of first and last object to look at in the format [idx_first, idx_last]
        :type select: list containing two integers"""
        return
    
    def plotLocalTHist(self, HIST_NB_BINS, str VIZ_DEST, frac_r200, list select, bint reduced = False, bint shell_based = False):
        """ Plot local triaxiality histogram at depth ``frac_r200``
        
        :param HIST_NB_BINS: number of histogram bins
        :type HIST_NB_BINS: int
        :param VIZ_DEST: visualization folder
        :type VIZ_DEST: string
        :param frac_r200: depth of objects to plot triaxiality, in units of R200
        :type frac_r200: float
        :param select: index of first and last object to look at in the format [idx_first, idx_last]
        :type select: list containing two integers
        :param reduced: whether or not reduced shape tensor (1/r^2 factor)
        :type reduced: boolean
        :param shell_based: whether shell-based or ellipsoid-based algorithm should be run
        :type shell_based: boolean"""
        return
    
    def plotGlobalTHist(self, HIST_NB_BINS, str VIZ_DEST, list select, bint reduced = False):
        """ Plot global triaxiality histogram
        
        :param HIST_NB_BINS: number of histogram bins
        :type HIST_NB_BINS: int
        :param VIZ_DEST: visualization folder
        :type VIZ_DEST: string
        :param select: index of first and last object to look at in the format [idx_first, idx_last]
        :type select: list containing two integers
        :param reduced: whether or not reduced shape tensor (1/r^2 factor)
        :type reduced: boolean"""
        return
    
    def plotShapeProfs(self, int nb_bins, str VIZ_DEST, list select, bint reduced = False, bint shell_based = False):
        """ Draws shape profiles, also mass bin-decomposed ones
        
        :param nb_bins: Number of mass bins to plot density profiles for
        :type nb_bins: int
        :param VIZ_DEST: visualization folder
        :type VIZ_DEST: string
        :param reduced: whether or not reduced shape tensor (1/r^2 factor)
        :type reduced: boolean
        :param select: index of first and last object to look at in the format [idx_first, idx_last]
        :type select: list containing two integers
        :param shell_based: whether shell-based or ellipsoid-based algorithm should be run
        :type shell_based: boolean"""
        return
    
    def dumpShapeCatLocal(self, str CAT_DEST, list select, bint reduced = False, bint shell_based = False):
        """ Dumps all relevant local shape data into ``CAT_DEST``
        
        :param CAT_DEST: catalogue folder
        :type CAT_DEST: string
        :param select: index of first and last object to look at in the format [idx_first, idx_last]
        :type select: list containing two integers
        :param reduced: whether or not reduced shape tensor (1/r^2 factor)
        :type reduced: boolean
        :param shell_based: whether shell-based or ellipsoid-based algorithm should be run
        :type shell_based: boolean"""
        return
    
    def dumpShapeCatGlobal(self, str CAT_DEST, list select, bint reduced = False):
        """ Dumps all relevant global shape data into ``CAT_DEST``
        
        :param CAT_DEST: catalogue folder
        :type CAT_DEST: string
        :param select: index of first and last object to look at in the format [idx_first, idx_last]
        :type select: list containing two integers
        :param reduced: whether or not reduced shape tensor (1/r^2 factor)
        :type reduced: boolean"""
        return
    
    def getObjInfo(self):
        """ Print basic info about the objects"""
        return

cdef class DensShapeProfsHDF5(DensProfsHDF5):
    """ Class for density profile and shape profile calculations for Gadget-style HDF5 data
    
    Its public methods are ``getShapeCatLocal()``, ``getShapeCatGlobal()``, 
    ``vizLocalShapes()``, ``vizGlobalShapes()``, ``plotGlobalEpsHist()``, ``plotLocalEpsHist()``.
    ``plotGlobalTHist()``, ``plotLocalTHist()``, ``dumpShapeCatLocal()``,
    ``dumpShapeCatGlobal()``, ``dumpShapeCatVelLocal()``, ``dumpShapeCatVelGlobal()``,
    ``getObjInfo()``."""
    
    cdef int D_LOGSTART
    cdef int D_LOGEND
    cdef int D_BINS
    cdef float IT_TOL
    cdef int IT_WALL
    cdef int IT_MIN
    
    def __init__(self, str HDF5_SNAP_DEST, str HDF5_GROUP_DEST, str SNAP, float L_BOX, int MIN_NUMBER_PTCS, int D_LOGSTART, int D_LOGEND, int D_BINS, float IT_TOL, int IT_WALL, int IT_MIN, str CENTER, str RVIR_OR_R200, str OBJ_TYPE):
        """
        :param HDF5_SNAP_DEST: where we can find the snapshot
        :type HDF5_SNAP_DEST: string
        :param HDF5_GROUP_DEST: where we can find the group files
        :type HDF5_GROUP_DEST: string
        :param SNAP: e.g. '024'
        :type SNAP: string
        :param SNAP: snapshot identifier, e.g. '024'
        :type SNAP: string
        :param L_BOX: simulation box side length in config.InUnitLength_in_cm
        :type L_BOX: float
        :param MIN_NUMBER_PTCS: minimum number of particles for object to qualify for morphology calculation
        :type MIN_NUMBER_PTCS: int
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
        :param CENTER: shape quantities will be calculated with respect to CENTER = 'mode' (point of highest density)
            or 'com' (center of mass) of each halo
        :type CENTER: str
        :param RVIR_OR_R200: 'Rvir' if we want quantities (e.g. D_LOGSTART) to be expressed 
            with respect to the virial radius R_vir, 'R200' for the overdensity radius R_200
        :type RVIR_OR_R200: str
        :param OBJ_TYPE: which simulation particles to consider, 'dm', 'gas' or 'stars'
        :type OBJ_TYPE: str"""
        super().__init__(HDF5_SNAP_DEST, HDF5_GROUP_DEST, SNAP, L_BOX, MIN_NUMBER_PTCS, CENTER, RVIR_OR_R200, OBJ_TYPE)
        self.D_LOGSTART = D_LOGSTART
        self.D_LOGEND = D_LOGEND
        self.D_BINS = D_BINS
        self.IT_TOL = IT_TOL
        self.IT_WALL = IT_WALL
        self.IT_MIN = IT_MIN
        
    def estDensProfs(self, ROverR200, list select, bint direct_binning = True, bint spherical = True, bint reduced = False, bint shell_based = False):
        """ Estimate density profiles
        
        :param ROverR200: normalized radii at which to-be-estimated density profiles are defined
        :type ROverR200: (r_res,) floats
        :param select: index of first and last object to look at in the format [idx_first, idx_last]
        :type select: list containing two integers
        :param direct_binning: whether or not direct binning approach or
            kernel-based approach should be used
        :type direct_binning: boolean
        :param spherical: whether or not spherical shell-based or ellipsoidal shell-based
            should be used, ignored if ``direct_binning`` = False
        :type spherical: boolean
        :param reduced: whether or not reduced shape tensor (1/r^2 factor)
        :type reduced: boolean
        :param shell_based: whether shell-based or ellipsoid-based algorithm should be run
        :type shell_based: boolean
        :return: density profiles in units of config.OutUnitMass_in_g/config.OutUnitLength_in_cm**3
        :rtype: (N2, r_res) floats"""
        return
    
    def getShapeCatLocal(self, list select, bint reduced = False, bint shell_based = False):
        """ Get all relevant local shape data
        
        :param select: index of first and last object to look at in the format [idx_first, idx_last]
        :type select: list containing two integers
        :param reduced: whether or not reduced shape tensor (1/r^2 factor)
        :type reduced: boolean
        :param shell_based: whether shell-based or ellipsoid-based algorithm should be run
        :type shell_based: boolean
        :return: d, q, s, minor, inter, major, obj_centers in units of config.OutUnitLength_in_cm,
            obj_masses in units of config.OutUnitMass_in_g
        :rtype: 3 x (number_of_objs, D_BINS+1) float arrays, 
            3 x (number_of_objs, D_BINS+1, 3) float arrays, 
            (number_of_objs,3) float array, (number_of_objs,) float array"""
        return
    
    def getShapeCatGlobal(self, list select, bint reduced = False):
        """ Get all relevant global shape data
        
        :param select: index of first and last object to look at in the format [idx_first, idx_last]
        :type select: list containing two integers
        :param reduced: whether or not reduced shape tensor (1/r^2 factor)
        :type reduced: boolean
        :return: d, q, s, minor, inter, major, obj_centers in units of config.OutUnitLength_in_cm,
            obj_masses in units of config.OutUnitMass_in_g
        :rtype: 3 x (number_of_objs,) float arrays, 
            3 x (number_of_objs, 3) float arrays, 
            (number_of_objs, 3) float array, (number_of_objs,) float array"""
        return
        
    def getShapeCatVelLocal(self, list select, bint reduced = False, bint shell_based = False): # Public Method
        """ Get all relevant local velocity shape data
        
        :param select: index of first and last object to look at in the format [idx_first, idx_last]
        :type select: list containing two integers
        :param reduced: whether or not reduced shape tensor (1/r^2 factor)
        :type reduced: boolean
        :param shell_based: whether shell-based or ellipsoid-based algorithm should be run
        :type shell_based: boolean
        :return: d, q, s, minor, inter, major, obj_centers in units of config.OutUnitLength_in_cm,
            obj_masses in units of config.OutUnitMass_in_g
        :rtype: 3 x (number_of_objs, D_BINS+1) float arrays,
            3 x (number_of_objs, D_BINS+1, 3) float arrays, 
            (number_of_objs,3) float array, (number_of_objs,) float array"""
        return
    
    def getShapeCatVelGlobal(self, list select, bint reduced = False):
        """ Get all relevant global velocity shape data
        
        :param select: index of first and last object to look at in the format [idx_first, idx_last]
        :type select: list containing two integers
        :param reduced: whether or not reduced shape tensor (1/r^2 factor)
        :type reduced: boolean
        :return: d, q, s, minor, inter, major, obj_centers in units of config.OutUnitLength_in_cm,
            obj_masses in units of config.OutUnitMass_in_g
        :rtype: 3 x (number_of_objs,) float arrays, 
            3 x (number_of_objs, 3) float arrays, 
            (number_of_objs, 3) float array, (number_of_objs,) float array"""
        return
    
    def vizLocalShapes(self, obj_numbers, str VIZ_DEST, bint reduced = False, bint shell_based = False):
        """ Visualize local shape of objects with numbers ``obj_numbers``
        
        :param obj_numbers: list of object indices for which to visualize local shapes
        :type obj_numbers: list of int
        :param VIZ_DEST: visualization folder
        :type VIZ_DEST: strings
        :param reduced: whether or not reduced shape tensor (1/r^2 factor)
        :type reduced: boolean
        :param shell_based: whether shell-based or ellipsoid-based algorithm should be run
        :type shell_based: boolean"""
        return
        
    def vizGlobalShapes(self, obj_numbers, str VIZ_DEST, bint reduced = False):
        """ Visualize global shape of objects with numbers ``obj_numbers``
        
        :param obj_numbers: list of object indices for which to visualize global shapes
        :type obj_numbers: list of ints
        :param VIZ_DEST: visualization folder
        :type VIZ_DEST: string
        :param reduced: whether or not reduced shape tensor (1/r^2 factor)
        :type reduced: boolean"""
        return
    
    def plotGlobalEpsHist(self, HIST_NB_BINS, str VIZ_DEST, list select):
        """ Plot global ellipticity histogram
        
        :param HIST_NB_BINS: number of histogram bins
        :type HIST_NB_BINS: int
        :param VIZ_DEST: visualization folder
        :type VIZ_DEST: string
        :param select: index of first and last object to look at in the format [idx_first, idx_last]
        :type select: list containing two integers"""
        return

    def plotLocalEpsHist(self, frac_r200, HIST_NB_BINS, str VIZ_DEST, list select):
        """ Plot local ellipticity histogram at depth ``frac_r200``
        
        :param frac_r200: depth of objects to plot ellipticity, in units of R200
        :type frac_r200: float
        :param HIST_NB_BINS: number of histogram bins
        :type HIST_NB_BINS: int
        :param VIZ_DEST: visualization folder
        :type VIZ_DEST: string
        :param select: index of first and last object to look at in the format [idx_first, idx_last]
        :type select: list containing two integers"""
        return
    
    def plotLocalTHist(self, HIST_NB_BINS, str VIZ_DEST, frac_r200, list select, bint reduced = False, bint shell_based = False):
        """ Plot local triaxiality histogram at depth ``frac_r200``
        
        :param HIST_NB_BINS: number of histogram bins
        :type HIST_NB_BINS: int
        :param VIZ_DEST: visualization folder
        :type VIZ_DEST: string
        :param frac_r200: depth of objects to plot triaxiality, in units of R200
        :type frac_r200: float
        :param select: index of first and last object to look at in the format [idx_first, idx_last]
        :type select: list containing two integers
        :param reduced: whether or not reduced shape tensor (1/r^2 factor)
        :type reduced: boolean
        :param shell_based: whether shell-based or ellipsoid-based algorithm should be run
        :type shell_based: boolean"""
        return
    
    def plotGlobalTHist(self, HIST_NB_BINS, str VIZ_DEST, list select, bint reduced = False):
        """ Plot global triaxiality histogram
        
        :param reduced: whether or not reduced shape tensor (1/r^2 factor)
        :type reduced: boolean
        :param HIST_NB_BINS: number of histogram bins
        :type HIST_NB_BINS: int
        :param VIZ_DEST: visualization folder
        :type VIZ_DEST: string
        :param select: index of first and last object to look at in the format [idx_first, idx_last]
        :type select: list containing two integers
        :param reduced: whether or not reduced shape tensor (1/r^2 factor)
        :type reduced: boolean"""
        return
        
    def plotShapeProfs(self, int nb_bins, str VIZ_DEST, list select, bint reduced = False, bint shell_based = False):
        """ Draws shape profiles, also mass bin-decomposed ones
        
        :param nb_bins: Number of mass bins to plot density profiles for
        :type nb_bins: int
        :param VIZ_DEST: visualization folder
        :type VIZ_DEST: string
        :param select: index of first and last object to look at in the format [idx_first, idx_last]
        :type select: list containing two integers
        :param reduced: whether or not reduced shape tensor (1/r^2 factor)
        :type reduced: boolean
        :param shell_based: whether shell-based or ellipsoid-based algorithm should be run
        :type shell_based: boolean"""
        return

    def dumpShapeCatLocal(self, str CAT_DEST, list select, bint reduced = False, bint shell_based = False):
        """ Dumps all relevant local shape data into ``CAT_DEST``
        
        :param CAT_DEST: catalogue folder
        :type CAT_DEST: string
        :param select: index of first and last object to look at in the format [idx_first, idx_last]
        :type select: list containing two integers
        :param reduced: whether or not reduced shape tensor (1/r^2 factor)
        :type reduced: boolean
        :param shell_based: whether shell-based or ellipsoid-based algorithm should be run
        :type shell_based: boolean"""
        return

    def dumpShapeCatGlobal(self, str CAT_DEST, list select, bint reduced = False):
        """ Dumps all relevant global shape data into ``CAT_DEST``
        
        :param CAT_DEST: catalogue folder
        :type CAT_DEST: string
        :param select: index of first and last object to look at in the format [idx_first, idx_last]
        :type select: list containing two integers
        :param reduced: whether or not reduced shape tensor (1/r^2 factor)
        :type reduced: boolean"""
        return

    def dumpShapeVelCatLocal(self, str CAT_DEST, list select, bint reduced = False, bint shell_based = False):
        """ Dumps all relevant local velocity shape data into ``CAT_DEST``
        
        :param CAT_DEST: catalogue folder
        :type CAT_DEST: string
        :param select: index of first and last object to look at in the format [idx_first, idx_last]
        :type select: list containing two integers
        :param reduced: whether or not reduced shape tensor (1/r^2 factor)
        :type reduced: boolean
        :param shell_based: whether shell-based or ellipsoid-based algorithm should be run
        :type shell_based: boolean"""
        return

    def dumpShapeVelCatGlobal(self, str CAT_DEST, list select, bint reduced = False):
        """ Dumps all relevant global velocity shape data into ``CAT_DEST``
        
        :param CAT_DEST: catalogue folder
        :type CAT_DEST: string
        :param select: index of first and last object to look at in the format [idx_first, idx_last]
        :type select: list containing two integers
        :param reduced: whether or not reduced shape tensor (1/r^2 factor)
        :type reduced: boolean"""
        return

    def getObjInfo(self):
        """ Print basic info about the objects used for local shape estimation such as number of converged objects"""
        return