#cython: language_level=3
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from common import config
from dens_profs.dens_profs_classes cimport DensProfsBase
from common.cosmic_base_class cimport CosmicBase
from gadget.read_fof import getFoFSHData, getPartType
from gadget import readgadget
from gadget.gen_catalogues import calcObjCat
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
      
cdef class DensShapeProfsBase(DensProfsBase):
    """ Class for density profile and shape profile calculations
    
    Its public methods are ``getShapeCatLocal()``, ``getShapeCatGlobal()``, 
    ``vizLocalShapes()``, ``vizGlobalShapes()``, ``plotGlobalEpsHist()``, 
    ``plotLocalEpsHist()``, ``plotGlobalTHist()``, ``plotLocalTHist()``, 
    ``dumpShapeCatLocal()``, ``dumpShapeCatGlobal()`` and those of
    ``DensProfsBase``: `getR200()``, ``getIdxCat()``,
    ``getXYZMasses()``, ``getMassesCenters()``, ``_getMassesCenters()``, ``estDensProfs()``, 
    ``fitDensProfs()``, ``estConcentrations()``, ``plotDensProfs()``, ``getObjInfo()``."""
    
    cdef int D_LOGSTART
    cdef int D_LOGEND
    cdef int D_BINS
    cdef float IT_TOL
    cdef int IT_WALL
    cdef int IT_MIN
    
    def __init__(self, float[:,:] xyz, float[:] masses, idx_cat, float[:] r200, int[:] obj_size, str SNAP, float L_BOX, int MIN_NUMBER_PTCS, int D_LOGSTART, int D_LOGEND, int D_BINS, float IT_TOL, int IT_WALL, int IT_MIN, str CENTER, str VIZ_DEST, str CAT_DEST, str SUFFIX):
        """
        :param xyz: positions of all simulation particles in config.InUnitLength_in_cm
        :type xyz: (N2,3) floats, N2 >> N1
        :param masses: masses of all simulation particles in config.InUnitMass_in_g
        :type masses: (N2,) floats
        :param idx_cat: contains indices of particles belonging to an object
        :type idx_cat: (N3,) integers
        :param r200: R_200 radii of the parent halos in config.InUnitLength_in_cm
        :type r200: (N1,) floats
        :param obj_size: indicates how many particles are in each object
        :type obj_size: (N1,) integers
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
        :type IT_WALL: int
        :param IT_MIN: minimum number of particles (DM or star particle) in any iteration; 
            if undercut, shape is unclassified
        :type IT_MIN: int
        :param CENTER: shape quantities will be calculated with respect to CENTER = 'mode' (point of highest density)
            or 'com' (center of mass) of each halo
        :type CENTER: str
        :param VIZ_DEST: visualization folder
        :type VIZ_DEST: string
        :param CAT_DEST: catalogue destination
        :type CAT_DEST: string
        :param SUFFIX: either '_dm_' or '_gx_' or '_' (latter for CosmicProfsDirect)
        :type SUFFIX: string"""
        super().__init__(xyz.base, masses.base, idx_cat, r200.base, obj_size.base, SNAP, L_BOX, MIN_NUMBER_PTCS, CENTER, VIZ_DEST, CAT_DEST, SUFFIX)
        self.D_LOGSTART = D_LOGSTART
        self.D_LOGEND = D_LOGEND
        self.D_BINS = D_BINS
        self.IT_TOL = IT_TOL
        self.IT_WALL = IT_WALL
        self.IT_MIN = IT_MIN
        
    def getShapeCatLocal(self, obj_numbers, bint reduced = False, bint shell_based = False): # Public Method
        """ Get all relevant local shape data
        
        :param obj_numbers: list of object indices of interest
        :type obj_numbers: list of int
        :param reduced: whether or not reduced shape tensor (1/r^2 factor)
        :type reduced: boolean
        :param shell_based: whether shell-based or ellipsoid-based algorithm should be run
        :type shell_based: boolean
        :return: d in units of config.OutUnitLength_in_cm, q, s, minor, inter, major, obj_centers in units of config.OutUnitLength_in_cm,
            obj_masses in units of config.OutUnitMass_in_g
        :rtype: 3 x (number_of_objs, D_BINS+1) float arrays, 
            3 x (number_of_objs, D_BINS+1, 3) float arrays, 
            (number_of_objs,3) float array, (number_of_objs,) float array"""
        return
    
    def getShapeCatGlobal(self, obj_numbers, bint reduced = False): # Public Method
        """ Get all relevant global shape data
        
        :param obj_numbers: list of object indices of interest
        :type obj_numbers: list of int
        :param reduced: whether or not reduced shape tensor (1/r^2 factor)
        :type reduced: boolean
        :return: d in units of config.OutUnitLength_in_cm, q, s, minor, inter, major, obj_centers in units of config.OutUnitLength_in_cm,
            obj_masses in units of config.OutUnitMass_in_g
        :rtype: 3 x (number_of_objs,) float arrays, 
            3 x (number_of_objs, 3) float arrays, 
            (number_of_objs, 3) float array, (number_of_objs,) float array"""
        return
    
    def vizLocalShapes(self, obj_numbers, bint reduced = False, bint shell_based = False): # Public Method
        """ Visualize local shape of objects with numbers ``obj_numbers``
        
        :param obj_numbers: list of object indices for which to visualize local shapes
        :type obj_numbers: list of int
        :param reduced: whether or not reduced shape tensor (1/r^2 factor)
        :type reduced: boolean
        :param shell_based: whether shell-based or ellipsoid-based algorithm should be run
        :type shell_based: boolean"""
        return
        
    def vizGlobalShapes(self, obj_numbers, bint reduced = False): # Public Method
        """ Visualize global shape of objects with numbers ``obj_numbers``
        
        :param obj_numbers: list of object indices for which to visualize global shapes
        :type obj_numbers: list of ints
        :param reduced: whether or not reduced shape tensor (1/r^2 factor)
        :type reduced: boolean"""
        return
    
    def plotGlobalEpsHist(self, HIST_NB_BINS, obj_numbers): # Public Method
        """ Plot global ellipticity histogram
        
        :param HIST_NB_BINS: number of histogram bins
        :type HIST_NB_BINS: int
        :param obj_numbers: list of object indices of interest
        :type obj_numbers: list of int"""
        return
    
    def plotLocalEpsHist(self, frac_r200, HIST_NB_BINS, obj_numbers): # Public Method
        """ Plot local ellipticity histogram at depth ``frac_r200``
        
        :param frac_r200: depth of objects to plot ellipticity, in units of R200
        :type frac_r200: float
        :param HIST_NB_BINS: number of histogram bins
        :type HIST_NB_BINS: int
        :param obj_numbers: list of object indices of interest
        :type obj_numbers: list of int"""
        return
    
    def plotLocalTHist(self, HIST_NB_BINS, frac_r200, obj_numbers, bint reduced = False, bint shell_based = False): # Public Method
        """ Plot local triaxiality histogram at depth ``frac_r200``
        
        :param HIST_NB_BINS: number of histogram bins
        :type HIST_NB_BINS: int
        :param frac_r200: depth of objects to plot triaxiality, in units of R200
        :type frac_r200: float
        :param obj_numbers: list of object indices of interest
        :type obj_numbers: list of int
        :param reduced: whether or not reduced shape tensor (1/r^2 factor)
        :type reduced: boolean
        :param shell_based: whether shell-based or ellipsoid-based algorithm should be run
        :type shell_based: boolean"""
        return
    
    def plotGlobalTHist(self, HIST_NB_BINS, obj_numbers, bint reduced = False): # Public Method
        """ Plot global triaxiality histogram
        
        :param HIST_NB_BINS: number of histogram bins
        :type HIST_NB_BINS: int
        :param obj_numbers: list of object indices of interest
        :type obj_numbers: list of int
        :param reduced: whether or not reduced shape tensor (1/r^2 factor)
        :type reduced: boolean"""
        return
    
    def plotShapeProfs(self, int nb_bins, obj_numbers, bint reduced = False, bint shell_based = False): # Public Method
        """ Draws shape profiles, also mass bin-decomposed ones
        
        :param nb_bins: Number of mass bins to plot density profiles for
        :type nb_bins: int
        :param reduced: whether or not reduced shape tensor (1/r^2 factor)
        :type reduced: boolean
        :param obj_numbers: list of object indices of interest
        :type obj_numbers: list of int
        :param shell_based: whether shell-based or ellipsoid-based algorithm should be run
        :type shell_based: boolean"""
        return
    
    def dumpShapeCatLocal(self, obj_numbers, bint reduced = False, bint shell_based = False): # Public Method
        """ Dumps all relevant local shape data into ``CAT_DEST``
        
        :param obj_numbers: list of object indices of interest
        :type obj_numbers: list of int
        :param reduced: whether or not reduced shape tensor (1/r^2 factor)
        :type reduced: boolean
        :param shell_based: whether shell-based or ellipsoid-based algorithm should be run
        :type shell_based: boolean"""
        return
    
    def dumpShapeCatGlobal(self, obj_numbers, bint reduced = False): # Public Method
        """ Dumps all relevant global shape data into ``CAT_DEST``

        :param obj_numbers: list of object indices of interest
        :type obj_numbers: list of int
        :param reduced: whether or not reduced shape tensor (1/r^2 factor)
        :type reduced: boolean"""
        return


############################################################################################################################
        
################################## User provides index catalogue directly ##################################################
        
############################################################################################################################
cdef class DensShapeProfs(DensShapeProfsBase):
    """ Class for density profile calculations
    
    Its public methods are the same as those of ``DensShapeProfsBase``: 
    ``getShapeCatLocal()``, ``getShapeCatGlobal()``, 
    ``vizLocalShapes()``, ``vizGlobalShapes()``, ``plotGlobalEpsHist()``, 
    ``plotLocalEpsHist()``, ``plotGlobalTHist()``, ``plotLocalTHist()``, 
    ``dumpShapeCatLocal()``, ``dumpShapeCatGlobal()``, ``getR200()``, ``getIdxCat()``,
    ``getXYZMasses()``, ``getMassesCenters()``, ``_getMassesCenters()``, ``estDensProfs()``, 
    ``fitDensProfs()``, ``estConcentrations()``, ``plotDensProfs()``, ``getObjInfo()``"""
    
    def __init__(self, float[:,:] xyz, float[:] masses, idx_cat, float[:] r200, str SNAP, float L_BOX, int MIN_NUMBER_PTCS, int D_LOGSTART, int D_LOGEND, int D_BINS, float IT_TOL, int IT_WALL, int IT_MIN, str CENTER, str VIZ_DEST, str CAT_DEST):
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
        :type CENTER: str
        :param VIZ_DEST: visualization folder
        :type VIZ_DEST: string
        :param CAT_DEST: catalogue destination
        :type CAT_DEST: string"""
        assert xyz.shape[0] == masses.shape[0], "xyz.shape[0] must be equal to masses.shape[0]"
        cdef int nb_objs = len(idx_cat)
        cdef int p
        cdef int[:] obj_pass = np.zeros((nb_objs,), dtype = np.int32)
        cdef int[:] obj_size = np.zeros((nb_objs,), dtype = np.int32)
        for p in range(nb_objs):
            if len(idx_cat[p]) >= MIN_NUMBER_PTCS: # Only add objects that have sufficient resolution
                obj_pass[p] = 1      
                obj_size[p] = len(idx_cat[p]) 
        cdef int nb_pass = np.sum(obj_pass.base)
        cat_arr = np.empty((0,), dtype = np.int32)
        cdef int[:] idxs_compr = np.zeros((nb_objs,), dtype = np.int32)
        idxs_compr.base[obj_pass.base.nonzero()[0]] = np.arange(np.sum(obj_pass.base))
        for p in range(nb_objs):
            if obj_pass[p] == 1:
                cat_arr = np.hstack((cat_arr, np.int32(idx_cat[p])))
        m_curr_over_target = config.InUnitMass_in_g/1.989e43
        l_curr_over_target = config.InUnitLength_in_cm/3.085678e24
        SUFFIX = '_'
        super().__init__(xyz.base*np.float32(l_curr_over_target), masses.base*np.float32(m_curr_over_target), cat_arr, r200.base[obj_pass.base.nonzero()[0]]*np.float32(l_curr_over_target), obj_size.base[obj_pass.base.nonzero()[0]], SNAP, L_BOX*np.float32(l_curr_over_target), MIN_NUMBER_PTCS, D_LOGSTART, D_LOGEND, D_BINS, IT_TOL, IT_WALL, IT_MIN, CENTER, VIZ_DEST, CAT_DEST, SUFFIX)        
        

############################################################################################################################
        
################################## User provides Gadget I, II or HDF5 snapshot #############################################
        
############################################################################################################################
cdef class DensShapeProfsGadget(DensShapeProfsBase):
    """ Class for density profile and shape profile calculations for Gadget-style HDF5 data
    
    Its public methods are ``getShapeCatVelLocal()``, ``getShapeCatVelGlobal()``,
    ``dumpShapeVelCatLocal()``, ``dumpShapeVelCatGlobal()``, ``getXYZMasses()``, 
    ``_getXYZMasses()``, ``getVelXYZ()``, ``_getVelXYZ()``, ``getObjInfoGadget()``, 
    ``getHeader()``and those of ``DensShapeProfsBase``: ``getShapeCatLocal()``, ``getShapeCatGlobal()``, 
    ``vizLocalShapes()``, ``vizGlobalShapes()``, ``plotGlobalEpsHist()``, 
    ``plotLocalEpsHist()``, ``plotGlobalTHist()``, ``plotLocalTHist()``, 
    ``dumpShapeCatLocal()``, ``dumpShapeCatGlobal()``, ``getR200()``, ``getIdxCat()``,
    ``getXYZMasses()``, ``getMassesCenters()``, ``_getMassesCenters()``, ``estDensProfs()``, 
    ``fitDensProfs()``, ``estConcentrations()``, ``plotDensProfs()``, ``getObjInfo()``."""
    
    cdef str SNAP_DEST
    cdef str GROUP_DEST
    cdef str RVIR_OR_R200
    cdef str OBJ_TYPE
    
    def __init__(self, str SNAP_DEST, str GROUP_DEST, str SNAP, int MIN_NUMBER_PTCS, int D_LOGSTART, int D_LOGEND, int D_BINS, float IT_TOL, int IT_WALL, int IT_MIN, str CENTER, str RVIR_OR_R200, str OBJ_TYPE, str VIZ_DEST, str CAT_DEST):
        """
        :param SNAP_DEST: where we can find the snapshot
        :type SNAP_DEST: string
        :param GROUP_DEST: where we can find the group files
        :type GROUP_DEST: string
        :param SNAP: e.g. '024'
        :type SNAP: string
        :param SNAP: snapshot identifier, e.g. '024'
        :type SNAP: string
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
        :type OBJ_TYPE: str
        :param VIZ_DEST: visualization folder
        :type VIZ_DEST: string
        :param CAT_DEST: catalogue destination
        :type CAT_DEST: string"""
        self.SNAP_DEST = SNAP_DEST
        self.GROUP_DEST = GROUP_DEST
        self.RVIR_OR_R200 = RVIR_OR_R200
        self.OBJ_TYPE = OBJ_TYPE
        SUFFIX = '_{}_'.format(OBJ_TYPE)
        l_curr_over_target = config.InUnitLength_in_cm/3.085678e24
        # Import hdf5 halo data
        nb_shs, sh_len, fof_sizes, group_r200 = getFoFSHData(self.GROUP_DEST, self.RVIR_OR_R200, getPartType(OBJ_TYPE))
        # Import particle data
        xyz = readgadget.read_block(self.SNAP_DEST,"POS ",ptype=[getPartType(self.OBJ_TYPE)]) # Should be in 3.085678e24 cm units
        masses = readgadget.read_block(self.SNAP_DEST,"MASS",ptype=[getPartType(self.OBJ_TYPE)])
        # Raise Error message if empty
        if len(nb_shs) == 0:
            raise ValueError("No subhalos found in HDF5 files.")
        if rank == 0:
            # Construct catalogue
            obj_cat, obj_r200, obj_size = calcObjCat(nb_shs, sh_len, fof_sizes, group_r200, MIN_NUMBER_PTCS)
            del nb_shs; del sh_len; del fof_sizes; del group_r200
        else:
            del nb_shs; del sh_len; del fof_sizes; del group_r200; del xyz; del masses  
            obj_cat = None
            obj_r200 = None
            obj_size = None
            xyz = None
            masses = None
        # Find L_BOX
        head = readgadget.header(self.SNAP_DEST)
        L_BOX = np.float32(head.boxsize)
        super().__init__(xyz, masses, obj_cat, obj_r200, obj_size, SNAP, L_BOX*np.float32(l_curr_over_target), MIN_NUMBER_PTCS, D_LOGSTART, D_LOGEND, D_BINS, IT_TOL, IT_WALL, IT_MIN, CENTER, VIZ_DEST, CAT_DEST, SUFFIX)
    
    def getShapeCatVelLocal(self, obj_numbers, bint reduced = False, bint shell_based = False): # Public Method
        """ Get all relevant local velocity shape data
        
        :param obj_numbers: list of object indices of interest
        :type obj_numbers: list of int
        :param reduced: whether or not reduced shape tensor (1/r^2 factor)
        :type reduced: boolean
        :param shell_based: whether shell-based or ellipsoid-based algorithm should be run
        :type shell_based: boolean
        :return: d in units of config.OutUnitLength_in_cm, q, s, minor, inter, major, obj_centers in units of config.OutUnitLength_in_cm,
            obj_masses in units of config.OutUnitMass_in_g
        :rtype: 3 x (number_of_objs, D_BINS+1) float arrays,
            3 x (number_of_objs, D_BINS+1, 3) float arrays, 
            (number_of_objs,3) float array, (number_of_objs,) float array"""
        return
    
    def getShapeCatVelGlobal(self, obj_numbers, bint reduced = False): # Public Method
        """ Get all relevant global velocity shape data
        
        :param obj_numbers: list of object indices of interest
        :type obj_numbers: list of int
        :param reduced: whether or not reduced shape tensor (1/r^2 factor)
        :type reduced: boolean
        :return: d in units of config.OutUnitLength_in_cm, q, s, minor, inter, major, obj_centers in units of config.OutUnitLength_in_cm,
            obj_masses in units of config.OutUnitMass_in_g
        :rtype: 3 x (number_of_objs,) float arrays, 
            3 x (number_of_objs, 3) float arrays, 
            (number_of_objs, 3) float array, (number_of_objs,) float array"""
        return

    def dumpShapeVelCatLocal(self, obj_numbers, bint reduced = False, bint shell_based = False): # Public Method
        """ Dumps all relevant local velocity shape data into ``CAT_DEST``
        
        :param obj_numbers: list of object indices of interest
        :type obj_numbers: list of int
        :param reduced: whether or not reduced shape tensor (1/r^2 factor)
        :type reduced: boolean
        :param shell_based: whether shell-based or ellipsoid-based algorithm should be run
        :type shell_based: boolean"""
        return

    def dumpShapeVelCatGlobal(self, obj_numbers, bint reduced = False): # Public Method
        """ Dumps all relevant global velocity shape data into ``CAT_DEST``
        
        :param obj_numbers: list of object indices of interest
        :type obj_numbers: list of int
        :param reduced: whether or not reduced shape tensor (1/r^2 factor)
        :type reduced: boolean"""
        return
    
    def getXYZMasses(self): # Public Method
        """ Retrieve positions and masses of particles
        
        :return xyz, masses: positions in config.OutUnitLength_in_cm and masses 
            in config.OutUnitMass_in_g
        :rtype: (N2,3) floats, (N2,) floats"""
        return
        
    def _getXYZMasses(self):
        """ Retrieve positions and masses of particles
        
        :return xyz, masses: positions in Mpc/h and masses in 10^10*M_sun*h^2/(Mpc)**3
        :rtype: (N2,3) floats, (N2,) floats"""
        return
    
    def getVelXYZ(self): # Public Method
        """ Retrieve velocities of particles
        
        :return velxyz: velocity array in config.OutUnitVelocity_in_cm_per_s
        :rtype: (N2,3) floats"""
        return
    
    def _getVelXYZ(self):
        """ Retrieve velocities of particles
        
        :return velxyz: velocity array in km/s
        :rtype: (N2,3) floats"""
        return
    
    def getR200(self): # Public Method
        """ Fetch R200 values
        
        :return obj_r200: R200 value of parent halos in config.OutUnitLength_in_cm
        :rtype: (N1,) floats"""
        return
        
    def getIdxCat(self): # Public Method
        """ Fetch catalogue
        
        :return idx_cat: each row contains indices of particles belonging to an object,
            obj_size: number of particles in each object
        :rtype: (N1, N3) integers and (N1,) integers"""
        return
        
    def getObjInfoHDF5(self): # Public Method
        """ Print basic info about the objects"""
        return
        
    def getHeader(self): # Header
        """ Get header of first file in snapshot"""
        return