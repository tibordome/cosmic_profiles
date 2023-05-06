#cython: language_level=3
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from common import config
from dens_profs.dens_profs_classes cimport DensProfsBase
from common.cosmic_base_class cimport CosmicBase
from common.python_routines import default_katz_config
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
    
    cdef float IT_TOL
    cdef int IT_WALL
    cdef int IT_MIN
    
    def __init__(self, double[:,:] xyz, double[:] masses, idx_cat, double[:] r200, int[:] obj_size, str SNAP, double L_BOX, int MIN_NUMBER_PTCS, str CENTER, str VIZ_DEST, str CAT_DEST, str SUFFIX):
        """
        :param xyz: positions of all simulation particles in Mpc/h (internal length units)
        :type xyz: (N2,3) floats, N2 >> N1
        :param masses: masses of all simulation particles in 10^10*M_sun/h (internal mass units)
        :type masses: (N2,) floats
        :param idx_cat: contains indices of particles belonging to an object
        :type idx_cat: (N3,) integers
        :param r200: R_200 radii of the parent halos in Mpc/h (internal length units)
        :type r200: (N1,) floats
        :param obj_size: indicates how many particles are in each object
        :type obj_size: (N1,) integers
        :param SNAP: snapshot identifier, e.g. '024'
        :type SNAP: string
        :param L_BOX: simulation box side length in Mpc/h (internal length units)
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
        :param SUFFIX: either '_dm_' or '_gx_' or '_' (latter for CosmicProfsDirect)
        :type SUFFIX: string"""
        super().__init__(xyz.base, masses.base, idx_cat, r200.base, obj_size.base, SNAP, L_BOX, MIN_NUMBER_PTCS, CENTER, VIZ_DEST, CAT_DEST, SUFFIX)
        
    def getShapeCatLocal(self, obj_numbers, katz_config = default_katz_config): # Public Method
        """ Get all relevant local shape data
        
        :param obj_numbers: list of object indices of interest
        :type obj_numbers: list of int
        :param katz_config: dictionary with parameters to the Katz algorithm, with fields 'ROverR200', 'IT_TOL', 'IT_WALL', 'IT_MIN', 'REDUCED', 'SHELL_BASED'
        :type katz_config: dictionary
        :return: d in units of config.OutUnitLength_in_cm, q, s, minor, inter, major, obj_centers in units of config.OutUnitLength_in_cm,
            obj_masses in units of config.OutUnitMass_in_g
        :rtype: structured array, containing 3 x (number_of_objs, r_res) double arrays, 
            3 x (number_of_objs, r_res, 3) double arrays"""
        return
    
    def getShapeCatGlobal(self, obj_numbers, katz_config = default_katz_config): # Public Method
        """ Get all relevant global shape data
        
        :param obj_numbers: list of object indices of interest
        :type obj_numbers: list of int
        :param katz_config: dictionary with parameters to the Katz algorithm, with fields 'ROverR200', 'IT_TOL', 'IT_WALL', 'IT_MIN', 'REDUCED', 'SHELL_BASED'
        :type katz_config: dictionary
        :return: d in units of config.OutUnitLength_in_cm, q, s, minor, inter, major, obj_centers in units of config.OutUnitLength_in_cm,
            obj_masses in units of config.OutUnitMass_in_g
        :rtype: structured array, containing 3 x (number_of_objs,) double arrays, 
            3 x (number_of_objs, 3) double arrays"""
        return
    
    def vizLocalShapes(self, obj_numbers, katz_config = default_katz_config): # Public Method
        """ Visualize local shape of objects with numbers ``obj_numbers``
        
        :param obj_numbers: list of object indices of interest
        :type obj_numbers: list of int
        :param katz_config: dictionary with parameters to the Katz algorithm, with fields 'ROverR200', 'IT_TOL', 'IT_WALL', 'IT_MIN', 'REDUCED', 'SHELL_BASED'
        :type katz_config: dictionary"""
        return
        
    def vizGlobalShapes(self, obj_numbers, katz_config = default_katz_config): # Public Method
        """ Visualize global shape of objects with numbers ``obj_numbers``
        
        :param obj_numbers: list of object indices of interest
        :type obj_numbers: list of int
        :param katz_config: dictionary with parameters to the Katz algorithm, with fields 'ROverR200', 'IT_TOL', 'IT_WALL', 'IT_MIN', 'REDUCED', 'SHELL_BASED'
        :type katz_config: dictionary"""
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
    
    def plotLocalTHist(self, HIST_NB_BINS, frac_r200, obj_numbers, katz_config = default_katz_config): # Public Method
        """ Plot local triaxiality histogram at depth ``frac_r200``
        
        :param HIST_NB_BINS: number of histogram bins
        :type HIST_NB_BINS: int
        :param frac_r200: depth of objects to plot triaxiality, in units of R200
        :type frac_r200: float
        :param obj_numbers: list of object indices of interest
        :type obj_numbers: list of int
        :param katz_config: dictionary with parameters to the Katz algorithm, with fields 'ROverR200', 'IT_TOL', 'IT_WALL', 'IT_MIN', 'REDUCED', 'SHELL_BASED'
        :type katz_config: dictionary"""
        return
    
    def plotGlobalTHist(self, HIST_NB_BINS, obj_numbers, katz_config = default_katz_config): # Public Method
        """ Plot global triaxiality histogram
        
        :param katz_dubinski_config: dictionary with parameters to the Katz-Dubinski algorithm, with fields 'ROverR200', 'IT_TOL', 'IT_WALL', 'IT_MIN', 'REDUCED', 'SHELL_BASED'
        :type katz_dubinski_config: dictionary
        :param HIST_NB_BINS: number of histogram bins
        :type HIST_NB_BINS: int
        :param obj_numbers: list of object indices of interest
        :type obj_numbers: list of int
        :param katz_config: dictionary with parameters to the Katz algorithm, with fields 'ROverR200', 'IT_TOL', 'IT_WALL', 'IT_MIN', 'REDUCED', 'SHELL_BASED'
        :type katz_config: dictionary"""
        return
    
    def plotShapeProfs(self, int nb_bins, obj_numbers, katz_config = default_katz_config): # Public Method
        """ Draws shape profiles, also mass bin-decomposed ones
        
        :param nb_bins: Number of mass bins to plot density profiles for
        :type nb_bins: int
        :param obj_numbers: list of object indices of interest
        :type obj_numbers: list of int
        :param katz_config: dictionary with parameters to the Katz algorithm, with fields 'ROverR200', 'IT_TOL', 'IT_WALL', 'IT_MIN', 'REDUCED', 'SHELL_BASED'
        :type katz_config: dictionary"""
        return
    
    def dumpShapeCatLocal(self, obj_numbers, katz_config = default_katz_config): # Public Method
        """ Dumps all relevant local shape data into ``CAT_DEST``
        
        :param obj_numbers: list of object indices of interest
        :type obj_numbers: list of int
        :param katz_config: dictionary with parameters to the Katz algorithm, with fields 'ROverR200', 'IT_TOL', 'IT_WALL', 'IT_MIN', 'REDUCED', 'SHELL_BASED'
        :type katz_config: dictionary"""
        return
    
    def dumpShapeCatGlobal(self, obj_numbers, katz_config = default_katz_config): # Public Method
        """ Dumps all relevant global shape data into ``CAT_DEST``

        :param obj_numbers: list of object indices of interest
        :type obj_numbers: list of int
        :param katz_config: dictionary with parameters to the Katz algorithm, with fields 'ROverR200', 'IT_TOL', 'IT_WALL', 'IT_MIN', 'REDUCED', 'SHELL_BASED'
        :type katz_config: dictionary"""
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
    
    def __init__(self, xyz, masses, idx_cat, r200, L_BOX, SNAP, VIZ_DEST, CAT_DEST, MIN_NUMBER_PTCS = 200, CENTER = 'mode'):
        """
        :param xyz: positions of all simulation particles in config.InUnitLength_in_cm
        :type xyz: (N2,3) floats, N2 >> N1
        :param masses: masses of all simulation particles in config.InUnitMass_in_g
        :type masses: (N2,) floats
        :param idx_cat: each entry of the list is a list containing indices of particles belonging to an object
        :type idx_cat: list of length N1
        :param r200: R_200 radii of the parent halos in config.InUnitLength_in_cm
        :type r200: (N1,) floats
        :param L_BOX: simulation box side length in config.InUnitLength_in_cm
        :type L_BOX: float
        :param SNAP: snapshot identifier, e.g. '024'
        :type SNAP: string
        :param VIZ_DEST: visualization folder
        :type VIZ_DEST: string
        :param CAT_DEST: catalogue destination
        :type CAT_DEST: string
        :param MIN_NUMBER_PTCS: minimum number of particles for object to qualify for morphology calculation
        :type MIN_NUMBER_PTCS: int
        :param CENTER: shape quantities will be calculated with respect to CENTER = 'mode' (point of highest density)
            or 'com' (center of mass) of each halo
        :type CENTER: str"""
        assert xyz.shape[0] == masses.shape[0], "xyz.shape[0] must be equal to masses.shape[0]"
        assert type(idx_cat) == list, "Please provide a list of lists (or at least one list) for idx_cat"
        if not hasattr(r200, "__len__"): # Need right dimensions, if only scalar then
            r200 = np.array([r200])
        if not hasattr(idx_cat[0], "__len__"): # If list not list of lists then
            idx_cat = [idx_cat]
        cdef int nb_objs = len(idx_cat)
        cdef int p
        cdef int[:] obj_pass = np.zeros((nb_objs,), dtype = np.int32)
        cdef int[:] obj_size = np.zeros((nb_objs,), dtype = np.int32)
        for p in range(nb_objs):
            if len(idx_cat[p]) >= np.int32(MIN_NUMBER_PTCS): # Only add objects that have sufficient resolution
                obj_pass[p] = 1      
                obj_size[p] = len(idx_cat[p]) 
        cdef int nb_pass = np.sum(obj_pass.base)
        cat_arr = np.empty((0,), dtype = np.int32)
        cdef int[:] idxs_compr = np.zeros((nb_objs,), dtype = np.int32)
        idxs_compr.base[obj_pass.base.nonzero()[0]] = np.arange(np.sum(obj_pass.base))
        for p in range(nb_objs):
            if obj_pass[p] == 1:
                cat_arr = np.hstack((cat_arr, np.int32(idx_cat[p])))
        l_internal, m_internal, vel_internal = config.getLMVInternal()
        m_curr_over_target = config.InUnitMass_in_g/m_internal
        l_curr_over_target = config.InUnitLength_in_cm/l_internal
        SUFFIX = '_'
        r200 = np.atleast_1d(np.float64(r200))[obj_pass.base.nonzero()[0]]*np.float64(l_curr_over_target)
        super().__init__(np.float64(xyz)*np.float64(l_curr_over_target), np.float64(masses)*np.float64(m_curr_over_target), cat_arr, r200, obj_size.base[obj_pass.base.nonzero()[0]], SNAP, np.float64(L_BOX)*np.float64(l_curr_over_target), np.int32(MIN_NUMBER_PTCS), CENTER, VIZ_DEST, CAT_DEST, SUFFIX)        
        

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
    
    def __init__(self, SNAP_DEST, GROUP_DEST, OBJ_TYPE, SNAP, VIZ_DEST, CAT_DEST, RVIR_OR_R200 = 'Rvir', MIN_NUMBER_PTCS = 200, CENTER = 'mode'):
        """
        :param SNAP_DEST: where we can find the snapshot
        :type SNAP_DEST: string
        :param GROUP_DEST: where we can find the group files
        :type GROUP_DEST: string
        :param OBJ_TYPE: which simulation particles to consider, 'dm', 'gas' or 'stars'
        :type OBJ_TYPE: str
        :param SNAP: snapshot identifier, e.g. '024'
        :type SNAP: string
        :param VIZ_DEST: visualization folder
        :type VIZ_DEST: string
        :param CAT_DEST: catalogue destination
        :type CAT_DEST: string
        :param RVIR_OR_R200: 'Rvir' if we want quantities (e.g. r_over_r200) to be expressed 
            with respect to the virial radius R_vir, 'R200' for the overdensity radius R_200
        :type RVIR_OR_R200: str
        :param MIN_NUMBER_PTCS: minimum number of particles for object to qualify for morphology calculation
        :type MIN_NUMBER_PTCS: int
        :param CENTER: shape quantities will be calculated with respect to CENTER = 'mode' (point of highest density)
            or 'com' (center of mass) of each halo
        :type CENTER: str"""
        self.SNAP_DEST = SNAP_DEST
        self.GROUP_DEST = GROUP_DEST
        self.RVIR_OR_R200 = RVIR_OR_R200
        self.OBJ_TYPE = OBJ_TYPE
        SUFFIX = '_{}_'.format(OBJ_TYPE)
        l_internal, m_internal, vel_internal = config.getLMVInternal()
        l_curr_over_target = config.InUnitLength_in_cm/l_internal
        # Import hdf5 halo data
        nb_shs, sh_len, fof_sizes, group_r200 = getFoFSHData(self.GROUP_DEST, self.RVIR_OR_R200, getPartType(OBJ_TYPE))
        # Import particle data
        xyz = readgadget.read_block(self.SNAP_DEST,"POS ",ptype=[getPartType(self.OBJ_TYPE)]) # Should be in internal length units
        masses = readgadget.read_block(self.SNAP_DEST,"MASS",ptype=[getPartType(self.OBJ_TYPE)])
        # Raise Error message if empty
        if len(nb_shs) == 0:
            raise ValueError("No subhalos found in HDF5 files.")
        if rank == 0:
            # Construct catalogue
            obj_cat, obj_r200, obj_size = calcObjCat(nb_shs, sh_len, fof_sizes, group_r200, np.int32(MIN_NUMBER_PTCS))
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
        L_BOX = np.float64(head.boxsize)
        super().__init__(xyz, masses, obj_cat, obj_r200, obj_size, SNAP, L_BOX*np.float64(l_curr_over_target), np.int32(MIN_NUMBER_PTCS), CENTER, VIZ_DEST, CAT_DEST, SUFFIX)
    
    def getShapeCatVelLocal(self, obj_numbers, katz_config = default_katz_config): # Public Method
        """ Get all relevant local velocity shape data
        
        :param obj_numbers: list of object indices of interest
        :type obj_numbers: list of int
        :param katz_config: dictionary with parameters to the Katz algorithm, with fields 'ROverR200', 'IT_TOL', 'IT_WALL', 'IT_MIN', 'REDUCED', 'SHELL_BASED'
        :type katz_config: dictionary
        :return: d in units of config.OutUnitLength_in_cm, q, s, minor, inter, major, obj_centers in units of config.OutUnitLength_in_cm,
            obj_masses in units of config.OutUnitMass_in_g
        :rtype: structured array, containing 3 x (number_of_objs, r_res) double arrays,
            3 x (number_of_objs, r_res, 3) double arrays"""
        return
    
    def getShapeCatVelGlobal(self, obj_numbers, katz_config = default_katz_config): # Public Method
        """ Get all relevant global velocity shape data
        
        :param obj_numbers: list of object indices of interest
        :type obj_numbers: list of int
        :param katz_config: dictionary with parameters to the Katz algorithm, with fields 'ROverR200', 'IT_TOL', 'IT_WALL', 'IT_MIN', 'REDUCED', 'SHELL_BASED'
        :type katz_config: dictionary
        :return: d in units of config.OutUnitLength_in_cm, q, s, minor, inter, major, obj_centers in units of config.OutUnitLength_in_cm,
            obj_masses in units of config.OutUnitMass_in_g
        :rtype: structured array, containing 3 x (number_of_objs,) double arrays, 
            3 x (number_of_objs, 3) double arrays"""
        return

    def dumpShapeVelCatLocal(self, obj_numbers, katz_config = default_katz_config): # Public Method
        """ Dumps all relevant local velocity shape data into ``CAT_DEST``
        
        :param obj_numbers: list of object indices of interest
        :type obj_numbers: list of int
        :param katz_config: dictionary with parameters to the Katz algorithm, with fields 'ROverR200', 'IT_TOL', 'IT_WALL', 'IT_MIN', 'REDUCED', 'SHELL_BASED'
        :type katz_config: dictionary"""
        return

    def dumpShapeVelCatGlobal(self, obj_numbers, katz_config = default_katz_config): # Public Method
        """ Dumps all relevant global velocity shape data into ``CAT_DEST``
        
        :param obj_numbers: list of object indices of interest
        :type obj_numbers: list of int
        :param katz_config: dictionary with parameters to the Katz algorithm, with fields 'ROverR200', 'IT_TOL', 'IT_WALL', 'IT_MIN', 'REDUCED', 'SHELL_BASED'
        :type katz_config: dictionary"""
        return
    
    def getXYZMasses(self): # Public Method
        """ Retrieve positions and masses of particles
        
        :return xyz, masses: positions in config.OutUnitLength_in_cm and masses 
            in config.OutUnitMass_in_g
        :rtype: (N2,3) floats, (N2,) floats"""
        return
        
    def _getXYZMasses(self):
        """ Retrieve positions and masses of particles
        
        :return xyz, masses: positions in Mpc/h (internal length units) and masses in 10^10*M_sun*h^2/(Mpc)**3
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
        
        :return idx_cat: contains indices of particles belonging to an object,
            obj_size: number of particles in each object
        :rtype: (N3) integers and (N1,) integers"""
        return
        
    def getObjInfoGadget(self): # Public Method
        """ Print basic info about the objects"""
        return
        
    def getHeader(self): # Header
        """ Get header of first file in snapshot"""
        return