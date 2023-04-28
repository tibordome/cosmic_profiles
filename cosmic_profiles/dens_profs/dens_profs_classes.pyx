#cython: language_level=3
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from cosmic_profiles.common.cosmic_base_class cimport CosmicBase
from cosmic_profiles.common import config
from cosmic_profiles.common.python_routines import print_status, isValidSelection, getSubSetIdxCat
from cosmic_profiles.gadget.read_fof import getFoFSHData, getPartType
from cosmic_profiles.gadget.gen_catalogues import calcObjCat
from cosmic_profiles.gadget import readgadget
import time
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
        
cdef class DensProfsBase(CosmicBase):
    """ Class for density profile calculations
    
    Its public methods are ``getR200()``, ``getIdxCat()``,
    ``getXYZMasses()``, ``getMassesCenters()``, ``_getMassesCenters()``, ``estDensProfs()``, 
    ``fitDensProfs()``, ``estConcentrations()``, ``plotDensProfs()``, ``getObjInfo()``."""
    
    def __init__(self, float[:,:] xyz, float[:] masses, int[:] idx_cat, float[:] r200, int[:] obj_size, str SNAP, float L_BOX, int MIN_NUMBER_PTCS, str CENTER, str VIZ_DEST, str CAT_DEST, str SUFFIX):
        """
        :param xyz: positions of all simulation particles in Mpc/h (internal length units)
        :type xyz: (N2,3) floats, N2 >> N1
        :param masses: masses of all simulation particles in 10^10 M_sun/h
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
        super().__init__(SNAP, L_BOX, MIN_NUMBER_PTCS, CENTER, VIZ_DEST, CAT_DEST, SUFFIX)
        assert xyz.shape[0] == masses.shape[0], "xyz.shape[0] must be equal to masses.shape[0]"
        self.xyz = xyz.base
        self.masses = masses.base
        self.idx_cat = idx_cat.base
        self.obj_size = obj_size.base
        self.r200 = r200.base
       
    def getR200(self): # Public Method
        """ Get overdensity radii in config.OutUnitLength_in_cm units"""
        print_status(rank,self.start_time,'Starting getR200() with snap {0}'.format(self.SNAP))
        
        if rank == 0:
            l_internal, m_internal, vel_internal = config.getLMVInternal()
            l_curr_over_target = l_internal/config.OutUnitLength_in_cm
            return self.r200.base*l_curr_over_target
        else:
            return None
    
    def getIdxCat(self): # Public Method
        """ Fetch catalogue
        
        :return idx_cat: contains indices of particles belonging to an object,
            obj_size: number of particles in each object
        :rtype: (N3) integers and (N1,) integers"""
        if rank == 0:
            return self.idx_cat.base, self.obj_size.base
        else:
            return None, None
        
    def getXYZMasses(self): # Public Method
        """ Retrieve positions and masses of particles
        
        :return xyz, masses: positions in config.OutUnitLength_in_cm and masses 
            in config.OutUnitMass_in_g
        :rtype: (N2,3) floats, (N2,) floats"""
        if rank == 0:
            l_internal, m_internal, vel_internal = config.getLMVInternal()
            l_curr_over_target = l_internal/config.OutUnitLength_in_cm
            m_curr_over_target = m_internal/config.OutUnitMass_in_g
            return self.xyz.base*l_curr_over_target, self.masses.base*m_curr_over_target
        else:
            return None, None
            
    def getMassesCenters(self, obj_numbers): # Public Method
        """ Calculate total mass and centers of objects 
        
        Note that the units will be in config.OutUnitLength_in_cm and config.OutUnitMass_in_g.
        
        :param obj_numbers: list of object indices of interest
        :type obj_numbers: list of int
        :return centers, m: centers in config.OutUnitLength_in_cm and masses in config.OutUnitMass_in_g
        :rtype: (N,3) and (N,) floats"""
        if type(obj_numbers) == list:
            obj_numbers = np.int32(obj_numbers)
        if rank == 0:
            idx_cat_len = len(self.obj_size.base)
            isValidSelection(obj_numbers, idx_cat_len)
            subset_idx_cat = getSubSetIdxCat(self.idx_cat.base, self.obj_size.base, obj_numbers)
            centers, ms = self._getMassesCentersBase(self.xyz.base, self.masses.base, subset_idx_cat, self.obj_size.base[obj_numbers])
            l_internal, m_internal, vel_internal = config.getLMVInternal()
            l_curr_over_target = l_internal/config.OutUnitLength_in_cm
            m_curr_over_target = m_internal/config.OutUnitMass_in_g
            return centers*l_curr_over_target, ms*m_curr_over_target
        else:
            return None, None
    
    def _getMassesCenters(self, obj_numbers):
        """ Calculate total mass and centers of objects
        
        :param obj_numbers: list of object indices of interest
        :type obj_numbers: list of int
        :return centers, m: centers in Mpc/h (internal length units) and masses in 10^10*M_sun*h^2/(Mpc)**3
        :rtype: (N,3) and (N,) floats"""
        if type(obj_numbers) == list:
            obj_numbers = np.int32(obj_numbers)
        if rank == 0:
            idx_cat_len = len(self.obj_size.base)
            isValidSelection(obj_numbers, idx_cat_len)
            subset_idx_cat = getSubSetIdxCat(self.idx_cat.base, self.obj_size.base, obj_numbers)
            centers, ms = self._getMassesCentersBase(self.xyz.base, self.masses.base, subset_idx_cat, self.obj_size.base[obj_numbers])
            return centers, ms
        else:
            return None, None
        
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
        :param shell_based: whether shell-based or ellipsoid-based algorithm should be run,
            ignored if ``direct_binning`` = False
        :type shell_based: boolean
        :return: density profiles in units of config.OutUnitMass_in_g/config.OutUnitLength_in_cm**3
        :rtype: (N2, r_res) floats"""
        cdef int D_LOGSTART
        cdef int D_LOGEND 
        cdef int D_BINS
        cdef float IT_TOL
        cdef int IT_WALL
        cdef int IT_MIN
        try:
            D_LOGSTART = self.D_LOGSTART
            D_LOGEND = self.D_LOGEND
            D_BINS = self.D_BINS
            IT_TOL = self.IT_TOL
            IT_WALL = self.IT_WALL
            IT_MIN = self.IT_MIN
        except:
            D_LOGSTART = 0
            D_LOGEND = 0
            D_BINS = 0
            IT_TOL = 0.0
            IT_WALL = 0
            IT_MIN = 0
        dens_profs = self._estDensProfsBase(self.xyz.base, self.masses.base, self.r200.base, self.idx_cat.base, self.obj_size.base, np.float32(ROverR200), obj_numbers, direct_binning, spherical, reduced, shell_based, D_LOGSTART, D_LOGEND, D_BINS, IT_TOL, IT_WALL, IT_MIN)
        return dens_profs
    
    def fitDensProfs(self, dens_profs, ROverR200, str method, obj_numbers): # Public Method
        """ Get best-fit results for density profile fitting
        
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
        best_fits = self._fitDensProfsBase(self.r200.base, self.obj_size.base, np.float32(dens_profs), np.float32(ROverR200), method, obj_numbers)
        return best_fits
        
    def estConcentrations(self, dens_profs, ROverR200, str method, obj_numbers): # Public Method
        """ Get best-fit concentration values of objects from density profile fitting
        
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
        cs = self._estConcentrationsBase(self.r200.base, self.obj_size.base, np.float32(dens_profs), np.float32(ROverR200), method, obj_numbers)
        return cs
        
    def plotDensProfs(self, dens_profs, ROverR200, dens_profs_fit, ROverR200_fit, str method, int nb_bins, obj_numbers): # Public Method
        """ Draws some simplistic density profiles
        
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
        :param nb_bins: Number of mass bins to plot density profiles for
        :type nb_bins: int
        :param obj_numbers: list of object indices of interest
        :type obj_numbers: list of int
        """
        self._plotDensProfsBase(self.r200.base, self.obj_size.base, np.float32(dens_profs), np.float32(ROverR200), np.float32(dens_profs_fit), np.float32(ROverR200_fit), method, self.SUFFIX, nb_bins, obj_numbers)
            
    def getObjInfo(self): # Public Method
        """ Print basic info about the objects"""
        print_status(rank,self.start_time,'Starting getObjInfo() with snap {0}'.format(self.SNAP))
        obj_type = 'unspecified'
        self._getObjInfoBase(self.idx_cat.base, self.obj_size.base, obj_type)
        
############################################################################################################################
        
################################## User provides index catalogue directly ##################################################
        
############################################################################################################################
cdef class DensProfs(DensProfsBase):
    """ Class for density profile calculations
    
    Its public methods are the same as those of ``DensProfsBase``: ``getR200()``, ``getIdxCat()``,
    ``getXYZMasses()``, ``getMassesCenters()``, ``_getMassesCenters()``, ``estDensProfs()``, 
    ``fitDensProfs()``, ``estConcentrations()``, ``plotDensProfs()``, ``getObjInfo()``."""
    
    def __init__(self, xyz, masses, idx_cat, r200, SNAP, L_BOX, VIZ_DEST, CAT_DEST, MIN_NUMBER_PTCS = 200, CENTER = 'mode'):
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
        :param VIZ_DEST: visualization folder
        :type VIZ_DEST: string
        :param CAT_DEST: catalogue destination
        :type CAT_DEST: string
        :param MIN_NUMBER_PTCS: minimum number of particles for object to qualify for morphology calculation
        :type MIN_NUMBER_PTCS: int
        :param CENTER: shape quantities will be calculated with respect to CENTER = 'mode' (point of highest density)
            or 'com' (center of mass) of each halo
        :type CENTER: str
        """
        assert xyz.shape[0] == masses.shape[0], "xyz.shape[0] must be equal to masses.shape[0]"
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
        super().__init__(np.float32(xyz)*np.float32(l_curr_over_target), np.float32(masses)*np.float32(m_curr_over_target), cat_arr, np.float32(r200)[obj_pass.base.nonzero()[0]]*np.float32(l_curr_over_target), obj_size.base[obj_pass.base.nonzero()[0]], SNAP, np.float32(L_BOX)*np.float32(l_curr_over_target), np.int32(MIN_NUMBER_PTCS), CENTER, VIZ_DEST, CAT_DEST, SUFFIX)
        
        
        
############################################################################################################################
        
################################## User provides Gadget I, II or HDF5 snapshot #############################################
        
############################################################################################################################
cdef class DensProfsGadget(DensProfsBase):
    """ Class for density profile calculations for Gadget-style HDF5 data
    
    Its public methods are ``getXYZMasses()``, ``_getXYZMasses()``, ``getVelXYZ()``, 
    ``_getVelXYZ()``, ``getObjInfoGadget()``, ``getHeader()`` and those of ``DensProfsBase``: ``getR200()``, ``getIdxCat()``,  ``getMassesCenters()``, 
    ``_getMassesCenters()``, ``estDensProfs()``, ``fitDensProfs()``, ``estConcentrations()``,
    ``plotDensProfs()``, ``getObjInfo()``."""
    
    def __init__(self, SNAP_DEST, GROUP_DEST, SNAP, OBJ_TYPE, VIZ_DEST, CAT_DEST, RVIR_OR_R200 = 'Rvir', MIN_NUMBER_PTCS = 200, CENTER = 'mode'):
        """
        :param SNAP_DEST: where we can find the snapshot
        :type SNAP_DEST: string
        :param GROUP_DEST: where we can find the group files
        :type GROUP_DEST: string
        :param SNAP: snapshot identifier, e.g. '024'
        :type SNAP: string
        :param OBJ_TYPE: which simulation particles to consider, 'dm', 'gas' or 'stars'
        :type OBJ_TYPE: str
        :param VIZ_DEST: visualization folder
        :type VIZ_DEST: string
        :param CAT_DEST: catalogue destination
        :type CAT_DEST: string
        :param RVIR_OR_R200: 'Rvir' if we want quantities (e.g. D_LOGSTART) to be expressed 
            with respect to the virial radius R_vir, 'R200' for the overdensity radius R_200
        :type RVIR_OR_R200: str
        :param MIN_NUMBER_PTCS: minimum number of particles for object to qualify for morphology calculation
        :type MIN_NUMBER_PTCS: int
        :param CENTER: shape quantities will be calculated with respect to CENTER = 'mode' (point of highest density)
            or 'com' (center of mass) of each halo
        :type CENTER: str
        """
        self.SNAP_DEST = SNAP_DEST
        self.GROUP_DEST = GROUP_DEST
        self.RVIR_OR_R200 = RVIR_OR_R200
        self.OBJ_TYPE = OBJ_TYPE
        SUFFIX = '_{}_'.format(OBJ_TYPE)
        l_internal, m_internal, vel_internal = config.getLMVInternal()
        l_curr_over_target = config.InUnitLength_in_cm/l_internal
        # Import HDF5 Fof halo data
        nb_shs, sh_len, fof_sizes, group_r200 = getFoFSHData(self.GROUP_DEST, self.RVIR_OR_R200, getPartType(OBJ_TYPE))
        # Import particle data
        xyz = readgadget.read_block(self.SNAP_DEST,"POS ",ptype=[getPartType(self.OBJ_TYPE)]) # Should be in 3.085678e24 cm units
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
        L_BOX = np.float32(head.boxsize)
        super().__init__(xyz, masses, obj_cat, obj_r200, obj_size, SNAP, L_BOX*np.float32(l_curr_over_target), np.int32(MIN_NUMBER_PTCS), CENTER, VIZ_DEST, CAT_DEST, SUFFIX)
        
    def getXYZMasses(self): # Public Method
        """ Retrieve positions and masses of particles
        
        :return xyz, masses: positions in config.OutUnitLength_in_cm and masses 
            in config.OutUnitMass_in_g
        :rtype: (N2,3) floats, (N2,) floats"""
        xyz = readgadget.read_block(self.SNAP_DEST,"POS ",ptype=[getPartType(self.OBJ_TYPE)])
        masses = readgadget.read_block(self.SNAP_DEST,"MASS",ptype=[getPartType(self.OBJ_TYPE)])
        if rank == 0:
            l_internal, m_internal, vel_internal = config.getLMVInternal()
            l_curr_over_target = l_internal/config.OutUnitLength_in_cm
            m_curr_over_target = m_internal/config.OutUnitMass_in_g
            return xyz*l_curr_over_target, masses*m_curr_over_target
        else:
            del xyz; del masses
            return None, None
        
    def _getXYZMasses(self):
        """ Retrieve positions and masses of particles
        
        :return xyz, masses: positions in Mpc/h (internal length units) and masses in 10^10*M_sun*h^2/(Mpc)**3
        :rtype: (N2,3) floats, (N2,) floats"""
        xyz = readgadget.read_block(self.SNAP_DEST,"POS ",ptype=[getPartType(self.OBJ_TYPE)])
        masses = readgadget.read_block(self.SNAP_DEST,"MASS",ptype=[getPartType(self.OBJ_TYPE)])
        if rank == 0:
            return xyz, masses
        else:
            del xyz; del masses
            return None, None
    
    def getVelXYZ(self): # Public Method
        """ Retrieve velocities of particles
        
        :return velxyz: velocity array in config.OutUnitVelocity_in_cm_per_s
        :rtype: (N2,3) floats"""
        velxyz = readgadget.read_block(self.SNAP_DEST,"VEL ",ptype=[getPartType(self.OBJ_TYPE)])
        if rank == 0:
            l_internal, m_internal, vel_internal = config.getLMVInternal()
            v_curr_over_target = vel_internal/config.OutUnitVelocity_in_cm_per_s
            return velxyz*v_curr_over_target
        else:
            del velxyz
            return None
    
    def _getVelXYZ(self):
        """ Retrieve velocities of particles
        
        :return velxyz: velocity array in km/s
        :rtype: (N2,3) floats"""
        velxyz = readgadget.read_block(self.SNAP_DEST,"VEL ",ptype=[getPartType(self.OBJ_TYPE)])
        if rank == 0:
            return velxyz
        else:
            del velxyz
            return None
    
    def getR200(self): # Public Method
        """ Fetch R200 values
        
        :return obj_r200: R200 value of parent halos in config.OutUnitLength_in_cm
        :rtype: (N1,) floats"""
        
        # Import hdf5 data
        nb_shs, sh_len, fof_sizes, group_r200 = getFoFSHData(self.GROUP_DEST, self.RVIR_OR_R200, getPartType(self.OBJ_TYPE))
        # Raise Error message if empty
        if len(nb_shs) == 0:
            raise ValueError("No subhalos found in HDF5 files.")
        if rank == 0:
            # Construct catalogue
            obj_cat, obj_r200, obj_size = calcObjCat(nb_shs, sh_len, fof_sizes, group_r200, self.MIN_NUMBER_PTCS)
            del nb_shs; del sh_len; del fof_sizes; del group_r200; del obj_cat; del obj_size
            self.r200 = obj_r200
            l_internal, m_internal, vel_internal = config.getLMVInternal()
            l_curr_over_target = l_internal/config.OutUnitLength_in_cm
            return obj_r200*l_curr_over_target
        else:
            del nb_shs; del sh_len; del fof_sizes; del group_r200
            return None
        
    def getIdxCat(self): # Public Method
        """ Fetch catalogue
        
        :return idx_cat: contains indices of particles belonging to an object,
            obj_size: number of particles in each object
        :rtype: (N3) integers and (N1,) integers"""
        
        # Import hdf5 data
        nb_shs, sh_len, fof_sizes, group_r200 = getFoFSHData(self.GROUP_DEST, self.RVIR_OR_R200, getPartType(self.OBJ_TYPE))
        # Raise Error message if empty
        if len(nb_shs) == 0:
            raise ValueError("No subhalos found in HDF5 files.")
        if rank == 0:
            # Construct catalogue
            obj_cat, obj_r200, obj_size = calcObjCat(nb_shs, sh_len, fof_sizes, group_r200, self.MIN_NUMBER_PTCS)
            self.r200 = obj_r200
            del nb_shs; del sh_len; del fof_sizes; del group_r200
            return obj_cat, obj_size
        else:
            del nb_shs; del sh_len; del fof_sizes; del group_r200
            return None, None
        
    def getObjInfoGadget(self): # Public Method
        """ Print basic info about the objects"""
        print_status(rank,self.start_time,'Starting getObjInfoLocal() with snap {0}'.format(self.SNAP))
        
        self._getObjInfoBase(self.idx_cat.base, self.obj_size.base, self.OBJ_TYPE)
        nb_shs, sh_len, fof_sizes, group_r200 = getFoFSHData(self.GROUP_DEST, self.RVIR_OR_R200, getPartType(self.OBJ_TYPE))
        # Raise Error message if empty
        if len(nb_shs) == 0:
            raise ValueError("No subhalos found in HDF5 files.")
        print_status(rank, self.start_time, "More detailed info on central subhalo catalogue. The total number of objects with > 0 SHs is {0}".format(nb_shs[nb_shs != 0].shape[0]))
        print_status(rank, self.start_time, "The total number of objects is {0}".format(len(nb_shs)))
        print_status(rank, self.start_time, "The total number of SHs (subhalos) is {0}".format(len(sh_len)))
        print_status(rank, self.start_time, "The number of objects that have no SH is {0}".format(nb_shs[nb_shs == 0].shape[0]))
        print_status(rank, self.start_time, "The total number of objects (central subhalos) that have sufficient resolution is {0}".format(len([x for x in self.idx_cat.base if x != []])))
        del nb_shs; del sh_len; del fof_sizes; del group_r200
        
    def getHeader(self): # Header
        """ Get header of first file in snapshot"""
        print_status(rank,self.start_time,'Starting getHeader() with snap {0}'.format(self.SNAP))
        
        header = readgadget.header(self.SNAP_DEST)
        return header