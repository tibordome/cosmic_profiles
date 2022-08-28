#cython: language_level=3
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from common.cosmic_base_class cimport CosmicBase
import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

cdef class DensProfs(CosmicBase):
    """ Class for density profile calculations
    
    Its public methods are ``getIdxCat()``,
    ``getMassesCenters()``, ``estDensProfs``, ``fitDensProfs()``, ``estConcentrations()``, 
    ``plotDensProfs()``."""
    
    def __init__(self, float[:,:] xyz, float[:] masses, idx_cat, float[:] r200, str SNAP, float L_BOX, int MIN_NUMBER_PTCS, str CENTER):
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
        :type CENTER: str"""
        super().__init__(SNAP, L_BOX, MIN_NUMBER_PTCS, CENTER)
        assert xyz.shape[0] == masses.shape[0], "xyz.shape[0] must be equal to masses.shape[0]"
        self.xyz = xyz.base
        self.masses = masses.base
        cdef int nb_objs = len(idx_cat)
        cdef int p
        cdef int[:] obj_pass = np.zeros((nb_objs,), dtype = np.int32)
        cdef int[:] obj_size = np.zeros((nb_objs,), dtype = np.int32)
        for p in range(nb_objs):
            if len(idx_cat[p]) >= MIN_NUMBER_PTCS: # Only add objects that have sufficient resolution
                obj_pass[p] = 1      
                obj_size[p] = len(idx_cat[p]) 
        cdef int nb_pass = np.sum(obj_pass.base)
        cdef int[:,:] cat_arr = np.zeros((nb_pass,np.max([len(idx_cat[p]) for p in range(nb_objs)])), dtype = np.int32)
        cdef int[:] idxs_compr = np.zeros((nb_objs,), dtype = np.int32)
        idxs_compr.base[obj_pass.base.nonzero()[0]] = np.arange(np.sum(obj_pass.base))
        for p in range(nb_objs):
            if obj_pass[p] == 1:
                cat_arr.base[idxs_compr[p],:obj_size[p]] = np.array(idx_cat[p])
        self.idx_cat = cat_arr.base
        self.obj_size = obj_size.base[obj_pass.base.nonzero()[0]]
        self.r200 = r200.base[obj_pass.base.nonzero()[0]]
       
    def getIdxCat(self):
        """ Fetch catalogue
        
        :return idx_cat: each row contains indices of particles belonging to an object,
            obj_size: number of particles in each object
        :rtype: (N1, N3) integers and (N1,) integers"""
        return
    
    
    def getMassesCenters(self, list select):
        """ Calculate total mass and centers of objects
        
        :param select: index of first and last object to look at in the format [idx_first, idx_last]
        :type select: list containing two integers
        :return centers, m: centers and masses
        :rtype: (N,3) and (N,) floats"""
        return
    
    def estDensProfs(self, ROverR200, list select, bint direct_binning = True, bint spherical = True):
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
        :return: density profiles
        :rtype: (N2, r_res) floats"""
        return
    
    def fitDensProfs(self, dens_profs, ROverR200, str method, list select):
        """ Get best-fit results for density profile fitting
        
        :param dens_profs: density profiles to be fit, in units of M_sun*h^2/(Mpc)**3
        :type dens_profs: (N3, r_res) floats
        :param ROverR200: normalized radii at which ``dens_profs`` are defined
        :type ROverR200: (r_res,) floats
        :param method: string describing density profile model assumed for fitting
        :type method: string, either `einasto`, `alpha_beta_gamma`, `hernquist`, `nfw`
        :param select: index of first and last object to look at in the format [idx_first, idx_last]
        :type select: list containing two integers
        :return: best-fits for each object
        :rtype: (N3, n) floats, where n is the number of free parameters in the model ``method``"""
        return
        
    def estConcentrations(self, dens_profs, ROverR200, str method, list select):
        """ Get best-fit concentration values of objects from density profile fitting
        
        :param dens_profs: density profiles whose concentrations are to be determined, 
            in units of M_sun*h^2/(Mpc)**3
        :type dens_profs: (N3, r_res) floats
        :param ROverR200: normalized radii at which ``dens_profs`` are defined
        :type ROverR200: (r_res,) floats
        :param method: string describing density profile model assumed for fitting
        :type method: string, either `einasto`, `alpha_beta_gamma`, `hernquist`, `nfw`
        :param select: index of first and last object to look at in the format [idx_first, idx_last]
        :type select: list containing two integers
        :return: best-fit concentration for each object
        :rtype: (N3,) floats"""
        return
        
    def plotDensProfs(self, dens_profs, ROverR200, dens_profs_fit, ROverR200_fit, str method, int nb_bins, str VIZ_DEST, list select):
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
        :param nb_bins: Number of mass bins to plot density profiles for
        :type nb_bins: int
        :param VIZ_DEST: visualization folder
        :type VIZ_DEST: string
        :param select: index of first and last object to look at in the format [idx_first, idx_last]
        :type select: list containing two integers
        """
        return
            
cdef class DensProfsHDF5(CosmicBase):
    """ Class for density profile calculations for Gadget-style HDF5 data
    
    Its public methods are ``getXYZMasses()``, ``getVelXYZ()``, 
    ``getIdxCat()``,  ``getMassesCenters()``, 
    ``estDensProfs()``, ``fitDensProfs()``, ``estConcentrations()``,
    ``plotDensProfs()``."""
    
    def __init__(self, str HDF5_SNAP_DEST, str HDF5_GROUP_DEST, str SNAP, float L_BOX, int MIN_NUMBER_PTCS, str OBJ_TYPE, str CENTER, str RVIR_OR_R200):
        """
        :param HDF5_SNAP_DEST: where we can find the snapshot
        :type HDF5_SNAP_DEST: string
        :param HDF5_GROUP_DEST: where we can find the group files
        :type HDF5_GROUP_DEST: string
        :param SNAP: e.g. '024'
        :type SNAP: string
        :param SNAP: snapshot identifier, e.g. '024'
        :type SNAP: string
        :param L_BOX: simulation box side length
        :type L_BOX: float, units: Mpc/h
        :param MIN_NUMBER_PTCS: minimum number of particles for object to qualify for morphology calculation
        :type MIN_NUMBER_PTCS: int
        :param CENTER: shape quantities will be calculated with respect to CENTER = 'mode' (point of highest density)
            or 'com' (center of mass) of each halo
        :type CENTER: str
        :param RVIR_OR_R200: 'Rvir' if we want quantities (e.g. D_LOGSTART) to be expressed 
            with respect to the virial radius R_vir, 'R200' for the overdensity radius R_200
        :type RVIR_OR_R200: str"""
        super().__init__(SNAP, L_BOX, MIN_NUMBER_PTCS, CENTER)
        self.HDF5_SNAP_DEST = HDF5_SNAP_DEST
        self.HDF5_GROUP_DEST = HDF5_GROUP_DEST
        self.RVIR_OR_R200 = RVIR_OR_R200
        self.OBJ_TYPE = OBJ_TYPE
        
    def getPartType(self):
        """ Return particle type number
        
        :returns: particle type number
        :rtype: int"""
        return
    
    def getXYZMasses(self):
        """ Retrieve positions and masses of objects
        
        :return xyz, masses: positions and masses
        :rtype: (N2,3) floats, (N2,) floats"""
        return
        
    def getVelXYZ(self):
        """ Retrieve velocities of objects
        
        :return velxyz: velocity array
        :rtype: (N2,3) floats"""
        return
    
    def getR200(self):
        """ Fetch R200 values
        
        :return obj_r200: R200 value of parent halos
        :rtype: (N1,) floats"""
        return
    
    def getIdxCat(self):
        """ Fetch catalogue
        
        :return idx_cat: each row contains indices of particles belonging to an object,
            obj_size: number of particles in each object
        :rtype: (N1, N3) integers and (N1,) integers"""
        return
    
    def getMassesCenters(self, list select):
        """ Calculate total mass and centers of objects
        
        :param select: index of first and last object to look at in the format [idx_first, idx_last]
        :type select: list containing two integers
        :return centers, m: centers and masses
        :rtype: (N,3) and (N,) floats"""
        return
        
    def estDensProfs(self, ROverR200, list select, bint direct_binning = True, bint spherical = True):
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
        :return: density profiles
        :rtype: (N2, r_res) floats"""
        return
        
    def fitDensProfs(self, dens_profs, ROverR200, str method, list select):
        """ Get best-fit results for density profile fitting
        
        :param dens_profs: density profiles to be fit, in units of M_sun*h^2/(Mpc)**3
        :type dens_profs: (N3, r_res) floats
        :param ROverR200: normalized radii at which ``dens_profs`` are defined
        :type ROverR200: (r_res,) floats
        :param method: string describing density profile model assumed for fitting
        :type method: string, either `einasto`, `alpha_beta_gamma`, `hernquist`, `nfw`
        :param select: index of first and last object to look at in the format [idx_first, idx_last]
        :type select: list containing two integers
        :return: best-fits for each object
        :rtype: (N3, n) floats, where n is the number of free parameters in the model ``method``"""
        return
    
    def estConcentrations(self, dens_profs, ROverR200, str method, list select):
        """ Get best-fit concentration values of objects from density profile fitting
        
        :param dens_profs: density profiles to be fit, in units of M_sun*h^2/(Mpc)**3
        :type dens_profs: (N3, r_res) floats
        :param ROverR200: normalized radii at which ``dens_profs`` are defined
        :type ROverR200: (r_res,) floats
        :param method: string describing density profile model assumed for fitting
        :type method: string, either `einasto`, `alpha_beta_gamma`, `hernquist`, `nfw`
        :param select: index of first and last object to look at in the format [idx_first, idx_last]
        :type select: list containing two integers
        :return: best-fit concentration for each object
        :rtype: (N3,) floats"""
        return
        
    def plotDensProfs(self, dens_profs, ROverR200, dens_profs_fit, ROverR200_fit, str method, int nb_bins, str VIZ_DEST, list select):
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
        :param nb_bins: Number of mass bins to plot density profiles for
        :type nb_bins: int
        :param VIZ_DEST: visualization folder
        :type VIZ_DEST: string
        :param select: index of first and last object to look at in the format [idx_first, idx_last]
        :type select: list containing two integers
        """
        return