#cython: language_level=3
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from common.cosmic_base_class cimport CosmicBase
from common.python_routines import default_katz_config
from gadget.read_fof import getFoFSHData, getPartType
from gadget import readgadget
from gadget.gen_catalogues import calcObjCat
import numpy as np
from common import config
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
        return
    
    def getIdxCat(self): # Public Method
        """ Fetch catalogue
        
        :return idx_cat: contains indices of particles belonging to an object,
            obj_size: number of particles in each object
        :rtype: (N3) integers and (N1,) integers"""
        return
        
    def getXYZMasses(self): # Public Method
        """ Retrieve positions and masses of particles
        
        :return xyz, masses: positions in config.OutUnitLength_in_cm and masses 
            in config.OutUnitMass_in_g
        :rtype: (N2,3) floats, (N2,) floats"""
        return
            
    def getMassesCenters(self, obj_numbers): # Public Method
        """ Calculate total mass and centers of objects 
        
        Note that the units will be in config.OutUnitLength_in_cm and config.OutUnitMass_in_g.
        
        :param obj_numbers: list of object indices of interest
        :type obj_numbers: list of int
        :return centers, m: centers in config.OutUnitLength_in_cm and masses in config.OutUnitMass_in_g
        :rtype: structured array, containing (N,3) and (N,) floats"""
        return
    
    def _getMassesCenters(self, obj_numbers):
        """ Calculate total mass and centers of objects
        
        :param obj_numbers: list of object indices of interest
        :type obj_numbers: list of int
        :return centers, m: centers in Mpc/h (internal length units) and masses in 10^10*M_sun*h^2/(Mpc)**3
        :rtype: (N,3) and (N,) floats"""
        return
        
    def estDensProfs(self, r_over_r200, obj_numbers, bint direct_binning = True, bint spherical = True, katz_config = default_katz_config): # Public Method
        """ Estimate density profiles
        
        :param r_over_r200: normalized radii at which to-be-estimated density profiles are defined
        :type r_over_r200: (r_res,) floats
        :param obj_numbers: list of object indices of interest
        :type obj_numbers: list of int
        :param direct_binning: whether or not direct binning approach or
            kernel-based approach should be used
        :type direct_binning: boolean
        :param spherical: whether or not spherical shell-based or ellipsoidal shell-based
            should be used, ignored if ``direct_binning`` = False
        :type spherical: boolean
        :param katz_config: dictionary with parameters to the Katz algorithm, with fields 'ROverR200', 'IT_TOL', 'IT_WALL', 'IT_MIN', 'REDUCED', 'SHELL_BASED'
        :type katz_config: dictionary
        :return: density profiles in units of config.OutUnitMass_in_g/config.OutUnitLength_in_cm**3
        :rtype: (N2, r_res) floats"""
        return
    
    def fitDensProfs(self, dens_profs, r_over_r200, str method, obj_numbers): # Public Method
        """ Get best-fit results for density profile fitting
        
        :param dens_profs: density profiles to be fit, in units of config.OutUnitMass_in_g/config.OutUnitLength_in_cm**3
        :type dens_profs: (N3, r_res) floats
        :param r_over_r200: normalized radii at which ``dens_profs`` are defined
        :type r_over_r200: (r_res,) floats
        :param method: string describing density profile model assumed for fitting
        :type method: string, either `einasto`, `alpha_beta_gamma`, `hernquist`, `nfw`
        :param obj_numbers: list of object indices of interest
        :type obj_numbers: list of int
        :return: best-fits for each object
        :rtype: structured array, containing (N3, n) floats, where n is the number of free parameters in the model ``method``"""
        return
        
    def estConcentrations(self, dens_profs, r_over_r200, str method, obj_numbers): # Public Method
        """ Get best-fit concentration values of objects from density profile fitting
        
        :param dens_profs: density profiles whose concentrations are to be determined, 
            in units of config.OutUnitMass_in_g/config.OutUnitLength_in_cm**3
        :type dens_profs: (N3, r_res) floats
        :param r_over_r200: normalized radii at which ``dens_profs`` are defined
        :type r_over_r200: (r_res,) floats
        :param method: string describing density profile model assumed for fitting
        :type method: string, either `einasto`, `alpha_beta_gamma`, `hernquist`, `nfw`
        :param obj_numbers: list of object indices of interest
        :type obj_numbers: list of int
        :return: best-fit concentration for each object
        :rtype: (N3,) floats"""
        return
        
    def plotDensProfs(self, dens_profs, r_over_r200, dens_profs_fit, r_over_r200_fit, str method, int nb_bins, obj_numbers): # Public Method
        """ Draws some simplistic density profiles
        
        :param dens_profs: estimated density profiles, in units of 
            config.OutUnitMass_in_g/config.OutUnitLength_in_cm**3
        :type dens_profs: (N2, r_res) floats
        :param r_over_r200: radii at which ``dens_profs`` are defined
        :type r_over_r200: (r_res,) floats
        :param dens_profs_fit: density profiles to be fit, in units of 
            config.OutUnitMass_in_g/config.OutUnitLength_in_cm**3
        :type dens_profs_fit: (N2, r_res2) floats
        :param r_over_r200_fit: radii at which best-fits shall be calculated
        :type r_over_r200_fit: (r_res2,) floats
        :param method: string describing density profile model assumed for fitting
        :type method: string, either `einasto`, `alpha_beta_gamma`, `hernquist`, `nfw`
        :param nb_bins: Number of mass bins to plot density profiles for
        :type nb_bins: int
        :param obj_numbers: list of object indices of interest
        :type obj_numbers: list of int
        """
        return
    
    def getObjInfo(self): # Public Method
        """ Print basic info about the objects"""
        return