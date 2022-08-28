#cython: language_level=3
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from cosmic_profiles.common.cosmic_base_class cimport CosmicBase
from cosmic_profiles.common.python_routines import print_status, isValidSelection
from cosmic_profiles.dens_profs.dens_profs_tools import drawDensProfs
from cosmic_profiles.gadget_hdf5.get_hdf5 import getHDF5GxData, getHDF5SHData, getHDF5ObjData
from cosmic_profiles.gadget_hdf5.gen_catalogues import calcObjCat
import time
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
        if rank == 0:
            return self.idx_cat.base, self.obj_size.base
        else:
            return None, None
    
    def getMassesCenters(self, list select):
        """ Calculate total mass and centers of objects
        
        :param select: index of first and last object to look at in the format [idx_first, idx_last]
        :type select: list containing two integers
        :return centers, m: centers and masses
        :rtype: (N,3) and (N,) floats"""
        if rank == 0:
            centers, ms = self.getMassesCentersBase(self.xyz.base, self.masses.base, self.idx_cat.base[select[0]:select[1]+1], self.obj_size.base[select[0]:select[1]+1])
            return centers, ms
        else:
            return None, None
    
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
        print_status(rank,self.start_time,'Starting {} estDensProfs() with snap {}'.format('direct binning' if direct_binning == True else 'kernel based', self.SNAP))
        isValidSelection(select, self.idx_cat.shape[0])
        if direct_binning == True and spherical == False:
            print_status(rank,self.start_time,'DensProfs objects cannot call estDensProfs(ROverR200, spherical) with spherical == False. Use DensShapeProfs instead.')
            return None
        if rank == 0:
            if direct_binning:
                dens_profs = self.getDensProfsSphDirectBinningBase(self.xyz.base, self.masses.base, self.r200.base[select[0]:select[1]+1], self.idx_cat.base[select[0]:select[1]+1], self.obj_size.base[select[0]:select[1]+1], np.float32(ROverR200))
            else:
                dens_profs = self.getDensProfsKernelBasedBase(self.xyz.base, self.masses.base, self.r200.base[select[0]:select[1]+1], self.idx_cat.base[select[0]:select[1]+1], self.obj_size.base[select[0]:select[1]+1], np.float32(ROverR200))
            return dens_profs
        else:
            return None
    
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
        print_status(rank,self.start_time,'Starting fitDensProfs() with snap {0}'.format(self.SNAP))
        if len(dens_profs) > select[1] - select[0] + 1:
            raise ValueError("The `select` argument is inconsistent with the `dens_profs` handed over to the `fitDensProfs()` function. Please double-check and use the same `select` as used for the density profile estimation!")
        if rank == 0:
            best_fits = self.getDensProfsBestFitsBase(np.float32(dens_profs), np.float32(ROverR200), self.r200.base[select[0]:select[1]+1], method)
            return best_fits
        else:
            return None
        
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
        print_status(rank,self.start_time,'Starting estConcentrations() with snap {0}'.format(self.SNAP))
        if len(dens_profs) > select[1] - select[0] + 1:
            raise ValueError("The `select` argument is inconsistent with the `dens_profs` handed over to the `estConcentrations()` function. Please double-check and use the same `select` as used for the density profile estimation!")
        if rank == 0:
            cs = self.getConcentrationsBase(np.float32(dens_profs), np.float32(ROverR200), self.r200.base[select[0]:select[1]+1], method)
            return cs
        else:
            return None
        
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
        print_status(rank,self.start_time,'Starting plotDensProfs() with snap {0}'.format(self.SNAP))
        if len(dens_profs) > select[1] - select[0] + 1:
            raise ValueError("The `select` argument is inconsistent with the `dens_profs` handed over to the `plotDensProfs()` function. Please double-check and use the same `select` as used for the density profile estimation!")
        obj_centers, obj_masses = self.getMassesCenters(select = select)
        
        if rank == 0:
            suffix = '_'
            drawDensProfs(VIZ_DEST, self.SNAP, self.r200.base[select[0]:select[1]+1], dens_profs_fit, ROverR200_fit, dens_profs, np.float32(ROverR200), obj_masses, obj_centers, method, nb_bins, self.start_time, self.MASS_UNIT, suffix = suffix)
            del obj_centers; del obj_masses; del ROverR200_fit; del dens_profs; del ROverR200
        else:
            del obj_centers; del obj_masses
            
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
        if self.OBJ_TYPE == 'dm':
            return 1
        elif self.OBJ_TYPE == 'stars':
            return 4
        else:
            assert self.OBJ_TYPE == 'gas'
            return 0
        
    def getXYZMasses(self):
        """ Retrieve positions and masses of objects
        
        :return xyz, masses: positions and masses
        :rtype: (N2,3) floats, (N2,) floats"""
        xyz, masses, velxyz = getHDF5ObjData(self.HDF5_SNAP_DEST, self.getPartType(self.OBJ_TYPE))
        del velxyz
        if rank == 0:
            return xyz, masses
        else:
            del xyz; del masses
            return None, None
        
    def getVelXYZ(self):
        """ Retrieve velocities of objects
        
        :return velxyz: velocity array
        :rtype: (N2,3) floats"""
        xyz, masses, velxyz = getHDF5ObjData(self.HDF5_SNAP_DEST, self.getPartType(self.OBJ_TYPE))
        del masses; del xyz
        if rank == 0:
            return velxyz
        else:
            del velxyz
            return None
        
    def getR200(self):
        """ Fetch R200 values
        
        :return obj_r200: R200 value of parent halos
        :rtype: (N1,) floats"""
        
        # Import hdf5 data
        nb_shs, sh_len, fof_sizes, group_r200 = getHDF5SHData(self.HDF5_GROUP_DEST, self.RVIR_OR_R200, self.getPartType(self.OBJ_TYPE))
        if rank == 0:
            # Construct catalogue
            obj_cat, obj_r200, obj_size = calcObjCat(nb_shs, sh_len, fof_sizes, group_r200, self.MIN_NUMBER_PTCS)
            del nb_shs; del sh_len; del fof_sizes; del group_r200; del obj_cat; del obj_size
            return obj_r200
        else:
            del nb_shs; del sh_len; del fof_sizes; del group_r200
            return None
        
    def getIdxCat(self):
        """ Fetch catalogue
        
        :return idx_cat: each row contains indices of particles belonging to an object,
            obj_size: number of particles in each object
        :rtype: (N1, N3) integers and (N1,) integers"""
        
        # Import hdf5 data
        nb_shs, sh_len, fof_sizes, group_r200 = getHDF5SHData(self.HDF5_GROUP_DEST, self.RVIR_OR_R200, self.getPartType(self.OBJ_TYPE))
        if rank == 0:
            # Construct catalogue
            obj_cat, obj_r200, obj_size = calcObjCat(nb_shs, sh_len, fof_sizes, group_r200, self.MIN_NUMBER_PTCS)
            del nb_shs; del sh_len; del fof_sizes; del group_r200; del obj_r200
            return obj_cat, obj_size
        else:
            del nb_shs; del sh_len; del fof_sizes; del group_r200
            return None, None
    
    def getMassesCenters(self, list select):
        """ Calculate total mass and centers of objects
        
        :param select: index of first and last object to look at in the format [idx_first, idx_last]
        :type select: list containing two integers
        :return centers, m: centers and masses
        :rtype: (N,3) and (N,) floats"""
        xyz, masses = self.getXYZMasses()
        idx_cat, obj_size = self.getIdxCat()
        if rank == 0:
            idx_cat_len = len(idx_cat)
            isValidSelection(select, idx_cat_len)
            centers, ms = self.getMassesCentersBase(xyz, masses, idx_cat[select[0]:select[1]+1], obj_size[select[0]:select[1]+1])
            del xyz; del masses; del idx_cat; del obj_size
            return centers, ms
        else:
            del xyz; del masses
            return None, None
        
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
        print_status(rank,self.start_time,'Starting {} estDensProfs() with snap {}'.format('direct binning' if direct_binning == True else 'kernel based', self.SNAP))
        if direct_binning == True and spherical == False:
            print_status(rank,self.start_time,'DensProfsHDF5 objects cannot call estDensProfs(ROverR200, spherical) with spherical == False. Use DensShapeProfs instead.')
            return None
        xyz, masses = self.getXYZMasses()
        idx_cat, obj_size = self.getIdxCat()
        if rank == 0:
            idx_cat_len = len(idx_cat)
            isValidSelection(select, idx_cat_len)
        if rank == 0:
            if direct_binning:
                dens_profs = self.getDensProfsSphDirectBinningBase(xyz, masses, self.r200.base[select[0]:select[1]+1], idx_cat[select[0]:select[1]+1], obj_size[select[0]:select[1]+1], np.float32(ROverR200))
            else:
                dens_profs = self.getDensProfsKernelBasedBase(xyz, masses, self.r200.base[select[0]:select[1]+1], idx_cat[select[0]:select[1]+1], obj_size[select[0]:select[1]+1], np.float32(ROverR200))
            del xyz; del masses
            return dens_profs
        else:
            del xyz; del masses
            return None
        
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
        print_status(rank,self.start_time,'Starting fitDensProfs() with snap {0}'.format(self.SNAP))
        if len(dens_profs) > select[1] - select[0] + 1:
            raise ValueError("The `select` argument is inconsistent with the `dens_profs` handed over to the `fitDensProfs()` function. Please double-check and use the same `select` as used for the density profile estimation!")
        xyz, masses = self.getXYZMasses()
        if rank == 0:
            best_fits = self.getDensProfsBestFitsBase(np.float32(dens_profs), np.float32(ROverR200), self.r200.base[select[0]:select[1]+1], method)
            del xyz; del masses
            return best_fits
        else:
            del xyz; del masses
            return None
    
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
        print_status(rank,self.start_time,'Starting estConcentrations() with snap {0}'.format(self.SNAP))
        if len(dens_profs) > select[1] - select[0] + 1:
            raise ValueError("The `select` argument is inconsistent with the `dens_profs` handed over to the `estConcentrations()` function. Please double-check and use the same `select` as used for the density profile estimation!")
        xyz, masses = self.getXYZMasses()
        if rank == 0:
            cs = self.getConcentrationsBase(np.float32(dens_profs), np.float32(ROverR200), self.r200.base[select[0]:select[1]+1], method)
            del xyz; del masses
            return cs
        else:
            del xyz; del masses
            return None
        
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
        print_status(rank,self.start_time,'Starting plotDensProfs() with snap {0}'.format(self.SNAP))
        if len(dens_profs) > select[1] - select[0] + 1:
            raise ValueError("The `select` argument is inconsistent with the `dens_profs` handed over to the `plotDensProfs()` function. Please double-check and use the same `select` as used for the density profile estimation!")
        obj_centers, obj_masses = self.getMassesCenters(select = select)
        
        if rank == 0:
            suffix = '_{}_'.format(self.OBJ_TYPE)
            drawDensProfs(VIZ_DEST, self.SNAP, self.r200.base[select[0]:select[1]+1], dens_profs_fit, ROverR200_fit, dens_profs, ROverR200, obj_masses, obj_centers, method, nb_bins, self.start_time, self.MASS_UNIT, suffix = suffix)
            del obj_centers; del obj_masses; del ROverR200_fit; del dens_profs; del ROverR200