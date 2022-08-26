#cython: language_level=3
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from common.cosmic_base_class cimport CosmicBase
from cosmic_profiles.common.python_routines import print_status, isValidSelection, getIdxCorr
from cosmic_profiles.dens_profs.dens_profs_tools import drawDensProfs
from cosmic_profiles.gadget_hdf5.get_hdf5 import getHDF5GxData, getHDF5SHDMData, getHDF5SHGxData, getHDF5DMData
from cosmic_profiles.gadget_hdf5.gen_catalogues import calcCSHCat, calcGxCat
import time
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

cdef class DensProfs(CosmicBase):
    """ Class for density profile calculations
    
    Its public methods are ``getIdxCat()``, ``getIdxCatSuffRes()``,
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
        self.idx_cat = idx_cat
        self.r200 = r200.base
       
    def getIdxCat(self):
        """ Fetch catalogue
        
        :return cat: list of indices defining the objects
        :rtype: list of length N1, each consisting of a list of int indices"""
        return self.idx_cat
    
    def getIdxCatSuffRes(self):
        """ Fetch catalogue, objects with insufficient resolution are set to empty list []
        
        :return cat: list of indices defining the objects
        :rtype: list of length N1, each consisting of a list of int indices"""
        idx_cat = self.idx_cat
        idx_cat = [x if len(x) >= self.MIN_NUMBER_PTCS else [] for x in idx_cat]
        return idx_cat
    
    def getMassesCenters(self, list select):
        """ Calculate total mass and centers of objects
        
        :param select: index of first and last object to look at in the format [idx_first, idx_last]
        :type select: list containing two integers
        :return centers, m: centers and masses
        :rtype: (N,3) and (N,) floats"""
        if rank == 0:
            centers, ms = self.getMassesCentersBase(self.xyz.base, self.masses.base, self.idx_cat[select[0]:select[1]+1], self.MIN_NUMBER_PTCS)
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
        isValidSelection(select, len(self.idx_cat))
        if direct_binning == True and spherical == False:
            print_status(rank,self.start_time,'DensProfs objects cannot call estDensProfs(ROverR200, spherical) with spherical == False. Use DensShapeProfs instead.')
            return None
        if rank == 0:
            if direct_binning:
                dens_profs = self.getDensProfsSphDirectBinningBase(self.xyz.base, self.masses.base, self.r200.base[select[0]:select[1]+1], self.idx_cat[select[0]:select[1]+1], self.MIN_NUMBER_PTCS, np.float32(ROverR200))
            else:
                dens_profs = self.getDensProfsKernelBasedBase(self.xyz.base, self.masses.base, self.r200.base[select[0]:select[1]+1], self.idx_cat[select[0]:select[1]+1], self.MIN_NUMBER_PTCS, np.float32(ROverR200))
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
            best_fits = self.getDensProfsBestFitsBase(np.float32(dens_profs), np.float32(ROverR200), self.r200.base[select[0]:select[1]+1], self.idx_cat[select[0]:select[1]+1], self.MIN_NUMBER_PTCS, method)
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
            cs = self.getConcentrationsBase(np.float32(dens_profs), np.float32(ROverR200), self.r200.base[select[0]:select[1]+1], self.idx_cat[select[0]:select[1]+1], self.MIN_NUMBER_PTCS, method)
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
            drawDensProfs(VIZ_DEST, self.SNAP, self.idx_cat[select[0]:select[1]+1], self.r200.base[select[0]:select[1]+1], dens_profs_fit, ROverR200_fit, dens_profs, np.float32(ROverR200), obj_masses, obj_centers, method, nb_bins, self.start_time, self.MASS_UNIT, suffix = suffix)
            del obj_centers; del obj_masses; del ROverR200_fit; del dens_profs; del ROverR200
        else:
            del obj_centers; del obj_masses
            
cdef class DensProfsHDF5(CosmicBase):
    """ Class for density profile calculations for Gadget-style HDF5 data
    
    Its public methods are ``getXYZMasses()``, ``getVelXYZ()``, 
    ``getIdxCat()``,  ``getMassesCenters()``, 
    ``estDensProfs()``, ``fitDensProfs()``, ``estConcentrations()``,
    ``plotDensProfs()``."""
    
    def __init__(self, str HDF5_SNAP_DEST, str HDF5_GROUP_DEST, str SNAP, float L_BOX, int MIN_NUMBER_PTCS, int MIN_NUMBER_STAR_PTCS, str CENTER, bint WANT_RVIR):
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
        :param MIN_NUMBER_PTCS: minimum number of DM particles for halo to qualify for morphology calculation
        :type MIN_NUMBER_PTCS: int
        :param MIN_NUMBER_STAR_PTCS: minimum number of star particles for galaxy to qualify for morphology calculation
        :type MIN_NUMBER_STAR_PTCS: int
        :param CENTER: shape quantities will be calculated with respect to CENTER = 'mode' (point of highest density)
            or 'com' (center of mass) of each halo
        :type CENTER: str
        :param WANT_RVIR: Whether or not we want quantities (e.g. D_LOGSTART) expressed 
            with respect to the virial radius R_vir or the overdensity radius R_200
        :type WANT_RVIR: boolean"""
        super().__init__(SNAP, L_BOX, MIN_NUMBER_PTCS, CENTER)
        self.HDF5_SNAP_DEST = HDF5_SNAP_DEST
        self.HDF5_GROUP_DEST = HDF5_GROUP_DEST
        self.MIN_NUMBER_STAR_PTCS = MIN_NUMBER_STAR_PTCS
        self.WANT_RVIR = WANT_RVIR
        
    def getXYZMasses(self, str obj_type = 'dm'):
        """ Retrieve positions and masses of DM/gx
        
        :param obj_type: either 'dm' or 'gx', depending on what catalogue we are looking at
        :type obj_type: string
        :return xyz, masses, MIN_NUMBER_PTCS: positions, masses, and minimum number of particles
        :rtype: (N2,3) floats, (N2,) floats, int"""
        if obj_type == 'dm':
            xyz, masses, velxyz = getHDF5DMData(self.HDF5_SNAP_DEST)
            del velxyz
            MIN_NUMBER_PTCS = self.MIN_NUMBER_PTCS
        else:
            xyz, nb_shs, masses, velxyz, is_star = getHDF5GxData(self.HDF5_SNAP_DEST, self.HDF5_GROUP_DEST)
            del nb_shs; del velxyz; del is_star
            MIN_NUMBER_PTCS = self.MIN_NUMBER_STAR_PTCS
        if rank == 0:
            return xyz, masses, MIN_NUMBER_PTCS
        else:
            del xyz; del masses
            return None, None, None
        
    def getVelXYZ(self, str obj_type = 'dm'):
        """ Retrieve velocities of DM/gx
        
        :param obj_type: either 'dm' or 'gx', depending on what catalogue we are looking at
        :type obj_type: string
        :return velxyz: velocity array
        :rtype: (N2,3) floats"""
        if obj_type == 'dm':
            xyz, masses, velxyz = getHDF5DMData(self.HDF5_SNAP_DEST)
            del xyz; del masses
        else:
            xyz, nb_shs, masses, velxyz, is_star = getHDF5GxData(self.HDF5_SNAP_DEST, self.HDF5_GROUP_DEST)
            del xyz; del nb_shs; del masses; del is_star
        if rank == 0:
            return velxyz
        else:
            del velxyz
            return None
    
    def getIdxCat(self, str obj_type = 'dm'):
        """ Fetch catalogue
        
        :param obj_type: either 'dm' or 'gx', depending on what catalogue we are looking at
        :type obj_type: string
        :return cat: list of indices defining the objects
        :rtype: list of length N1, each consisting of a list of int indices"""
        
        if rank != 0:
            return None
        if obj_type == 'dm':
            # Import hdf5 data
            nb_shs, sh_len, fof_dm_sizes, group_r200, halo_masses = getHDF5SHDMData(self.HDF5_GROUP_DEST, self.WANT_RVIR)
            # Construct catalogue
            h_cat, h_r200, h_pass = calcCSHCat(np.array(nb_shs), np.array(sh_len), np.array(fof_dm_sizes), group_r200, halo_masses, self.MIN_NUMBER_PTCS)
            nb_shs_vec = np.array(nb_shs)
            h_cat_l = [(np.ma.masked_where(h_cat[i-getIdxCorr(h_pass, i)] == 0, h_cat[i-getIdxCorr(h_pass, i)]).compressed()-1).tolist() if h_pass[i] == 1 else [] for i in range(len(nb_shs))]
            self.r200 = h_r200
            del nb_shs; del sh_len; del fof_dm_sizes; del group_r200; del h_cat; del halo_masses; del h_r200; del h_pass
            return h_cat_l
        else:
            if self.r200 is None:
                # Import hdf5 data
                nb_shs, sh_len, fof_dm_sizes, group_r200, halo_masses = getHDF5SHDMData(self.HDF5_GROUP_DEST, self.WANT_RVIR)
                # Construct catalogue
                h_cat, h_r200, h_pass = calcCSHCat(np.array(nb_shs), np.array(sh_len), np.array(fof_dm_sizes), group_r200, halo_masses, self.MIN_NUMBER_PTCS)
                nb_shs_vec = np.array(nb_shs)         
                self.r200 = h_r200
                del nb_shs; del sh_len; del fof_dm_sizes; del group_r200; del h_cat; del halo_masses; del h_r200
            # Import hdf5 data
            nb_shs, sh_len_gx, fof_gx_sizes = getHDF5SHGxData(self.HDF5_GROUP_DEST)
            # Construct catalogue
            gx_cat, gx_pass = calcGxCat(np.array(nb_shs), np.array(sh_len_gx), np.array(fof_gx_sizes), self.MIN_NUMBER_STAR_PTCS)
            gx_cat_l = [(np.ma.masked_where(gx_cat[i-getIdxCorr(gx_pass, i)] == 0, gx_cat[i-getIdxCorr(gx_pass, i)]).compressed()-1).tolist() if gx_pass[i] == 1 else [] for i in range(len(nb_shs))]
            del nb_shs; del sh_len_gx; del fof_gx_sizes; del gx_cat; del gx_pass
            return gx_cat_l
    
    def getMassesCenters(self, list select, str obj_type = 'dm'):
        """ Calculate total mass and centers of objects
        
        :param select: index of first and last object to look at in the format [idx_first, idx_last]
        :type select: list containing two integers
        :param obj_type: either 'dm' or 'gx', depending on what catalogue we are looking at
        :type obj_type: string
        :return centers, m: centers and masses
        :rtype: (N,3) and (N,) floats"""
        xyz, masses, MIN_NUMBER_PTCS = self.getXYZMasses(obj_type)
        if rank == 0:
            idx_cat = self.getIdxCat(obj_type)
            centers, ms = self.getMassesCentersBase(xyz, masses, idx_cat[select[0]:select[1]+1], MIN_NUMBER_PTCS)
            del xyz; del masses
            return centers, ms
        else:
            del xyz; del masses
            return None, None
        
    def estDensProfs(self, ROverR200, list select, bint direct_binning = True, bint spherical = True, str obj_type = 'dm'):
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
        :param obj_type: either 'dm' or 'gx', depending on what catalogue we are looking at
        :type obj_type: string
        :return: density profiles
        :rtype: (N2, r_res) floats"""
        print_status(rank,self.start_time,'Starting {} estDensProfs() with snap {}'.format('direct binning' if direct_binning == True else 'kernel based', self.SNAP))
        if direct_binning == True and spherical == False:
            print_status(rank,self.start_time,'DensProfsHDF5 objects cannot call estDensProfs(ROverR200, spherical) with spherical == False. Use DensShapeProfs instead.')
            return None
        xyz, masses, MIN_NUMBER_PTCS = self.getXYZMasses(obj_type)
        if rank == 0:
            idx_cat = self.getIdxCat(obj_type)
            isValidSelection(select, len(idx_cat))
            if direct_binning:
                dens_profs = self.getDensProfsSphDirectBinningBase(xyz, masses, self.r200.base[select[0]:select[1]+1], idx_cat[select[0]:select[1]+1], MIN_NUMBER_PTCS, np.float32(ROverR200))
            else:
                dens_profs = self.getDensProfsKernelBasedBase(xyz, masses, self.r200.base[select[0]:select[1]+1], idx_cat[select[0]:select[1]+1], MIN_NUMBER_PTCS, np.float32(ROverR200))
            del xyz; del masses; del idx_cat
            return dens_profs
        else:
            del xyz; del masses
            return None
        
    def fitDensProfs(self, dens_profs, ROverR200, str method, list select, str obj_type = 'dm'):
        """ Get best-fit results for density profile fitting
        
        :param dens_profs: density profiles to be fit, in units of M_sun*h^2/(Mpc)**3
        :type dens_profs: (N3, r_res) floats
        :param ROverR200: normalized radii at which ``dens_profs`` are defined
        :type ROverR200: (r_res,) floats
        :param method: string describing density profile model assumed for fitting
        :type method: string, either `einasto`, `alpha_beta_gamma`, `hernquist`, `nfw`
        :param select: index of first and last object to look at in the format [idx_first, idx_last]
        :type select: list containing two integers
        :param obj_type: either 'dm' or 'gx', depending on what catalogue we are looking at
        :type obj_type: string
        :return: best-fits for each object
        :rtype: (N3, n) floats, where n is the number of free parameters in the model ``method``"""
        print_status(rank,self.start_time,'Starting fitDensProfs() with snap {0} for obj_type {1}'.format(self.SNAP, obj_type))
        if len(dens_profs) > select[1] - select[0] + 1:
            raise ValueError("The `select` argument is inconsistent with the `dens_profs` handed over to the `fitDensProfs()` function. Please double-check and use the same `select` as used for the density profile estimation!")
        xyz, masses, MIN_NUMBER_PTCS = self.getXYZMasses(obj_type)
        if rank == 0:
            idx_cat = self.getIdxCat(obj_type)
            best_fits = self.getDensProfsBestFitsBase(np.float32(dens_profs), np.float32(ROverR200), self.r200.base[select[0]:select[1]+1], idx_cat[select[0]:select[1]+1], MIN_NUMBER_PTCS, method)
            del xyz; del masses; del idx_cat
            return best_fits
        else:
            del xyz; del masses
            return None
    
    def estConcentrations(self, dens_profs, ROverR200, str method, list select, str obj_type = 'dm'):
        """ Get best-fit concentration values of objects from density profile fitting
        
        :param dens_profs: density profiles to be fit, in units of M_sun*h^2/(Mpc)**3
        :type dens_profs: (N3, r_res) floats
        :param ROverR200: normalized radii at which ``dens_profs`` are defined
        :type ROverR200: (r_res,) floats
        :param method: string describing density profile model assumed for fitting
        :type method: string, either `einasto`, `alpha_beta_gamma`, `hernquist`, `nfw`
        :param select: index of first and last object to look at in the format [idx_first, idx_last]
        :type select: list containing two integers
        :param obj_type: either 'dm' or 'gx', depending on what catalogue we are looking at
        :type obj_type: string
        :return: best-fit concentration for each object
        :rtype: (N3,) floats"""
        print_status(rank,self.start_time,'Starting estConcentrations() with snap {0} for obj_type {1}'.format(self.SNAP, obj_type))
        if len(dens_profs) > select[1] - select[0] + 1:
            raise ValueError("The `select` argument is inconsistent with the `dens_profs` handed over to the `estConcentrations()` function. Please double-check and use the same `select` as used for the density profile estimation!")
        xyz, masses, MIN_NUMBER_PTCS = self.getXYZMasses(obj_type)
        if rank == 0:
            idx_cat = self.getIdxCat(obj_type)
            cs = self.getConcentrationsBase(np.float32(dens_profs), np.float32(ROverR200), self.r200.base[select[0]:select[1]+1], idx_cat[select[0]:select[1]+1], MIN_NUMBER_PTCS, method)
            del xyz; del masses; del idx_cat
            return cs
        else:
            del xyz; del masses
            return None
        
    def plotDensProfs(self, dens_profs, ROverR200, dens_profs_fit, ROverR200_fit, str method, int nb_bins, str VIZ_DEST, list select, str obj_type = 'dm'):
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
        :param obj_type: either 'dm' or 'gx', depending on what catalogue we are looking at
        :type obj_type: string
        """
        print_status(rank,self.start_time,'Starting plotDensProfs() with snap {0} for obj_type {1}'.format(self.SNAP, obj_type))
        if len(dens_profs) > select[1] - select[0] + 1:
            raise ValueError("The `select` argument is inconsistent with the `dens_profs` handed over to the `plotDensProfs()` function. Please double-check and use the same `select` as used for the density profile estimation!")
        if rank == 0:
            idx_cat = self.getIdxCat(obj_type)
            idx_cat_len = len(idx_cat)
        else:
            idx_cat_len = None
        idx_cat_len = comm.bcast(idx_cat_len, root = 0)
        obj_centers, obj_masses = self.getMassesCenters(select = select, obj_type = obj_type)
        
        if rank == 0:
            suffix = '_{}_'.format(obj_type)
            drawDensProfs(VIZ_DEST, self.SNAP, idx_cat[select[0]:select[1]+1], self.r200.base[select[0]:select[1]+1], dens_profs_fit, ROverR200_fit, dens_profs, ROverR200, obj_masses, obj_centers, method, nb_bins, self.start_time, self.MASS_UNIT, suffix = suffix)
            del obj_centers; del obj_masses; del ROverR200_fit; del dens_profs; del ROverR200