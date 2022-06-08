#cython: language_level=3
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 13:17:51 2021
"""

import numpy as np
from libc.math cimport sqrt
from copy import deepcopy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot
from matplotlib.font_manager import FontProperties
cimport openmp
from libc.math cimport isnan
from cython.parallel import prange
from libc.stdio cimport printf
from matplotlib.font_manager import FontProperties
import json
import h5py
from matplotlib import pyplot
cimport cython
import os
from python_helpers import print_status, set_axes_equal, fibonacci_ellipsoid, drawUniformFromEllipsoid, getMassDMParticle
from mpl_toolkits.mplot3d import Axes3D
from get_hdf5 import getHDF5Data, getHDF5GxData, getHDF5SHDMData, getHDF5DMData
from gen_csh_gx_cat import getCSHCat, getGxCat
from nbodykit.lab import cosmology, LogNormalCatalog
from pynverse import inversefunc
from scipy.integrate import quad
import math
import time
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

@cython.embedsignature(True)
def createLogNormUni(BoxSize, nbar, redshift, Nmesh, UNIT_MASS):
    """ Create mock simulation box by Poisson-sampling a lognormal density distribution
    
    The Poisson-sampled distribution is evolved according to the Zeldovich (1LPT) prescription
    up until redshift ``redshift`` under the constraint of an 'EisensteinHu' power spectrum
    
    :param BoxSize: size of to-be-obtained simulation box
    :type BoxSize: float
    :param nbar: number density of points (i.e. sampling density / resolution) in box, units: 1/(cMpc/h)**3
        Note: ``nbar`` is assumed to be constant across the box
    :type nbar: float
    :param redshift: redshift of interest
    :type redshift: float
    :param Nmesh: the mesh size to use when generating the density and displacement fields, 
        which are Poisson-sampled to particles
    :type Nmesh: int
    :param UNIT_MASS: in units of solar masses / h. Returned masses will have units UNIT_MASS*(solar_mass)/h
    :type UNIT_MASS: float
    :return: total number of particles, xyz-coordinates of DM particles, xyz-values of DM particle velocities, 
        masses of the DM particles (all identical)
    :rtype: int, (N,) floats, (N,) floats, (N,) floats, (N,) floats, (N,) floats, (N,) floats, (N,) floats"""
    return

@cython.embedsignature(True)
def getAlphaBetaGammaProf(r, alpha, beta, gamma, rho_0, r_s):
    """ Get alpha-beta-gamma profile
    
    :param r: radii at which profile should be returned
    :type r: float array, units of Mpc/h
    :param alpha: ``alpha`` parameter in alpha-beta-gamma density profile
    :type alpha: float
    :param beta: ``beta`` parameter in alpha-beta-gamma density profile 
    :type beta: float
    :param gamma: ``gamma`` parameter in alpha-beta-gamma density profile 
    :type gamma: float
    :param rho_0: ``rho_0`` parameter in alpha-beta-gamma density profile (density at the center)
    :type rho_0: float, units are M_sun*h^2/Mpc^3
    :param r_s: ``r_s`` parameter in alpha-beta-gamma density profile (scale radius)
    :type r_s: float
    :return: profile values
    :rtype: float array
    """
    
    return

@cython.embedsignature(True)
def genAlphaBetaGammaHalo(N0, alpha, beta, gamma, rho_0, r_s, a, b, c):
    """ Mock halo generator
    
    Create mock halo consisting of ``N0`` particles in 1st shell. The alpha-beta-gamma 
    density profile is a generalization of the Navarro-Frank-White (NFW) profile. Its definition
    can be looked up in Zemp et al 2011, https://arxiv.org/abs/1107.5582.
    
    :param N0: number of particles in 1st shell. Will be scaled appropriately 
        in each shell to satisfy alpha-beta-gamma profile
    :type N0: int
    :param alpha: ``alpha`` parameter in alpha-beta-gamma density profile
    :type alpha: float
    :param beta: ``beta`` parameter in alpha-beta-gamma density profile 
    :type beta: float
    :param gamma: ``gamma`` parameter in alpha-beta-gamma density profile 
    :type gamma: float
    :param rho_0: ``rho_0`` parameter in alpha-beta-gamma density profile (density at the center)
    :type rho_0: float, units are M_sun*h^2/Mpc^3
    :param r_s: ``r_s`` parameter in alpha-beta-gamma density profile (scale radius)
    :type r_s: float
    :param a: major axis array
    :type a: float array, units are Mpc/h
    :param b: intermediate axis array
    :type b: float array, units are Mpc/h
    :param c: minor axis array
    :type c: float array, units are Mpc/h
    :return: halo_x, halo_y, halo_z: arrays containing positions of halo particles, 
        mass_dm: mass of each DM ptc in units of M_sun/h
    :rtype: 3 (Nreals,) float arrays, 1 float
    """
    return

@cython.embedsignature(True)
cdef class CosmicShapes:
    """ Parent class governing low-level cosmic shape calculations
    
    Its public methods are ``runS1()``, ``runE1()``, ``runE1VelDisp()``, ``getObjMorphLocal()``, ``getObjMorphGlobal()``, 
    ``getObjMorphLocalVelDisp()``, ``getObjMorphGlobalVelDisp()``, ``getMorphLocal()``, ``getMorphGlobal()``, 
    ``getMorphLocalVelDisp()``, ``getMorphGlobalVelDisp()``, ``drawShapeCurves()``, ``plotLocalTHisto()``"""
    cdef str CAT_DEST
    cdef str VIZ_DEST
    cdef float L_BOX
    cdef int MIN_NUMBER_PTCS
    cdef int D_LOGSTART
    cdef int D_LOGEND
    cdef int D_BINS
    cdef float M_TOL
    cdef int N_WALL
    cdef int N_MIN
    cdef str CENTER
    cdef float SAFE # Units: cMpc/h. Ellipsoidal radius will be maxdist(COM,point)+SAFE where point is any point in the point cloud. The larger the better.
    cdef double start_time
    
    def __init__(self, str CAT_DEST, str VIZ_DEST, float L_BOX, int MIN_NUMBER_PTCS, int D_LOGSTART, int D_LOGEND, int D_BINS, float M_TOL, int N_WALL, int N_MIN, str CENTER, double start_time):
        """        
        :param CAT_DEST: catalogue destination
        :type CAT_DEST: string
        :param VIZ_DEST: visualisation folder destination
        :type VIZ_DEST: string
        :param L_BOX: simulation box side length
        :type L_BOX: float, units: cMpc/h
        :param MIN_NUMBER_PTCS: minimum number of particles for object to qualify for morphology calculation
        :type MIN_NUMBER_PTCS: int
        :param D_LOGSTART: logarithm of minimum ellipsoidal radius of interest, in units of R200 of parent halo
        :type D_LOGSTART: int
        :param D_LOGEND: logarithm of maximum ellipsoidal radius of interest, in units of R200 of parent halo
        :type D_LOGEND: int
        :param D_BINS: number of ellipsoidal radii of interest minus 1 (i.e. number of bins)
        :type D_BINS: int
        :param M_TOL: convergence tolerance, eigenvalue fractions must differ by less than ``M_TOL``
            for iteration to stop
        :type M_TOL: float
        :param N_WALL: maximum permissible number of iterations
        :type N_WALL: float
        :param N_MIN: minimum number of particles (DM or star particle) in any iteration; 
            if undercut, shape is unclassified
        :type N_MIN: int
        :param CENTER: shape quantities will be calculated with respect to CENTER = 'mode' (point of highest density)
            or 'com' (center of mass) of each halo
        :type CENTER: str
        :param start_time: time of start of object initialization
        :type start_time: float"""
        
        self.CAT_DEST = CAT_DEST
        self.VIZ_DEST = VIZ_DEST
        self.L_BOX = L_BOX
        self.MIN_NUMBER_PTCS = MIN_NUMBER_PTCS
        self.D_LOGSTART = D_LOGSTART
        self.D_LOGEND = D_LOGEND
        self.D_BINS = D_BINS
        self.M_TOL = M_TOL
        self.N_WALL = N_WALL
        self.N_MIN = N_MIN
        self.CENTER = CENTER
        self.SAFE = 6
        self.start_time = start_time
        
    def getDensProfs(cat, float[:,:] xyz, int[:] obj_keep, float[:] masses, float[:,:] centers, float[:] r200s, float[:] ROverR200, float L_BOX, str CENTER):
        """ Calculates density profiles for objects defined by indices found in `cat`
        
        
        The number of enclosed particles calculation assumes masses are identical,
        though easy to update.
        
        :param cat: list of indices defining the objects
        :type cat: list of length N1, each consisting of a list of int indices
        :param xyz: positions of all simulation particles
        :type xyz: (N2,3) floats, N2 >> N1
        :param obj_keep: which objects among the N1 different ones to consider. 1: keep, 0: ignore
            This can be used to select objects within a certain mass range, for instance. Having
            a 1 where `cat` has an empty list entry is not permitted.
        :type obj_keep: (N1,) ints
        :param masses: masses of all simulation particles
        :type masses: (N2,) floats
        :param centers: centers of the objects
        :type centers: (N1,3) floats
        :param r200s: R200 values of the objects
        :type r200s: (N1,) floats
        :param ROverR200: radii at which the density profiles should be calculated,
            normalized by R200
        :type ROverR200: (N3,) float array
        :param L_BOX: box size
        :type L_BOX: float
        :param CENTER: density profiles will be calculated with respect to CENTER = 'mode' (point of highest density)
            or 'com' (center of mass) of each halo
        :type CENTER: str
        :return: ROverR200 array, enclosed mass profiles, enclosed average density profiles, 
            enclosed number of particles profiles, density profiles
        :rtype: (N3,) float array, (nb_keep, N3) float and int arrays"""
        
        return
    
    def runS1(self, float[:] morph_info, float[:,:] xyz, float[:,:] xyz_princ, float[:] masses, int[:] shell, float[:] center, complex[::1,:] shape_tensor, double[::1] eigval, complex[::1,:] eigvec, float d, float delta_d, float M_TOL, int N_WALL, int N_MIN, str CENTER):
        """ S1 algorithm for halos/galaxies at elliptical radius ``d`` with shell width ``delta_d``
        
        Calculates the axis ratios at a distance ``d`` from the center of the entire particle distro.\n
        Note that before and during the iteration, ``d`` is defined with respect to the center of 
        the entire particle distro, not the center of the initial spherical volume as in Katz 1991.\n
        Differential version of E1.\n
        Shells can cross (except 2nd shell with 1st), and a shell is assumed to be equally thick everywhere.\n
        Whether we adopt the last assumption or let the thickness float (Tomassetti et al 2016) barely makes 
        any difference in terms of shapes found, but the convergence properties improve for the version with fixated thickness.
        For 1st shell: ``delta_d`` is ``d``
        
        :param morph_info: Array to be filled with morphological info. 1st entry: d,
            2nd entry: q, 3rd entry: s, 4th to 6th: normalized major axis, 7th to 9th: normalized intermediate axis,
            10th to 12th: normalized minor axis
        :type morph_info: (12,) floats
        :param xyz: position array
        :type xyz: (N x 3) floats
        :param xyz_princ: position arrays transformed into principal frame (varies from iteration to iteration)
        :type xyz_princ: (N x 3) floats, zeros
        :param masses: mass array
        :type masses: (N x 1) floats
        :param shell: indices of points that fall into shell (varies from iteration to iteration)
        :type shell: (N,) ints, zeros
        :param center: center of point cloud
        :type center: (3,) floats
        :param shape_tensor: shape tensor array to be filled
        :type shape_tensor: (3,3) complex, zeros
        :param eigval: eigenvalue array to be filled
        :type eigval: (3,) double, zeros
        :param eigvec: eigenvector array to be filled
        :type eigvec: (3,3) double, zeros
        :param d: distance from the center, kept fixed during iterative procedure
        :type d: float
        :param delta_d: thickness of the shell in real space (constant across shells in logarithmic space)
        :type delta_d: float
        :param M_TOL: convergence tolerance, eigenvalue fractions must differ by less than ``M_TOL``
            for iteration to stop
        :type M_TOL: float
        :param N_WALL: maximum permissible number of iterations
        :type N_WALL: float
        :param N_MIN: minimum number of particles (DM or star particle) in any iteration; 
            if undercut, shape is unclassified
        :type N_MIN: int
        :param CENTER: shape quantities will be calculated with respect to CENTER = 'mode' (point of highest density)
            or 'com' (center of mass) of each halo
        :type CENTER: str
        :return: ``morph_info`` containing d, q, s, eigframe info
        :rtype: (12,) float array"""
        
        return
    
    
    def runE1(self, float[:] morph_info, float[:,:] xyz, float[:,:] xyz_princ, float[:] masses, int[:] ellipsoid, float[:] center, complex[::1,:] shape_tensor, double[::1] eigval, complex[::1,:] eigvec, float d, float M_TOL, int N_WALL, int N_MIN, str CENTER):
        """ E1 algorithm for halos/galaxies at elliptical radius ``d``
        
        Calculates the axis ratios at a distance ``d`` from the center of the entire particle distro.\n
        Note that before and during the iteration, ``d`` is defined with respect to the center of 
        the entire particle distro, not the center of the initial spherical volume as in Katz 1991.\n
        
        :param morph_info: Array to be filled with morphological info. 1st entry: d,
            2nd entry: q, 3rd entry: s, 4th to 6th: normalized major axis, 7th to 9th: normalized intermediate axis,
            10th to 12th: normalized minor axis
        :type morph_info: (12,) floats
        :param xyz: position array
        :type xyz: (N x 3) floats
        :param xyz_princ: position arrays transformed into principal frame (varies from iteration to iteration)
        :type xyz_princ: (N x 3) floats, zeros
        :param masses: mass array
        :type masses: (N x 1) floats
        :param ellipsoid: indices of points that fall into ellipsoid (varies from iteration to iteration)
        :type ellipsoid: (N,) ints, zeros
        :param center: center of point cloud
        :type center: (3,) floats
        :param shape_tensor: shape tensor array to be filled
        :type shape_tensor: (3,3) complex, zeros
        :param eigval: eigenvalue array to be filled
        :type eigval: (3,) double, zeros
        :param eigvec: eigenvector array to be filled
        :type eigvec: (3,3) double, zeros
        :param d: distance from the center, kept fixed during iterative procedure
        :type d: float
        :param M_TOL: convergence tolerance, eigenvalue fractions must differ by less than ``M_TOL``
            for iteration to stop
        :type M_TOL: float
        :param N_WALL: maximum permissible number of iterations
        :type N_WALL: float
        :param N_MIN: minimum number of particles (DM or star particle) in any iteration; 
            if undercut, shape is unclassified
        :type N_MIN: int
        :param CENTER: shape quantities will be calculated with respect to CENTER = 'mode' (point of highest density)
            or 'com' (center of mass) of each halo
        :type CENTER: str
        :return: ``morph_info`` containing d, q, s, eigframe info
        :rtype: (12,) float array"""
        
        return
    
    def runE1VelDisp(self, float[:] morph_info, float[:,:] xyz, float[:,:] vxyz, float[:,:] xyz_princ, float[:] masses, int[:] ellipsoid, float[:] center, float[:] vcenter, complex[::1,:] shape_tensor, double[::1] eigval, complex[::1,:] eigvec, float d, float M_TOL, int N_WALL, int N_MIN, str CENTER):
        """ Similar to ``E1_obj`` algorithm for halos/galaxies but for velocity dispersion tensor
        
        Calculates the axis ratios at a distance ``d`` from the center of the entire particle distro.\n
        Note that before and during the iteration, ``d`` is defined with respect to the center of 
        the entire particle distro, not the center of the initial spherical volume as in Katz 1991.\n
        
        :param morph_info: Array to be filled with morphological info. 1st entry: d,
            2nd entry: q, 3rd entry: s, 4th to 6th: normalized major axis, 7th to 9th: normalized intermediate axis,
            10th to 12th: normalized minor axis
        :type morph_info: (12,) floats
        :param xyz: position array
        :type xyz: (N x 3) floats
        :param vxyz: velocity array
        :type vxyz: (N x 3) floats
        :param xyz_princ: position arrays transformed into principal frame (varies from iteration to iteration)
        :type xyz_princ: (N x 3) floats, zeros
        :param masses: mass array
        :type masses: (N x 1) floats
        :param ellipsoid: indices of points that fall into ellipsoid (varies from iteration to iteration)
        :type ellipsoid: (N,) ints, zeros
        :param center: center of point cloud
        :type center: (3,) floats
        :param vcenter: velocity-center of point cloud
        :type vcenter: (3,) floats
        :param shape_tensor: shape tensor array to be filled
        :type shape_tensor: (3,3) complex, zeros
        :param eigval: eigenvalue array to be filled
        :type eigval: (3,) double, zeros
        :param eigvec: eigenvector array to be filled
        :type eigvec: (3,3) double, zeros
        :param d: distance from the center, kept fixed during iterative procedure
        :type d: float
        :param M_TOL: convergence tolerance, eigenvalue fractions must differ by less than ``M_TOL``
            for iteration to stop
        :type M_TOL: float
        :param N_WALL: maximum permissible number of iterations
        :type N_WALL: float
        :param N_MIN: minimum number of particles (DM or star particle) in any iteration; 
            if undercut, shape is unclassified
        :type N_MIN: int
        :param CENTER: shape quantities will be calculated with respect to CENTER = 'mode' (point of highest density)
            or 'com' (center of mass) of each halo
        :type CENTER: str
        :return: ``morph_info`` containing d, q, s, eigframe info
        :rtype: (12,) float array"""
        
        return
    
    def getObjMorphLocal(self, float[:,:] morph_info, float r200, float[:] log_d, float[:,:] xyz, float[:,:] xyz_princ, float[:] masses, int[:] shell, float[:] center, complex[::1,:] shape_tensor, double[::1] eigval, complex[::1,:] eigvec, float M_TOL, int N_WALL, int N_MIN, str CENTER):
        """ Calculates the local axis ratios
        
        The local morphology is calculated for the ellipsoidal radius range [ ``r200`` x ``log_d`` [0], ``r200`` x ``log_d`` [-1]] 
        from the center of the point cloud
        
        :param morph_info: Array to be filled with morphological info. For each column, 1st entry: d,
            2nd entry: q, 3rd entry: s, 4th to 6th: normalized major axis, 7th to 9th: normalized intermediate axis,
            10th to 12th: normalized minor axis
        :type morph_info: (12,N) floats
        :param r200: R_200 (mean not critical) radius of the parent halo
        :type r200: (N2,) float array
        :param log_d: logarithmically equally spaced ellipsoidal radius array of interest, in units of R_200 
            radius of the parent halo, e.g. np.logspace(-2,1,100)
        :type log_d: (N3,) floats
        :param xyz: positions of particles in point cloud
        :type xyz: (N1 x 3) floats
        :param xyz_princ: position arrays transformed into principal frame (varies from iteration to iteration)
        :type xyz_princ: (N1 x 3) floats, zeros
        :param masses: masses of the particles expressed in unit mass
        :type masses: (N1 x 1) floats
        :param shell: indices of points that fall into shell (varies from iteration to iteration)
        :type shell: (N,) ints, zeros
        :param center: center of point cloud
        :type center: (3,) floats
        :param shape_tensor: shape tensor array to be filled
        :type shape_tensor: (3,3) complex, zeros
        :param eigval: eigenvalue array to be filled
        :type eigval: (3,) double, zeros
        :param eigvec: eigenvector array to be filled
        :type eigvec: (3,3) double, zeros
        :param M_TOL: convergence tolerance, eigenvalue fractions must differ by less than ``M_TOL``
            for iteration to stop
        :type M_TOL: float
        :param N_WALL: maximum permissible number of iterations
        :type N_WALL: float
        :param N_MIN: minimum number of particles (DM or star particle) in any iteration; 
            if undercut, shape is unclassified
        :type N_MIN: int
        :param CENTER: shape quantities will be calculated with respect to CENTER = 'mode' (point of highest density)
            or 'com' (center of mass) of each halo
        :type CENTER: str
        :return: ``morph_info`` containing d, q, s, eigframe info in each column, for each ellipsoidal radius
        :rtype: (12,N) float array"""
        
        return
    
    def getObjMorphGlobal(self, float[:] morph_info, float r200, float[:,:] xyz, float[:,:] xyz_princ, float[:] masses, int[:] ellipsoid, float[:] center, complex[::1,:] shape_tensor, double[::1] eigval, complex[::1,:] eigvec, float M_TOL, int N_WALL, int N_MIN, str CENTER):
        """ Calculates the global axis ratios and eigenframe of the point cloud
        
        :param morph_info: Array to be filled with morphological info. 1st entry: d,
            2nd entry: q, 3rd entry: s, 4th to 6th: normalized major axis, 7th to 9th: normalized intermediate axis,
            10th to 12th: normalized minor axis
        :type morph_info: (12,) floats
        :param r200: R_200 (mean not critical) radius of the parent halo
        :type r200: (N2,) float array
        :param xyz: positions of particles in point cloud
        :type xyz: (N1 x 3) floats
        :param xyz_princ: position arrays transformed into principal frame (varies from iteration to iteration)
        :type xyz_princ: (N1 x 3) floats, zeros
        :param masses: masses of the particles expressed in unit mass
        :type masses: (N1 x 1) floats
        :param ellipsoid: indices of points that fall into ellipsoid (varies from iteration to iteration)
        :type ellipsoid: (N,) ints, zeros
        :param center: center of point cloud
        :type center: (3,) floats
        :param shape_tensor: shape tensor array to be filled
        :type shape_tensor: (3,3) complex, zeros
        :param eigval: eigenvalue array to be filled
        :type eigval: (3,) double, zeros
        :param eigvec: eigenvector array to be filled
        :type eigvec: (3,3) double, zeros
        :param M_TOL: convergence tolerance, eigenvalue fractions must differ by less than ``M_TOL``
            for iteration to stop
        :type M_TOL: float
        :param N_WALL: maximum permissible number of iterations
        :type N_WALL: float
        :param N_MIN: minimum number of particles (DM or star particle) in any iteration; 
            if undercut, shape is unclassified
        :type N_MIN: int
        :param CENTER: shape quantities will be calculated with respect to CENTER = 'mode' (point of highest density)
            or 'com' (center of mass) of each halo
        :type CENTER: str
        :return: ``morph_info`` containing d, q, s, eigframe info
        :rtype: (12,) float array"""
        
        return
    
    def getObjMorphLocalVelDisp(self, float[:,:] morph_info, float r200, float[:] log_d, float[:,:] xyz, float[:,:] vxyz, float[:,:] xyz_princ, float[:] masses, int[:] shell, float[:] center, float[:] vcenter, complex[::1,:] shape_tensor, double[::1] eigval, complex[::1,:] eigvec, float M_TOL, int N_WALL, int N_MIN, str CENTER):
        """ Calculates the local axis ratios of the velocity dispersion tensor 
        
        The local morphology is calculated for the ellipsoidal radius range [ ``r200`` x ``log_d`` [0], ``r200`` x ``log_d`` [-1]] 
        from the center of the point cloud
        
        :param morph_info: Array to be filled with morphological info. For each column, 1st entry: d,
            2nd entry: q, 3rd entry: s, 4th to 6th: normalized major axis, 7th to 9th: normalized intermediate axis,
            10th to 12th: normalized minor axis
        :type morph_info: (12,N) floats
        :param r200: R_200 (mean not critical) radius of the parent halo
        :type r200: (N2,) float array
        :param log_d: logarithmically equally spaced ellipsoidal radius array of interest, in units of R_200 
            radius of the parent halo, e.g. np.logspace(-2,1,100)
        :type log_d: (N3,) floats
        :param xyz: positions of particles in point cloud
        :type xyz: (N1 x 3) floats
        :param vxyz: velocity array
        :type vxyz: (N x 3) floats
        :param xyz_princ: position arrays transformed into principal frame (varies from iteration to iteration)
        :type xyz_princ: (N1 x 3) floats, zeros
        :param masses: masses of the particles expressed in unit mass
        :type masses: (N1 x 1) floats
        :param shell: indices of points that fall into shell (varies from iteration to iteration)
        :type shell: (N,) ints, zeros
        :param center: center of point cloud
        :type center: (3,) floats
        :param vcenter: velocity-center of point cloud
        :type vcenter: (3,) floats
        :param shape_tensor: shape tensor array to be filled
        :type shape_tensor: (3,3) complex, zeros
        :param eigval: eigenvalue array to be filled
        :type eigval: (3,) double, zeros
        :param eigvec: eigenvector array to be filled
        :type eigvec: (3,3) double, zeros
        :param M_TOL: convergence tolerance, eigenvalue fractions must differ by less than ``M_TOL``
            for iteration to stop
        :type M_TOL: float
        :param N_WALL: maximum permissible number of iterations
        :type N_WALL: float
        :param N_MIN: minimum number of particles (DM or star particle) in any iteration; 
            if undercut, shape is unclassified
        :type N_MIN: int
        :param CENTER: shape quantities will be calculated with respect to CENTER = 'mode' (point of highest density)
            or 'com' (center of mass) of each halo
        :type CENTER: str
        :return: ``morph_info`` containing d (= ``r200``), q, s, eigframe info
        :rtype: (12,) float array"""
        
        return
    
    def getObjMorphGlobalVelDisp(self, float[:] morph_info, float r200, float[:,:] xyz, float[:,:] vxyz, float[:,:] xyz_princ, float[:] masses, int[:] ellipsoid, float[:] center, float[:] vcenter, complex[::1,:] shape_tensor, double[::1] eigval, complex[::1,:] eigvec, float M_TOL, int N_WALL, int N_MIN, str CENTER):
        """ Calculates the global axis ratios and eigenframe of the velocity dispersion tensor
        
        :param morph_info: Array to be filled with morphological info. 1st entry: d,
            2nd entry: q, 3rd entry: s, 4th to 6th: normalized major axis, 7th to 9th: normalized intermediate axis,
            10th to 12th: normalized minor axis
        :type morph_info: (12,) floats
        :param r200: R_200 (mean not critical) radius of the parent halo
        :type r200: (N2,) float array
        :param xyz: positions of particles in point cloud
        :type xyz: (N1 x 3) floats
        :param vxyz: velocity array
        :type vxyz: (N x 3) floats
        :param xyz_princ: position arrays transformed into principal frame (varies from iteration to iteration)
        :type xyz_princ: (N1 x 3) floats, zeros
        :param masses: masses of the particles expressed in unit mass
        :type masses: (N1 x 1) floats
        :param ellipsoid: indices of points that fall into ellipsoid (varies from iteration to iteration)
        :type ellipsoid: (N,) ints, zeros
        :param center: center of point cloud
        :type center: (3,) floats
        :param vcenter: velocity-center of point cloud
        :type vcenter: (3,) floats
        :param shape_tensor: shape tensor array to be filled
        :type shape_tensor: (3,3) complex, zeros
        :param eigval: eigenvalue array to be filled
        :type eigval: (3,) double, zeros
        :param eigvec: eigenvector array to be filled
        :type eigvec: (3,3) double, zeros
        :param d: distance from the center, kept fixed during iterative procedure
        :type d: float
        :param delta_d: thickness of the shell in real space (constant across shells in logarithmic space)
        :type delta_d: float
        :param M_TOL: convergence tolerance, eigenvalue fractions must differ by less than ``M_TOL``
            for iteration to stop
        :type M_TOL: float
        :param N_WALL: maximum permissible number of iterations
        :type N_WALL: float
        :param N_MIN: minimum number of particles (DM or star particle) in any iteration; 
            if undercut, shape is unclassified
        :type N_MIN: int
        :param CENTER: shape quantities will be calculated with respect to CENTER = 'mode' (point of highest density)
            or 'com' (center of mass) of each halo
        :type CENTER: str
        :return: ``morph_info`` containing d (= ``r200``), q, s, eigframe info
        :rtype: (12,) float array"""
        
        return
    
    def getMorphLocal(self, float[:,:] xyz, cat, float[:] masses, float[:] r200, float L_BOX, int MIN_NUMBER_PTCS, int D_LOGSTART, int D_LOGEND, int D_BINS, int M_TOL, int N_WALL, int N_MIN, str CENTER):
        """ Calculates the local shape catalogue
        
        Calls ``getObjMorphLocal()`` in a parallelized manner.\n
        Calculates the axis ratios for the range [ ``r200`` x 10**(``D_LOGSTART``), ``r200`` x 10**(``D_LOGEND``)] from the centers, for each object.
        
        :param xyz: positions of all (DM or star) particles in simulation box
        :type xyz: (N1 x 3) floats
        :param cat: each entry of the list is a list containing indices of particles belonging to an object
        :type cat: list of length N2
        :param masses: masses of the particles expressed in unit mass
        :type masses: (N1 x 1) floats
        :param r200: each entry of the list gives the R_200 (mean not critical) radius of the parent halo
        :type r200: list of length N2
        :param L_BOX: simulation box side length
        :type L_BOX: float, units: cMpc/h
        :param MIN_NUMBER_PTCS: minimum number of particles for object to qualify for morphology calculation
        :type MIN_NUMBER_PTCS: int
        :param D_LOGSTART: logarithm of minimum ellipsoidal radius of interest, in units of R200 of parent halo
        :type D_LOGSTART: int
        :param D_LOGEND: logarithm of maximum ellipsoidal radius of interest, in units of R200 of parent halo
        :type D_LOGEND: int
        :param D_BINS: number of ellipsoidal radii of interest minus 1 (i.e. number of bins)
        :type D_BINS: int
        :param M_TOL: convergence tolerance, eigenvalue fractions must differ by less than ``M_TOL``
            for iteration to stop
        :type M_TOL: float
        :param N_WALL: maximum permissible number of iterations
        :type N_WALL: float
        :param N_MIN: minimum number of particles (DM or star particle) in any iteration; 
            if undercut, shape is unclassified
        :type N_MIN: int
        :param CENTER: shape quantities will be calculated with respect to CENTER = 'mode' (point of highest density)
            or 'com' (center of mass) of each halo
        :type CENTER: str
        :return: d, q, s, eigframe, centers, masses, l_succeed: list of object indices for which morphology could be determined at R200 (length: N3)
        :rtype: (N3, ``D_BINS`` + 1) floats (for d, q, s, eigframe (x3)), (N3, 3) floats (for centers), (N3,) floats (for masses), N3-list of ints for l_succeed
        """
                        
        return
    
    def getMorphGlobal(self, float[:,:] xyz, cat, float[:] masses, float[:] r200, float L_BOX, int MIN_NUMBER_PTCS, int M_TOL, int N_WALL, int N_MIN, str CENTER):
        """ Calculates the overall shape catalogue
        
        Calls ``getObjMorphGlobal()`` in a parallelized manner.\n
        Calculates the overall axis ratios and eigenframe for each object.
        
        :param xyz: positions of all (DM or star) particles in simulation box
        :type xyz: (N1 x 3) floats
        :param cat: each entry of the list is a list containing indices of particles belonging to an object
        :type cat: list of length N2
        :param masses: masses of the particles expressed in unit mass
        :type masses: (N1 x 1) floats
        :param r200: each entry of the list gives the R_200 (mean not critical) radius of the parent halo
        :type r200: list of length N2
        :param L_BOX: simulation box side length
        :type L_BOX: float, units: cMpc/h
        :param MIN_NUMBER_PTCS: minimum number of particles for object to qualify for morphology calculation
        :type MIN_NUMBER_PTCS: int
        :param M_TOL: convergence tolerance, eigenvalue fractions must differ by less than ``M_TOL``
            for iteration to stop
        :type M_TOL: float
        :param N_WALL: maximum permissible number of iterations
        :type N_WALL: float
        :param N_MIN: minimum number of particles (DM or star particle) in any iteration; 
            if undercut, shape is unclassified
        :type N_MIN: int
        :param CENTER: shape quantities will be calculated with respect to CENTER = 'mode' (point of highest density)
            or 'com' (center of mass) of each halo
        :type CENTER: str
        :return: d, q, s, eigframe, centers, masses
        :rtype: (N3,) floats (for d, q, s, eigframe (x3)), (N3, 3) floats (for centers), (N3,) floats (for masses)
        """
        
        return
    
    def getMorphLocalVelDisp(self, float[:,:] xyz, float[:,:] vxyz, cat, float[:] masses, float[:] r200, float L_BOX, int MIN_NUMBER_PTCS, int D_LOGSTART, int D_LOGEND, int D_BINS, int M_TOL, int N_WALL, int N_MIN, str CENTER):
        """ Calculates the local velocity dispersion shape catalogue
        
        Calls ``getObjMorphLocalVelDisp()`` in a parallelized manner.\n
        Calculates the overall axis ratios and eigenframe for each object.
        
        :param xyz: positions of all (DM or star) particles in simulation box
        :type xyz: (N1 x 3) floats
        :param vxyz: velocities of all (DM or star) particles in simulation box
        :type vxyz: (N1 x 3) floats
        :param cat: each entry of the list is a list containing indices of particles belonging to an object
        :type cat: list of length N2
        :param masses: masses of the particles expressed in unit mass
        :type masses: (N1 x 1) floats
        :param r200: each entry of the list gives the R_200 (mean not critical) radius of the parent halo
        :type r200: list of length N2
        :param L_BOX: simulation box side length
        :type L_BOX: float, units: cMpc/h
        :param MIN_NUMBER_PTCS: minimum number of particles for object to qualify for morphology calculation
        :type MIN_NUMBER_PTCS: int
        :param D_LOGSTART: logarithm of minimum ellipsoidal radius of interest, in units of R200 of parent halo
        :type D_LOGSTART: int
        :param D_LOGEND: logarithm of maximum ellipsoidal radius of interest, in units of R200 of parent halo
        :type D_LOGEND: int
        :param D_BINS: number of ellipsoidal radii of interest mi
        :param M_TOL: convergence tolerance, eigenvalue fractions must differ by less than ``M_TOL``
            for iteration to stop
        :type M_TOL: float
        :param N_WALL: maximum permissible number of iterations
        :type N_WALL: float
        :param N_MIN: minimum number of particles (DM or star particle) in any iteration; 
            if undercut, shape is unclassified
        :type N_MIN: int
        :param CENTER: shape quantities will be calculated with respect to CENTER = 'mode' (point of highest density)
            or 'com' (center of mass) of each halo
        :type CENTER: str
        :return: d, q, s, eigframe, centers, masses, l_succeed: list of object indices for which morphology could be determined at R200 (length: N3)
        :rtype: (N3, ``D_BINS`` + 1) floats (for d, q, s, eigframe (x3)), (N3, 3) floats (for centers), (N3,) floats (for masses), N3-list of ints for l_succeed
        """
        
        return
    
    def getMorphGlobalVelDisp(self, float[:,:] xyz, float[:,:] vxyz, cat, float[:] masses, float[:] r200, float L_BOX, int MIN_NUMBER_PTCS, int M_TOL, int N_WALL, int N_MIN, str CENTER):
        """ Calculates the global velocity dipsersion shape catalogue
        
        Calls ``getObjMorphGlobalVelDisp()`` in a parallelized manner.\n
        Calculates the overall axis ratios and eigenframe for each object.
        
        :param xyz: positions of all (DM or star) particles in simulation box
        :type xyz: (N1 x 3) floats
        :param vxyz: velocities of all (DM or star) particles in simulation box
        :type vxyz: (N1 x 3) floats
        :param cat: each entry of the list is a list containing indices of particles belonging to an object
        :type cat: list of length N2
        :param masses: masses of the particles expressed in unit mass
        :type masses: (N1 x 1) floats
        :param r200: each entry of the list gives the R_200 (mean not critical) radius of the parent halo
        :type r200: list of length N2
        :param L_BOX: simulation box side length
        :type L_BOX: float, units: cMpc/h
        :param MIN_NUMBER_PTCS: minimum number of particles for object to qualify for morphology calculation
        :type MIN_NUMBER_PTCS: int
        :param M_TOL: convergence tolerance, eigenvalue fractions must differ by less than ``M_TOL``
            for iteration to stop
        :type M_TOL: float
        :param N_WALL: maximum permissible number of iterations
        :type N_WALL: float
        :param N_MIN: minimum number of particles (DM or star particle) in any iteration; 
            if undercut, shape is unclassified
        :type N_MIN: int
        :param CENTER: shape quantities will be calculated with respect to CENTER = 'mode' (point of highest density)
            or 'com' (center of mass) of each halo
        :type CENTER: str
        :return: d, q, s, eigframe, centers, masses
        :rtype: (N3, ``D_BINS`` + 1) floats (for d, q, s, eigframe (x3)), (N3, 3) floats (for centers), (N3,) floats (for masses)
        """
        
        return
    
    def drawShapeCurves(self, obj_type = ''):
        """ Draws some simplistic shape curves
        
        :param obj_type: either 'dm' or 'gx' for CosmicShapesGadgetHDF5 or '' for CosmicShapesDirect
        :type obj_type: string"""
        return
        
    def plotLocalTHisto(self, obj_type = ''):
        """ Plot the triaxiality-histogram
        
        :param obj_type: either 'dm' or 'gx' for CosmicShapesGadgetHDF5 or '' for CosmicShapesDirect
        :type obj_type: string"""
        return
    
cdef class CosmicShapesDirect(CosmicShapes):
    """ Subclass to calculate morphology for already identified objects
    
    The particle indices of the objects identified are stored in ``cat``.\n
    
    The public methods are ``calcGlobalShapes()``, ``calcLocalShapes()``,
    ``plotGlobalEpsHisto()``, ``vizGlobalShapes()``, ``vizLocalShapes()``."""
    
    cdef float[:,:] xyz
    cdef float[:] masses
    cdef int[:,:] cat_arr
    cdef float[:] r200
    cdef str SNAP
    cdef object cat

    def __init__(self, float[:,:] xyz, float[:] masses, cat, float[:] r200, str CAT_DEST, str VIZ_DEST, str SNAP, float L_BOX, int MIN_NUMBER_PTCS, int D_LOGSTART, int D_LOGEND, int D_BINS, float M_TOL, int N_WALL, int N_MIN, str CENTER, double start_time):
        """      
        :param xyz: positions of all (DM or star) particles in simulation box
        :type xyz: (N1 x 3) floats
        :param masses: masses of the particles expressed in unit mass
        :type masses: (N1 x 1) floats
        :param cat: each entry of the list is a list containing indices of particles belonging to an object
        :type cat: list of length N2
        :param r200: each entry of the list gives the R_200 (mean not critical) radius of the parent halo
        :type r200: list of length N2
        :param CAT_DEST: catalogue destination
        :type CAT_DEST: string
        :param VIZ_DEST: visualisation folder destination
        :type VIZ_DEST: string
        :param SNAP: e.g. '024'
        :type SNAP: string
        :param L_BOX: simulation box side length
        :type L_BOX: float, units: cMpc/h
        :param MIN_NUMBER_PTCS: minimum number of particles for object to qualify for morphology calculation
        :type MIN_NUMBER_PTCS: int
        :param D_LOGSTART: logarithm of minimum ellipsoidal radius of interest, in units of R200 of parent halo
        :type D_LOGSTART: int
        :param D_LOGEND: logarithm of maximum ellipsoidal radius of interest, in units of R200 of parent halo
        :type D_LOGEND: int
        :param D_BINS: number of ellipsoidal radii of interest minus 1 (i.e. number of bins)
        :type D_BINS: int
        :param M_TOL: convergence tolerance, eigenvalue fractions must differ by less than ``M_TOL``
            for iteration to stop
        :type M_TOL: float
        :param N_WALL: maximum permissible number of iterations
        :type N_WALL: float
        :param N_MIN: minimum number of particles (DM or star particle) in any iteration; 
            if undercut, shape is unclassified
        :type N_MIN: int
        :param CENTER: shape quantities will be calculated with respect to CENTER = 'mode' (point of highest density)
            or 'com' (center of mass) of each halo
        :type CENTER: str
        :param start_time: time of start of object initialization
        :type start_time: float"""
        super().__init__(CAT_DEST, VIZ_DEST, L_BOX, MIN_NUMBER_PTCS, D_LOGSTART, D_LOGEND, D_BINS, M_TOL, N_WALL, N_MIN, CENTER, start_time)
        assert xyz.shape[0] == masses.shape[0], "xyz.shape[0] must be equal to masses.shape[0]"
        self.xyz = xyz
        self.masses = masses
        self.cat = cat
        self.r200 = r200
        self.SNAP = SNAP
        
    def calcLocalShapes(self):   
        """ Calculates and saves local object shape catalogues"""       
        return
    
    def calcGlobalShapes(self):   
        """ Calculates and saves global object shape catalogues"""       
        return
    
    def vizLocalShapes(self, obj_numbers):
        """ Visualize local shape of objects with numbers ``obj_numbers``
        
        :param obj_numbers: list of object indices for which to visualize local shapes
        :type obj_numbers: list of ints"""
        return
    
    def vizGlobalShapes(self, obj_numbers):
        """ Visualize global shape of objects with numbers ``obj_numbers``
        
        :param obj_numbers: list of object indices for which to visualize global shapes
        :type obj_numbers: list of ints"""
        return

    def plotGlobalEpsHisto(self):
        """ Plot ellipticity histogram"""
        return
    
    def calcDensProfs(self, ROverR200):
        """ Calculate density profiles
        
        :param ROverR200: At which unitless radial values to calculate density profiles
        :type ROverR200: float array"""
        return
        
cdef class CosmicShapesGadgetHDF5(CosmicShapes):
    """ Subclass to calculate morphology for yet to-be-identified objects in Gadget HDF5 simulation
    
    The public methods are ``calcGlobalShapesDM()``, ``calcLocalShapesDM()``,
    ``calcGlobalShapesGx()``, ``calcLocalShapesGx()``, ``calcGlobalVelShapesDM()``, 
    ``calcLocalVelShapesDM()``, ``loadDMCat()``, ``plotGlobalEpsHisto()``, 
    ``vizGlobalShapes()``, ``vizLocalShapes()``."""
    
    cdef str HDF5_SNAP_DEST
    cdef str HDF5_GROUP_DEST
    cdef int MIN_NUMBER_STAR_PTCS
    cdef str SNAP
    cdef int SNAP_MAX
    cdef float[:] r200
    
    def __init__(self, str HDF5_SNAP_DEST, str HDF5_GROUP_DEST, str CAT_DEST, str VIZ_DEST, str SNAP, int SNAP_MAX, float L_BOX, int MIN_NUMBER_PTCS, int MIN_NUMBER_STAR_PTCS, int D_LOGSTART, int D_LOGEND, int D_BINS, float M_TOL, int N_WALL, int N_MIN, str CENTER, double start_time):
        """ 
        :param HDF5_SNAP_DEST: where we can find the snapshot
        :type HDF5_SNAP_DEST: string
        :param HDF5_GROUP_DEST: where we can find the group files
        :type HDF5_GROUP_DEST: string
        :param CAT_DEST: catalogue destination
        :type CAT_DEST: string
        :param VIZ_DEST: visualisation folder destination
        :type VIZ_DEST: string
        :param SNAP: e.g. '024'
        :type SNAP: string
        :param SNAP_MAX: e.g. '024'
        :type SNAP_MAX: string
        :param L_BOX: simulation box side length
        :type L_BOX: float, units: cMpc/h
        :param MIN_NUMBER_PTCS: minimum number of particles for halo to qualify for morphology calculation
        :type MIN_NUMBER_PTCS: int
        :param MIN_NUMBER_STAR_PTCS: minimum number of particles for galaxy (to-be-identified) to qualify for morphology calculation
        :type MIN_NUMBER_STAR_PTCS: int
        :param D_LOGSTART: logarithm of minimum ellipsoidal radius of interest, in units of R200 of parent halo
        :type D_LOGSTART: int
        :param D_LOGEND: logarithm of maximum ellipsoidal radius of interest, in units of R200 of parent halo
        :type D_LOGEND: int
        :param D_BINS: number of ellipsoidal radii of interest minus 1 (i.e. number of bins)
        :type D_BINS: int
        :param M_TOL: convergence tolerance, eigenvalue fractions must differ by less than ``M_TOL``
            for iteration to stop
        :type M_TOL: float
        :param N_WALL: maximum permissible number of iterations
        :type N_WALL: float
        :param N_MIN: minimum number of particles (DM or star particle) in any iteration; 
            if undercut, shape is unclassified
        :type N_MIN: int
        :param CENTER: shape quantities will be calculated with respect to CENTER = 'mode' (point of highest density)
            or 'com' (center of mass) of each halo
        :type CENTER: str
        :param start_time: time of start of object initialization
        :type start_time: float"""
        super().__init__(CAT_DEST, VIZ_DEST, L_BOX, MIN_NUMBER_PTCS, D_LOGSTART, D_LOGEND, D_BINS, M_TOL, N_WALL, N_MIN, CENTER, start_time)
        self.HDF5_SNAP_DEST = HDF5_SNAP_DEST
        self.HDF5_GROUP_DEST = HDF5_GROUP_DEST
        self.MIN_NUMBER_STAR_PTCS = MIN_NUMBER_STAR_PTCS
        self.SNAP = SNAP
        self.SNAP_MAX = SNAP_MAX
                
    def vizLocalShapes(self, obj_numbers, obj_type = 'dm'):
        """ Visualize local shape of objects with numbers ``obj_numbers``"""
        return
        
    def vizGlobalShapes(self, obj_numbers, obj_type = 'dm'):
        """ Visualize global shape of objects with numbers ``obj_numbers``"""
        return
    
    def loadDMCat(self):
        """ Loads halo (more precisely: CSH) catalogues from FOF data
        
        Stores R200 as self.r200"""
        return
    
    def loadGxCat(self):
        """ Loads galaxy catalogues from HDF5 data
        
        To discard wind particles, add the line
        `gx_cat_l = [[x for x in gx if is_star[x]] for gx in gx_cat_l]` before
        saving the catalogue."""
        return
    
    def calcLocalShapesGx(self):
        """ Calculates and saves local galaxy shape catalogues"""      
        return
    
    def calcGlobalShapesGx(self):   
        """ Calculates and saves global galaxy shape catalogues"""       
        return
    
    def calcLocalShapesDM(self):
        """ Calculates and saves local halo shape catalogues"""      
        return
    
    def calcGlobalShapesDM(self):   
        """ Calculates and saves global halo shape catalogues"""       
        return
    
    def calcLocalVelShapesDM(self):
        """ Calculates and saves local velocity dispersion tensor shape catalogues"""      
        return
    
    def calcGlobalVelShapesDM(self):
        """ Calculates and saves global velocity dispersion tensor shape catalogues"""      
        return
    
    def calcDensProfs(self, ROverR200, obj_type = ''):
        """ Calculate density profiles
        
        :param ROverR200: At which unitless radial values to calculate density profiles
        :type ROverR200: float array
        :param obj_type: either 'dm' or 'gx', depending on what catalogue 
            the ellipticity histogram should be plotted for
        :type obj_type: string
        :return: density profiles
        :rtype: (N,ROverR200.shape) float array"""
        return
    
    def plotGlobalEpsHisto(self, obj_type = ''):
        """ Plot ellipticity histogram
        
        :param obj_type: either 'dm' or 'gx', depending on what catalogue 
            the ellipticity histogram should be plotted for
        :type obj_type: string"""
        return