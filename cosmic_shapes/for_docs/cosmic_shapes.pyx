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
from get_hdf5 import getHDF5Data, getHDF5GxData, getHDF5SHData, getHDF5DMData
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
def createLogNormUni(BoxSize, nbar, redshift, Nmesh, UNIT_MASS, h):
    """ Create mock simulation box by Poisson-sampling a lognormal density distribution
    
    The Poisson-sampled distribution is evolved according to the Zeldovich (1LPT) prescription
    up until redshift `redshift` under the constraint of an 'EisensteinHu' power spectrum
    
    :param BoxSize: size of to-be-obtained simulation box
    :type BoxSize: float
    :param nbar: number density of points (i.e. sampling density / resolution) in box, units: 1/(cMpc/h)**3
        Note: `nbar` is assumed to be constant across the box
    :type nbar: float
    :param redshift: redshift of interest
    :type redshift: float
    :param Nmesh: the mesh size to use when generating the density and displacement fields, 
        which are Poisson-sampled to particles
    :type Nmesh: int
    :param UNIT_MASS: in units of solar masses / h. Returned masses will have units UNIT_MASS*(solar_mass)/h
    :type UNIT_MASS: float
    :param h: little H
    :type h: float
    :return: total number of particles, xyz-coordinates of DM particles, xyz-values of DM particle velocities, 
        masses of the DM particles (all identical)
    :rtype: int, (N,) floats, (N,) floats, (N,) floats, (N,) floats, (N,) floats, (N,) floats, (N,) floats"""
    return

@cython.embedsignature(True)
def createHaloFixedAxisRatioRhoAlphaBetaGamma(N_min, alpha, beta, gamma, rho_0, r_s, a, b, c):
    """ Mock halo generator
    
    Create mock halo consisting of `N_min` particles in 1st shell. The alpha-beta-gamma 
    density profile is a generalization of the Navarro-Frank-White (NFW) profile. Its definition
    can be looked up in Zemp et al 2011, https://arxiv.org/abs/1107.5582.
    
    :param N_min: number of particles in 1st shell. Will be scaled appropriately 
        in each shell to satisfy alpha-beta-gamma profile
    :type N_min: int
    :param alpha: `alpha` parameter in alpha-beta-gamma density profile
    :type alpha: float
    :param beta: `beta` parameter in alpha-beta-gamma density profile 
    :type beta: float
    :param gamma: `gamma` parameter in alpha-beta-gamma density profile 
    :type gamma: float
    :param rho_0: `rho_0` parameter in alpha-beta-gamma density profile (density at the center)
    :type rho_0: float
    :param r_s: `r_s` parameter in alpha-beta-gamma density profile (scale radius)
    :type r_s: float
    :param a: major axis array
    :type a: float array
    :param b: intermediate axis array
    :type b: float array
    :param c: minor axis array
    :type c: float array
    :return: halo_x, halo_y, halo_z: arrays containing positions of halo particles
    :rtype: 3 (N,) float arrays
    """
    return

@cython.embedsignature(True)
cdef class CosmicShapes:
    cdef str CAT_DEST
    cdef str VIZ_DEST
    cdef float L_BOX
    cdef int MIN_NUMBER_DM_PTCS
    cdef int D_LOGSTART
    cdef int D_LOGEND
    cdef int D_BINS
    cdef float M_TOL
    cdef int N_WALL
    cdef int N_MIN
    cdef float SAFE
    cdef float start_time
    
    def __init__(self, str CAT_DEST, str VIZ_DEST, float L_BOX, int MIN_NUMBER_DM_PTCS, int D_LOGSTART, int D_LOGEND, int D_BINS, float M_TOL, int N_WALL, int N_MIN, float SAFE, float start_time):
        """ Constructor method. Setting instance variables"""
        self.CAT_DEST = CAT_DEST
        self.VIZ_DEST = VIZ_DEST
        self.L_BOX = L_BOX
        self.MIN_NUMBER_DM_PTCS = MIN_NUMBER_DM_PTCS
        self.D_LOGSTART = D_LOGSTART
        self.D_LOGEND = D_LOGEND
        self.D_BINS = D_BINS
        self.M_TOL = M_TOL
        self.N_WALL = N_WALL
        self.N_MIN = N_MIN
        self.SAFE = SAFE
        self.start_time = start_time
    
    def S1_obj(self, float[:] morph_info, float[:,:] xyz, float[:,:] xyz_princ, float[:] masses, int[:] shell, float[:] com, complex[::1,:] shape_tensor, double[::1] eigval, complex[::1,:] eigvec, float d, float delta_d, float M_TOL, int N_WALL, int N_MIN):
        """ S1 algorithm for halos/galaxies at elliptical radius `d` with shell width `delta_d`
        
        Calculates the axis ratios at a distance `d` from the COM of the entire particle distro.\n
        Note that before and during the iteration, `d` is defined with respect to the COM of 
        the entire particle distro, not the COM of the initial spherical volume as in Katz 1991.\n
        Differential version of E1.\n
        Shells can cross (except 2nd shell with 1st), and a shell is assumed to be equally thick everywhere.\n
        Whether we adopt the last assumption or let the thickness float (Tomassetti et al 2016) barely makes 
        any difference in terms of shapes found, but the convergence properties improve for the version with fixated thickness.
        For 1st shell: `delta_d` is `d`
        
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
        :param com: COM of point cloud
        :type com: (3,) floats
        :param shape_tensor: shape tensor array to be filled
        :type shape_tensor: (3,3) complex, zeros
        :param eigval: eigenvalue array to be filled
        :type eigval: (3,) double, zeros
        :param eigvec: eigenvector array to be filled
        :type eigvec: (3,3) double, zeros
        :param d: Distance from the COM, kept fixed during iterative procedure
        :type d: float
        :param delta_d: thickness of the shell in real space (constant across shells in logarithmic space)
        :type delta_d: float
        :param M_TOL: convergence tolerance, eigenvalue fractions must differ by less than `M_TOL`
            for iteration to stop
        :type M_TOL: float
        :param N_WALL: maximum permissible number of iterations
        :type N_WALL: float
        :param N_MIN: minimum number of particles (DM or star particle) in any iteration; 
            if undercut, shape is unclassified
        :type N_MIN: int
        :return: `morph_info` containing d, q, s, eigframe info
        :rtype: (12,) float array"""
        
        return
    
    
    def E1_obj(self, float[:] morph_info, float[:,:] xyz, float[:,:] xyz_princ, float[:] masses, int[:] ellipsoid, float[:] com, complex[::1,:] shape_tensor, double[::1] eigval, complex[::1,:] eigvec, float d, float M_TOL, int N_WALL, int N_MIN):
        """ E1 algorithm for halos/galaxies at elliptical radius `d`
        
        Calculates the axis ratios at a distance `d` from the COM of the entire particle distro.\n
        Note that before and during the iteration, `d` is defined with respect to the COM of 
        the entire particle distro, not the COM of the initial spherical volume as in Katz 1991.\n
        
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
        :param com: COM of point cloud
        :type com: (3,) floats
        :param shape_tensor: shape tensor array to be filled
        :type shape_tensor: (3,3) complex, zeros
        :param eigval: eigenvalue array to be filled
        :type eigval: (3,) double, zeros
        :param eigvec: eigenvector array to be filled
        :type eigvec: (3,3) double, zeros
        :param d: Distance from the COM, kept fixed during iterative procedure
        :type d: float
        :param M_TOL: convergence tolerance, eigenvalue fractions must differ by less than `M_TOL`
            for iteration to stop
        :type M_TOL: float
        :param N_WALL: maximum permissible number of iterations
        :type N_WALL: float
        :param N_MIN: minimum number of particles (DM or star particle) in any iteration; 
            if undercut, shape is unclassified
        :type N_MIN: int
        :return: `morph_info` containing d, q, s, eigframe info
        :rtype: (12,) float array"""
        
        return
    
    def E1_vdisp(self, float[:] morph_info, float[:,:] xyz, float[:,:] vxyz, float[:,:] xyz_princ, float[:] masses, int[:] ellipsoid, float[:] com, float[:] vcom, complex[::1,:] shape_tensor, double[::1] eigval, complex[::1,:] eigvec, float d, float M_TOL, int N_WALL, int N_MIN):
        """ Similar to `E1_obj` algorithm for halos/galaxies but for velocity dispersion tensor
        
        Calculates the axis ratios at a distance `d` from the COM of the entire particle distro.\n
        Note that before and during the iteration, `d` is defined with respect to the COM of 
        the entire particle distro, not the COM of the initial spherical volume as in Katz 1991.\n
        
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
        :param com: COM of point cloud
        :type com: (3,) floats
        :param vcom: velocity-COM of point cloud
        :type vcom: (3,) floats
        :param shape_tensor: shape tensor array to be filled
        :type shape_tensor: (3,3) complex, zeros
        :param eigval: eigenvalue array to be filled
        :type eigval: (3,) double, zeros
        :param eigvec: eigenvector array to be filled
        :type eigvec: (3,3) double, zeros
        :param d: Distance from the COM, kept fixed during iterative procedure
        :type d: float
        :param M_TOL: convergence tolerance, eigenvalue fractions must differ by less than `M_TOL`
            for iteration to stop
        :type M_TOL: float
        :param N_WALL: maximum permissible number of iterations
        :type N_WALL: float
        :param N_MIN: minimum number of particles (DM or star particle) in any iteration; 
            if undercut, shape is unclassified
        :type N_MIN: int
        :return: `morph_info` containing d, q, s, eigframe info
        :rtype: (12,) float array"""
        
        return
    
    def getMorphLocalObj(self, float[:,:] morph_info, float r200, float[:] log_d, float[:,:] xyz, float[:,:] xyz_princ, float[:] masses, int[:] shell, float[:] com, complex[::1,:] shape_tensor, double[::1] eigval, complex[::1,:] eigvec, float M_TOL, int N_WALL, int N_MIN):
        """ Calculates the local axis ratios for the ellipsoidal radius range
            [ `r200` x `log_d` [0], `r200` x `log_d` [-1]] from the COM of the point cloud
        
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
        :param com: COM of point cloud
        :type com: (3,) floats
        :param shape_tensor: shape tensor array to be filled
        :type shape_tensor: (3,3) complex, zeros
        :param eigval: eigenvalue array to be filled
        :type eigval: (3,) double, zeros
        :param eigvec: eigenvector array to be filled
        :type eigvec: (3,3) double, zeros
        :param d: Distance from the COM, kept fixed during iterative procedure
        :type d: float
        :param delta_d: thickness of the shell in real space (constant across shells in logarithmic space)
        :type delta_d: float
        :param M_TOL: convergence tolerance, eigenvalue fractions must differ by less than `M_TOL`
            for iteration to stop
        :type M_TOL: float
        :param N_WALL: maximum permissible number of iterations
        :type N_WALL: float
        :param N_MIN: minimum number of particles (DM or star particle) in any iteration; 
            if undercut, shape is unclassified
        :type N_MIN: int
        :return: `morph_info` containing d, q, s, eigframe info in each column, for each ellipsoidal radius
        :rtype: (12,N) float array"""
        
        return
    
    def getMorphOvrlObj(self, float[:] morph_info, float r200, float[:,:] xyz, float[:,:] xyz_princ, float[:] masses, int[:] ellipsoid, float[:] com, complex[::1,:] shape_tensor, double[::1] eigval, complex[::1,:] eigvec, float M_TOL, int N_WALL, int N_MIN, float SAFE):
        """ Calculates the overall axis ratios and eigenframe of the point cloud
        
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
        :param com: COM of point cloud
        :type com: (3,) floats
        :param shape_tensor: shape tensor array to be filled
        :type shape_tensor: (3,3) complex, zeros
        :param eigval: eigenvalue array to be filled
        :type eigval: (3,) double, zeros
        :param eigvec: eigenvector array to be filled
        :type eigvec: (3,3) double, zeros
        :param d: Distance from the COM, kept fixed during iterative procedure
        :type d: float
        :param delta_d: thickness of the shell in real space (constant across shells in logarithmic space)
        :type delta_d: float
        :param M_TOL: convergence tolerance, eigenvalue fractions must differ by less than `M_TOL`
            for iteration to stop
        :type M_TOL: float
        :param N_WALL: maximum permissible number of iterations
        :type N_WALL: float
        :param N_MIN: minimum number of particles (DM or star particle) in any iteration; 
            if undercut, shape is unclassified
        :type N_MIN: int
        :param SAFE: ellipsoidal radius will be maxdist(COM,point)+SAFE where point is any point in the point cloud
        :type SAFE: float, units: cMpc/h
        :return: `morph_info` containing d, q, s, eigframe info
        :rtype: (12,) float array"""
        
        return
    
    def getMorphOvrlVDispObj(self, float[:] morph_info, float r200, float[:,:] xyz, float[:,:] vxyz, float[:,:] xyz_princ, float[:] masses, int[:] ellipsoid, float[:] com, float[:] vcom, complex[::1,:] shape_tensor, double[::1] eigval, complex[::1,:] eigvec, float M_TOL, int N_WALL, int N_MIN):
        """ Calculates the overall axis ratios and eigenframe of the point cloud
        
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
        :param com: COM of point cloud
        :type com: (3,) floats
        :param vcom: velocity-COM of point cloud
        :type vcom: (3,) floats
        :param shape_tensor: shape tensor array to be filled
        :type shape_tensor: (3,3) complex, zeros
        :param eigval: eigenvalue array to be filled
        :type eigval: (3,) double, zeros
        :param eigvec: eigenvector array to be filled
        :type eigvec: (3,3) double, zeros
        :param d: Distance from the COM, kept fixed during iterative procedure
        :type d: float
        :param delta_d: thickness of the shell in real space (constant across shells in logarithmic space)
        :type delta_d: float
        :param M_TOL: convergence tolerance, eigenvalue fractions must differ by less than `M_TOL`
            for iteration to stop
        :type M_TOL: float
        :param N_WALL: maximum permissible number of iterations
        :type N_WALL: float
        :param N_MIN: minimum number of particles (DM or star particle) in any iteration; 
            if undercut, shape is unclassified
        :type N_MIN: int
        :return: `morph_info` containing d (= `r200`), q, s, eigframe info
        :rtype: (12,) float array"""
        
        return
    
    def getMorphLocal(self, float[:,:] xyz, cat, float[:] masses, float[:] r200, float L_BOX, int MIN_NUMBER_PTCS, int D_LOGSTART, int D_LOGEND, int D_BINS, int M_TOL, int N_WALL, int N_MIN):
        """ Calculates the local shape catalogue
        
        Calls `getMorphLocalObj()` in an mpi4py-parallelized manner.\n
        Calculates the axis ratios for the range [ `r200` x 10**(`D_LOGSTART`), `r200` x 10**(`D_LOGEND`)] from the COMs, for each object.
        
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
        :param M_TOL: convergence tolerance, eigenvalue fractions must differ by less than `M_TOL`
            for iteration to stop
        :type M_TOL: float
        :param N_WALL: maximum permissible number of iterations
        :type N_WALL: float
        :param N_MIN: minimum number of particles (DM or star particle) in any iteration; 
            if undercut, shape is unclassified
        :type N_MIN: int
        :return: d, q, s, eigframe, COMs, masses, l_succeed: list of objects for which morphology could be determined at R200 (length: N3)
        :rtype: (N3, `D_BINS` + 1) floats (for d, q, s, eigframe (x3)), (N3, 3) floats (for COMs), (N3,) floats (for masses), N3-list of ints for l_succeed
        """
                        
        return
    
    def getMorphOvrl(self, float[:,:] xyz, cat, float[:] masses, float[:] r200, float L_BOX, int MIN_NUMBER_PTCS, int M_TOL, int N_WALL, int N_MIN, float SAFE):
        """ Calculates the overall shape catalogue
        
        Calls `getMorphOvrlObj()` in an mpi4py-parallelized manner.\n
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
        :param M_TOL: convergence tolerance, eigenvalue fractions must differ by less than `M_TOL`
            for iteration to stop
        :type M_TOL: float
        :param N_WALL: maximum permissible number of iterations
        :type N_WALL: float
        :param N_MIN: minimum number of particles (DM or star particle) in any iteration; 
            if undercut, shape is unclassified
        :type N_MIN: int
        :param SAFE: ellipsoidal radius will be maxdist(COM,point)+SAFE where point is any point in the point cloud
        :type SAFE: float, units: cMpc/h
        :return: d, q, s, eigframe, COMs, masses
        :rtype: (N3,) floats (for d, q, s, eigframe (x3)), (N3, 3) floats (for COMs), (N3,) floats (for masses)
        """
        
        return
    
    def getMorphOvrlVDisp(self, float[:,:] xyz, float[:,:] vxyz, cat, float[:] masses, float[:] r200, float L_BOX, int MIN_NUMBER_PTCS, int M_TOL, int N_WALL, int N_MIN):
        """ Calculates the overall velocity dipsersion shape catalogue
        
        Calls `getMorphOvrlVDispObj()` in an mpi4py-parallelized manner.\n
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
        :param M_TOL: convergence tolerance, eigenvalue fractions must differ by less than `M_TOL`
            for iteration to stop
        :type M_TOL: float
        :param N_WALL: maximum permissible number of iterations
        :type N_WALL: float
        :param N_MIN: minimum number of particles (DM or star particle) in any iteration; 
            if undercut, shape is unclassified
        :type N_MIN: int
        :return: q, s, major
        :rtype: (N3,) floats (for q, s, major (x3))
        """
        
        return
    
cdef class CosmicShapesDirect(CosmicShapes):
    
    cdef float[:,:] dm_xyz
    cdef float[:] dm_masses
    cdef int[:,:] cat_arr
    cdef float[:] r200
    cdef str SNAP
    cdef object cat

    def __init__(self, float[:,:] dm_xyz, float[:] dm_masses, cat, float[:] r200, str CAT_DEST, str VIZ_DEST, str SNAP, float L_BOX, int MIN_NUMBER_DM_PTCS, int D_LOGSTART, int D_LOGEND, int D_BINS, float M_TOL, int N_WALL, int N_MIN, float SAFE, float start_time):
        super().__init__(CAT_DEST, VIZ_DEST, L_BOX, MIN_NUMBER_DM_PTCS, D_LOGSTART, D_LOGEND, D_BINS, M_TOL, N_WALL, N_MIN, SAFE, start_time)
        self.dm_xyz = dm_xyz
        self.dm_masses = dm_masses
        self.cat = cat
        self.r200 = r200
        self.SNAP = SNAP
        
    def createCatMajorCOMDM(self):   
        """ Calculates and saves halo shape catalogues"""       
        return
        
cdef class CosmicShapesGadgetHDF5(CosmicShapes):
    
    cdef str HDF5_SNAP_DEST
    cdef str HDF5_GROUP_DEST
    cdef int MIN_NUMBER_STAR_PTCS
    cdef str SNAP
    cdef int SNAP_MAX
    cdef bint withVDisp

    def __init__(self, str HDF5_SNAP_DEST, str HDF5_GROUP_DEST, str CAT_DEST, str VIZ_DEST, str SNAP, int SNAP_MAX, float L_BOX, int MIN_NUMBER_DM_PTCS, int MIN_NUMBER_STAR_PTCS, int D_LOGSTART, int D_LOGEND, int D_BINS, float M_TOL, int N_WALL, int N_MIN, float SAFE, bint withVDisp, float start_time):
        super().__init__(CAT_DEST, VIZ_DEST, L_BOX, MIN_NUMBER_DM_PTCS, D_LOGSTART, D_LOGEND, D_BINS, M_TOL, N_WALL, N_MIN, SAFE, start_time)
        self.HDF5_SNAP_DEST = HDF5_SNAP_DEST
        self.HDF5_GROUP_DEST = HDF5_GROUP_DEST
        self.MIN_NUMBER_STAR_PTCS = MIN_NUMBER_STAR_PTCS
        self.SNAP = SNAP
        self.SNAP_MAX = SNAP_MAX
        self.withVDisp = withVDisp
        
    def vizObjShapeLocal(self, obj_numbers, obj_type = 'dm'):
        """ Visualize local shape of objects with numbers `obj_numbers`"""
        return
        
    def vizObjShapeOvrl(self, obj_numbers, obj_type = 'dm'):
        """ Visualize overall shape of objects with numbers `obj_numbers`"""
        return
    
    def createCatDM(self):
        """ Creates/Loads halo (more precisely: CSH) catalogues from FOF data
        
        Stores R200, masses of halos etc.."""
        return
    
    def createCatMajorCOMGx(self):
        """ Calculates and saves galaxy shape catalogues"""      
        return
    
    def createCatMajorCOMDM(self):   
        """ Calculates and saves halo shape catalogues"""       
        return