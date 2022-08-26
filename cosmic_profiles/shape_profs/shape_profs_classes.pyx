#cython: language_level=3
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot
from numpy.random import default_rng
from matplotlib.font_manager import FontProperties
import matplotlib
matplotlib.rcParams.update({'font.size': 13})
cimport cython
from cosmic_profiles.dens_profs.dens_profs_classes cimport DensProfs, DensProfsHDF5
from cosmic_profiles.common.python_routines import print_status, set_axes_equal, fibonacci_ellipsoid, respectPBCNoRef, isValidSelection
from cosmic_profiles.shape_profs.shape_profs_tools import getGlobalEpsHist, getLocalEpsHist
from cosmic_profiles.gadget_hdf5.get_hdf5 import getHDF5SHDMData, getHDF5SHGxData
import time
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
        :return: density profiles
        :rtype: (N2, r_res) floats"""
        print_status(rank,self.start_time,'Starting {} estDensProfs() with snap {}'.format('direct binning' if direct_binning == True else 'kernel based', self.SNAP))
        isValidSelection(select, self.idx_cat.shape[0])
        if rank == 0:
            if direct_binning:
                if spherical == False:
                    d, q, s, minor, inter, major, obj_centers, obj_masses = self.getShapeCatLocalBase(self.xyz.base, self.masses.base, self.r200.base[select[0]:select[1]+1], self.idx_cat.base[select[0]:select[1]+1], self.obj_size.base[select[0]:select[1]+1], self.D_LOGSTART, self.D_LOGEND, self.D_BINS, self.IT_TOL, self.IT_WALL, self.IT_MIN, reduced, shell_based)
                    dens_profs = self.getDensProfsEllDirectBinningBase(self.xyz.base, self.masses.base, self.r200.base[select[0]:select[1]+1], self.idx_cat.base[select[0]:select[1]+1], self.obj_size.base[select[0]:select[1]+1], np.float32(ROverR200), d, d*q, d*s, major, inter, minor)
                else:
                    dens_profs = self.getDensProfsSphDirectBinningBase(self.xyz.base, self.masses.base, self.r200.base[select[0]:select[1]+1], self.idx_cat.base[select[0]:select[1]+1], self.obj_size.base[select[0]:select[1]+1], np.float32(ROverR200))
            else:
                dens_profs = self.getDensProfsKernelBasedBase(self.xyz.base, self.masses.base, self.r200.base[select[0]:select[1]+1], self.idx_cat.base[select[0]:select[1]+1], self.obj_size.base[select[0]:select[1]+1], np.float32(ROverR200))
            return dens_profs
        else:
            return None
        
    def getShapeCatLocal(self, list select, bint reduced = False, bint shell_based = False):
        """ Get all relevant local shape data
        
        :param select: index of first and last object to look at in the format [idx_first, idx_last]
        :type select: list containing two integers
        :param reduced: whether or not reduced shape tensor (1/r^2 factor)
        :type reduced: boolean
        :param shell_based: whether shell-based or ellipsoid-based algorithm should be run
        :type shell_based: boolean
        :return: d, q, s, minor, inter, major, obj_center, obj_m
        :rtype: 3 x (number_of_objs, D_BINS+1) float arrays, 
            3 x (number_of_objs, D_BINS+1, 3) float arrays, 
            (number_of_objs,3) float array, (number_of_objs,) float array"""
        print_status(rank,self.start_time,'Starting getShapeCatLocal() with snap {0}'.format(self.SNAP))
        if rank == 0:
            d, q, s, minor, inter, major, obj_centers, obj_masses = self.getShapeCatLocalBase(self.xyz.base, self.masses.base, self.r200.base[select[0]:select[1]+1], self.idx_cat.base[select[0]:select[1]+1], self.obj_size.base[select[0]:select[1]+1], self.D_LOGSTART, self.D_LOGEND, self.D_BINS, self.IT_TOL, self.IT_WALL, self.IT_MIN, reduced, shell_based)
            return d, q, s, minor, inter, major, obj_centers, obj_masses
        else:
            return None, None, None, None, None, None, None, None
    
    def getShapeCatGlobal(self, list select, bint reduced = False):
        """ Get all relevant global shape data
        
        :param select: index of first and last object to look at in the format [idx_first, idx_last]
        :type select: list containing two integers
        :param reduced: whether or not reduced shape tensor (1/r^2 factor)
        :type reduced: boolean
        :return: d, q, s, minor, inter, major, obj_center, obj_m
        :rtype: 3 x (number_of_objs, D_BINS+1) float arrays, 
            3 x (number_of_objs, D_BINS+1, 3) float arrays, 
            (number_of_objs,3) float array, (number_of_objs,) float array,
        """
        print_status(rank,self.start_time,'Starting getShapeCatGlobal() with snap {0}'.format(self.SNAP))
        if rank == 0:
            d, q, s, minor, inter, major, obj_centers, obj_masses = self.getShapeCatGlobalBase(self.xyz.base, self.masses.base, self.r200.base[select[0]:select[1]+1], self.idx_cat.base[select[0]:select[1]+1], self.obj_size.base[select[0]:select[1]+1], self.IT_TOL, self.IT_WALL, self.IT_MIN, reduced)
            return d, q, s, minor, inter, major, obj_centers, obj_masses
        else:
            return None, None, None, None, None, None, None, None
    
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
        print_status(rank,self.start_time,'Starting vizLocalShapes() with snap {0}'.format(self.SNAP))
        
        if rank == 0:
            # Retrieve shape information
            d, q, s, minor, inter, major, centers, obj_masses = self.getShapeCatLocalBase(self.xyz.base, self.masses.base, self.r200.base, self.idx_cat.base, self.MIN_NUMBER_PTCS, self.D_LOGSTART, self.D_LOGEND, self.D_BINS, self.IT_TOL, self.IT_WALL, self.IT_MIN, reduced, shell_based)
            del obj_masses
                                    
            # Viz all objects under 'obj_numbers'
            for obj_number in obj_numbers:
                if obj_number >= d.shape[0]:
                    print_status(rank, self.start_time, "Given obj_number {} exceeds the maximum number. There are only {} objects of sufficient resolution. Skip.".format(obj_number, d.shape[0]))
                else:
                    major_obj = major[obj_number]
                    inter_obj = inter[obj_number]
                    minor_obj = minor[obj_number]
                    d_obj = d[obj_number]
                    q_obj = q[obj_number]
                    s_obj = s[obj_number]
                    center = centers[obj_number]
                    obj = np.zeros((self.obj_size[obj_number],3), dtype = np.float32)
                    masses_obj = np.zeros((self.obj_size[obj_number],), dtype = np.float32)
                    for idx, ptc in enumerate(self.idx_cat.base[obj_number]):
                        obj[idx] = self.xyz.base[ptc]
                        masses_obj[idx] = self.masses.base[ptc]
                    obj = respectPBCNoRef(obj, self.L_BOX)
                    # Plotting
                    fig = pyplot.figure()
                    ax = Axes3D(fig, auto_add_to_figure = False)
                    fig.add_axes(ax)
                    # If obj contains too many particles, choose some randomly for display
                    if len(obj) > 5000:
                        rng = default_rng(seed=42)
                        choose = rng.choice(np.arange(len(obj)), (5000,), replace = False)
                    else:
                        choose = np.arange(len(obj))
                    ax.scatter(obj[choose,0],obj[choose,1],obj[choose,2],s=masses_obj[choose]/np.average(masses_obj[choose])*0.3, label = "Particles")
                    ax.scatter(center[0],center[1],center[2],s=50,c="r", label = "COM")
                    
                    ell = fibonacci_ellipsoid(d_obj[-1], q_obj[-1]*d_obj[-1], s_obj[-1]*d_obj[-1], samples=500)
                    rot_matrix = np.hstack((np.reshape(major_obj[-1]/np.linalg.norm(major_obj[-1]), (3,1)), np.reshape(inter_obj[-1]/np.linalg.norm(inter_obj[-1]), (3,1)), np.reshape(minor_obj[-1]/np.linalg.norm(minor_obj[-1]), (3,1))))
                    for j in range(len(ell)): # Transformation into the principal frame
                        ell[j] = np.dot(rot_matrix, np.array(ell[j]))
                    ell_x = np.array([x[0] for x in ell])
                    ell_y = np.array([x[1] for x in ell])
                    ell_z = np.array([x[2] for x in ell])
                    
                    ax.scatter(ell_x+center[0],ell_y+center[1],ell_z+center[2],s=1, c="g", label = "Inferred; a = {:.2f}, b = {:.2f}, c = {:.2f}".format(d_obj[-1], q_obj[-1]*d_obj[-1], s_obj[-1]*d_obj[-1]))
                    for idx in np.arange(self.D_BINS-self.D_BINS//5, self.D_BINS):
                        if idx == self.D_BINS-1:
                            ax.quiver(*center, major_obj[idx][0], major_obj[idx][1], major_obj[idx][2], length=d_obj[idx], color='m', label= "Major")
                            ax.quiver(*center, inter_obj[idx][0], inter_obj[idx][1], inter_obj[idx][2], length=q_obj[idx]*d_obj[idx], color='c', label = "Intermediate")
                            ax.quiver(*center, minor_obj[idx][0], minor_obj[idx][1], minor_obj[idx][2], length=s_obj[idx]*d_obj[idx], color='y', label = "Minor")
                        else:
                            ax.quiver(*center, major_obj[idx][0], major_obj[idx][1], major_obj[idx][2], length=d_obj[idx], color='m')
                            ax.quiver(*center, inter_obj[idx][0], inter_obj[idx][1], inter_obj[idx][2], length=q_obj[idx]*d_obj[idx], color='c')
                            ax.quiver(*center, minor_obj[idx][0], minor_obj[idx][1], minor_obj[idx][2], length=s_obj[idx]*d_obj[idx], color='y')
                    for special in np.arange(-self.D_BINS//5,-self.D_BINS//5+1):
                        ell = fibonacci_ellipsoid(d_obj[special], q_obj[special]*d_obj[special], s_obj[special]*d_obj[special], samples=500)
                        rot_matrix = np.hstack((np.reshape(major_obj[special]/np.linalg.norm(major_obj[special]), (3,1)), np.reshape(inter_obj[special]/np.linalg.norm(inter_obj[special]), (3,1)), np.reshape(minor_obj[special]/np.linalg.norm(minor_obj[special]), (3,1))))
                        for j in range(len(ell)): # Transformation into the principal frame
                            ell[j] = np.dot(rot_matrix, np.array(ell[j])) 
                        ell_x = np.array([x[0] for x in ell])
                        ell_y = np.array([x[1] for x in ell])
                        ell_z = np.array([x[2] for x in ell])
                        ax.scatter(ell_x+center[0],ell_y+center[1],ell_z+center[2],s=1, c="r", label = "Inferred; a = {:.2f}, b = {:.2f}, c = {:.2f}".format(d_obj[special], q_obj[special]*d_obj[special], s_obj[special]*d_obj[special]))
                        ax.quiver(*center, major_obj[special][0], major_obj[special][1], major_obj[special][2], length=d_obj[special], color='limegreen', label= "Major {0}".format(special))
                        ax.quiver(*center, inter_obj[special][0], inter_obj[special][1], inter_obj[special][2], length=q_obj[special]*d_obj[special], color='darkorange', label = "Intermediate {0}".format(special))
                        ax.quiver(*center, minor_obj[special][0], minor_obj[special][1], minor_obj[special][2], length=s_obj[special]*d_obj[special], color='indigo', label = "Minor {0}".format(special))
                    fontP = FontProperties()
                    fontP.set_size('xx-small')
                    plt.legend(bbox_to_anchor=(0.95, 1), loc='upper right', prop=fontP)        
                    plt.xlabel(r"x (Mpc/h)")
                    plt.ylabel(r"y (Mpc/h)")
                    ax.set_zlabel(r"z (Mpc/h)")
                    ax.set_box_aspect([1,1,1])
                    set_axes_equal(ax)
                    fig.savefig("{}/LocalObj{}_{}.pdf".format(VIZ_DEST, obj_number, self.SNAP), bbox_inches='tight')
        
    def vizGlobalShapes(self, obj_numbers, str VIZ_DEST, bint reduced = False):
        """ Visualize global shape of objects with numbers ``obj_numbers``
        
        :param obj_numbers: list of object indices for which to visualize global shapes
        :type obj_numbers: list of ints
        :param VIZ_DEST: visualization folder
        :type VIZ_DEST: string
        :param reduced: whether or not reduced shape tensor (1/r^2 factor)
        :type reduced: boolean"""
        print_status(rank,self.start_time,'Starting vizGlobalShapes() with snap {0}'.format(self.SNAP))

        if rank == 0:
            # Retrieve shape information
            d, q, s, minor, inter, major, centers, obj_masses = self.getShapeCatGlobalBase(self.xyz.base, self.masses.base, self.r200.base, self.idx_cat.base, self.MIN_NUMBER_PTCS, self.IT_TOL, self.IT_WALL, self.IT_MIN, reduced)
            del obj_masses                        
            for obj_number in obj_numbers:
                if obj_number >= d.shape[0]:
                    print_status(rank, self.start_time, "Given obj_number {} exceeds the maximum number. There are only {} objects of sufficient resolution. Skip.".format(obj_number, d.shape[0]))
                else:
                    major_obj = major[obj_number]
                    inter_obj = inter[obj_number]
                    minor_obj = minor[obj_number]
                    d_obj = d[obj_number]
                    q_obj = q[obj_number]
                    s_obj = s[obj_number]
                    center = centers[obj_number]
                    obj = np.zeros((self.obj_size[obj_number],3), dtype = np.float32)
                    masses_obj = np.zeros((self.obj_size[obj_number],), dtype = np.float32)
                    for idx, ptc in enumerate(self.idx_cat.base[obj_number]):
                        obj[idx] = self.xyz.base[ptc]
                        masses_obj[idx] = self.masses.base[ptc]
                    obj = respectPBCNoRef(obj, self.L_BOX)
                    # Plotting
                    fig = pyplot.figure()
                    ax = Axes3D(fig, auto_add_to_figure = False)
                    fig.add_axes(ax)
                    # If obj contains too many particles, choose some randomly for display
                    if len(obj) > 5000:
                        rng = default_rng(seed=42)
                        choose = rng.choice(np.arange(len(obj)), (5000,), replace = False)
                    else:
                        choose = np.arange(len(obj))
                    ax.scatter(obj[choose,0],obj[choose,1],obj[choose,2],s=masses_obj[choose]/np.average(masses_obj[choose])*0.3, label = "Particles")
                    ax.scatter(center[0],center[1],center[2],s=50,c="r", label = "COM")
                    
                    ell = fibonacci_ellipsoid(d_obj[-1], q_obj[-1]*d_obj[-1], s_obj[-1]*d_obj[-1], samples=500)
                    rot_matrix = np.hstack((np.reshape(major_obj[-1]/np.linalg.norm(major_obj[-1]), (3,1)), np.reshape(inter_obj[-1]/np.linalg.norm(inter_obj[-1]), (3,1)), np.reshape(minor_obj[-1]/np.linalg.norm(minor_obj[-1]), (3,1))))
                    for j in range(len(ell)): # Transformation into the principal frame
                        ell[j] = np.dot(rot_matrix, np.array(ell[j]))
                    ell_x = np.array([x[0] for x in ell])
                    ell_y = np.array([x[1] for x in ell])
                    ell_z = np.array([x[2] for x in ell])
                    ax.scatter(ell_x+center[0],ell_y+center[1],ell_z+center[2],s=1, c="g", label = "Inferred; a = {:.2f}, b = {:.2f}, c = {:.2f}".format(d_obj[-1], q_obj[-1]*d_obj[-1], s_obj[-1]*d_obj[-1]))
                    ax.quiver(*center, major_obj[0][0], major_obj[0][1], major_obj[0][2], length=d_obj[0], color='m', label= "Major")
                    ax.quiver(*center, inter_obj[0][0], inter_obj[0][1], inter_obj[0][2], length=q_obj[0]*d_obj[0], color='c', label = "Intermediate")
                    ax.quiver(*center, minor_obj[0][0], minor_obj[0][1], minor_obj[0][2], length=s_obj[0]*d_obj[0], color='y', label = "Minor")
                    fontP = FontProperties()
                    fontP.set_size('xx-small')
                    plt.legend(bbox_to_anchor=(0.95, 1), loc='upper right', prop=fontP)  
                    plt.xlabel(r"x (Mpc/h)")
                    plt.ylabel(r"y (Mpc/h)")
                    ax.set_zlabel(r"z (Mpc/h)")
                    ax.set_box_aspect([1,1,1])
                    set_axes_equal(ax)
                    fig.savefig("{}/GlobalObj{}_{}.pdf".format(VIZ_DEST, obj_number, self.SNAP), bbox_inches='tight')
    
    def plotGlobalEpsHist(self, HIST_NB_BINS, str VIZ_DEST, list select):
        """ Plot global ellipticity histogram
        
        :param HIST_NB_BINS: number of histogram bins
        :type HIST_NB_BINS: int
        :param VIZ_DEST: visualization folder
        :type VIZ_DEST: string
        :param select: index of first and last object to look at in the format [idx_first, idx_last]
        :type select: list containing two integers"""
        print_status(rank,self.start_time,'Starting plotGlobalEpsHist() with snap {0}'.format(self.SNAP))
        
        if rank == 0:
            suffix = '_'
            getGlobalEpsHist(self.xyz.base, self.masses.base, self.idx_cat.base[select[0]:select[1]+1], self.obj_size.base[select[0]:select[1]+1], self.L_BOX, self.CENTER, VIZ_DEST, self.SNAP, suffix = suffix, HIST_NB_BINS = HIST_NB_BINS)
        
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
        print_status(rank,self.start_time,'Starting plotLocalEpsHist() with snap {0}'.format(self.SNAP))
        
        if rank == 0:
            suffix = '_'
            getLocalEpsHist(self.xyz.base, self.masses.base, self.r200.base[select[0]:select[1]+1], self.idx_cat.base[select[0]:select[1]+1], self.obj_size.base[select[0]:select[1]+1], self.L_BOX, self.CENTER, VIZ_DEST, self.SNAP, frac_r200, suffix = suffix, HIST_NB_BINS = HIST_NB_BINS)
    
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
        print_status(rank,self.start_time,'Starting plotLocalTHist() with snap {0}'.format(self.SNAP))
        if rank == 0:
            suffix = '_'
            self.plotLocalTHistBase(self.xyz.base, self.masses.base, self.r200.base[select[0]:select[1]+1], self.idx_cat.base[select[0]:select[1]+1], self.obj_size.base[select[0]:select[1]+1], self.D_LOGSTART, self.D_LOGEND, self.D_BINS, self.IT_TOL, self.IT_WALL, self.IT_MIN, VIZ_DEST, HIST_NB_BINS, frac_r200, reduced, shell_based, suffix = suffix)
    
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
        print_status(rank,self.start_time,'Starting plotGlobalTHist() with snap {0}'.format(self.SNAP))
        if rank == 0:
            suffix = '_'
            self.plotGlobalTHistBase(self.xyz.base, self.masses.base, self.r200.base[select[0]:select[1]+1], self.idx_cat.base[select[0]:select[1]+1], self.obj_size.base[select[0]:select[1]+1], self.IT_TOL, self.IT_WALL, self.IT_MIN, VIZ_DEST, HIST_NB_BINS, reduced, suffix = suffix)
    
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
        print_status(rank,self.start_time,'Starting plotShapeProfs() with snap {0}'.format(self.SNAP))
        if rank == 0:
            suffix = '_'
            self.plotShapeProfsBase(self.xyz.base, self.masses.base, self.r200.base[select[0]:select[1]+1], self.idx_cat.base[select[0]:select[1]+1], self.obj_size.base[select[0]:select[1]+1], self.D_LOGSTART, self.D_LOGEND, self.D_BINS, self.IT_TOL, self.IT_WALL, self.IT_MIN, VIZ_DEST, reduced, shell_based, nb_bins, suffix = suffix)
    
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
        print_status(rank,self.start_time,'Starting dumpShapeCatLocal() with snap {0}'.format(self.SNAP))
        if rank == 0:
            suffix = '_'
            self.dumpShapeCatLocalBase(self.xyz.base, self.masses.base, self.r200.base[select[0]:select[1]+1], self.idx_cat.base[select[0]:select[1]+1], self.obj_size.base[select[0]:select[1]+1], self.D_LOGSTART, self.D_LOGEND, self.D_BINS, self.IT_TOL, self.IT_WALL, self.IT_MIN, CAT_DEST, suffix, reduced, shell_based)
    
    def dumpShapeCatGlobal(self, str CAT_DEST, list select, bint reduced = False):
        """ Dumps all relevant global shape data into ``CAT_DEST``
        
        :param CAT_DEST: catalogue folder
        :type CAT_DEST: string
        :param select: index of first and last object to look at in the format [idx_first, idx_last]
        :type select: list containing two integers
        :param reduced: whether or not reduced shape tensor (1/r^2 factor)
        :type reduced: boolean"""
        print_status(rank,self.start_time,'Starting dumpShapeCatGlobal() with snap {0}'.format(self.SNAP))
        if rank == 0:
            suffix = '_'
            self.dumpShapeCatGlobalBase(self.xyz.base, self.masses.base, self.idx_cat.base[select[0]:select[1]+1], self.obj_size.base[select[0]:select[1]+1], self.IT_TOL, self.IT_WALL, self.IT_MIN, CAT_DEST, suffix, reduced)
    
    def getObjInfo(self):
        """ Print basic info about the objects"""
        print_status(rank,self.start_time,'Starting getObjInfo() with snap {0}'.format(self.SNAP))
        obj_type = 'unspecified'
        self.getObjInfoBase(self.xyz.base, self.masses.base, self.idx_cat.base, obj_type)

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
    
    def __init__(self, str HDF5_SNAP_DEST, str HDF5_GROUP_DEST, str SNAP, float L_BOX, int MIN_NUMBER_PTCS, int MIN_NUMBER_STAR_PTCS, int D_LOGSTART, int D_LOGEND, int D_BINS, float IT_TOL, int IT_WALL, int IT_MIN, str CENTER, bint WANT_RVIR):
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
        :param WANT_RVIR: Whether or not we want quantities (e.g. D_LOGSTART) expressed 
            with respect to the virial radius R_vir or the overdensity radius R_200
        :type WANT_RVIR: boolean"""
        super().__init__(HDF5_SNAP_DEST, HDF5_GROUP_DEST, SNAP, L_BOX, MIN_NUMBER_PTCS, MIN_NUMBER_STAR_PTCS, CENTER, WANT_RVIR)
        self.D_LOGSTART = D_LOGSTART
        self.D_LOGEND = D_LOGEND
        self.D_BINS = D_BINS
        self.IT_TOL = IT_TOL
        self.IT_WALL = IT_WALL
        self.IT_MIN = IT_MIN
        
    def estDensProfs(self, ROverR200, list select, bint direct_binning = True, bint spherical = True, bint reduced = False, bint shell_based = False, str obj_type = 'dm'):
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
        :param obj_type: either 'dm' or 'gx', depending on what catalogue we are looking at
        :type obj_type: string
        :return: density profiles
        :rtype: (N2, r_res) floats"""
        print_status(rank,self.start_time,'Starting {} estDensProfs() with snap {}'.format('direct binning' if direct_binning == True else 'kernel based', self.SNAP))
        xyz, masses, MIN_NUMBER_PTCS = self.getXYZMasses(obj_type)
        if rank == 0:
            isValidSelection(select, len(self.getIdxCat(obj_type)[0]))
            if direct_binning:
                if spherical:
                    dens_profs = self.getDensProfsSphDirectBinningBase(xyz, masses, self.r200.base[select[0]:select[1]+1], self.getIdxCat(obj_type)[0][select[0]:select[1]+1], self.getIdxCat(obj_type)[1][select[0]:select[1]+1], np.float32(ROverR200))
                else:
                    d, q, s, minor, inter, major, obj_centers, obj_masses = self.getShapeCatLocalBase(xyz, masses, self.r200.base[select[0]:select[1]+1], self.getIdxCat(obj_type)[0][select[0]:select[1]+1], self.getIdxCat(obj_type)[1][select[0]:select[1]+1], self.D_LOGSTART, self.D_LOGEND, self.D_BINS, self.IT_TOL, self.IT_WALL, self.IT_MIN, reduced, shell_based)
                    dens_profs = self.getDensProfsEllDirectBinningBase(xyz, masses, self.r200.base[select[0]:select[1]+1], self.getIdxCat(obj_type)[0][select[0]:select[1]+1], self.getIdxCat(obj_type)[1][select[0]:select[1]+1], np.float32(ROverR200), d, d*q, d*s, major, inter, minor)
                    del d; del q; del s; del minor; del inter; del major
            else:
                dens_profs = self.getDensProfsKernelBasedBase(xyz, masses, self.r200.base[select[0]:select[1]+1], self.getIdxCat(obj_type)[0][select[0]:select[1]+1], self.getIdxCat(obj_type)[1][select[0]:select[1]+1], np.float32(ROverR200))
            del xyz; del masses
            return dens_profs
        else:
            del xyz; del masses
            return None
    
    def getShapeCatLocal(self, list select, bint reduced = False, bint shell_based = False, str obj_type = 'dm'):
        """ Get all relevant local shape data
        
        :param select: index of first and last object to look at in the format [idx_first, idx_last]
        :type select: list containing two integers
        :param reduced: whether or not reduced shape tensor (1/r^2 factor)
        :type reduced: boolean
        :param shell_based: whether shell-based or ellipsoid-based algorithm should be run
        :type shell_based: boolean
        :param obj_type: either 'dm' or 'gx', depending on what catalogue we are looking at
        :type obj_type: string
        :return: d, q, s, minor, inter, major, obj_center, obj_m
        :rtype: 3 x (number_of_objs, D_BINS+1) float arrays, 
            3 x (number_of_objs, D_BINS+1, 3) float arrays, 
            (number_of_objs,3) float array, (number_of_objs,) float array"""
        print_status(rank,self.start_time,'Starting getShapeCatLocal() with snap {0} for obj_type {1}'.format(self.SNAP, obj_type))
        xyz, masses, MIN_NUMBER_PTCS = self.getXYZMasses(obj_type)
            
        if rank == 0:
            self.getIdxCat(obj_type)[0]
            d, q, s, minor, inter, major, obj_centers, obj_masses = self.getShapeCatLocalBase(xyz, masses, self.r200.base[select[0]:select[1]+1], self.getIdxCat(obj_type)[0][select[0]:select[1]+1], self.getIdxCat(obj_type)[1][select[0]:select[1]+1], self.D_LOGSTART, self.D_LOGEND, self.D_BINS, self.IT_TOL, self.IT_WALL, self.IT_MIN, reduced, shell_based)
            del xyz; del masses
            return d, q, s, minor, inter, major, obj_centers, obj_masses
        else:
            del xyz; del masses
            return None, None, None, None, None, None, None, None
    
    def getShapeCatGlobal(self, list select, bint reduced = False, str obj_type = 'dm'):
        """ Get all relevant global shape data
        
        :param select: index of first and last object to look at in the format [idx_first, idx_last]
        :type select: list containing two integers
        :param reduced: whether or not reduced shape tensor (1/r^2 factor)
        :type reduced: boolean
        :param obj_type: either 'dm' or 'gx', depending on what catalogue we are looking at
        :type obj_type: string
        :return: d, q, s, minor, inter, major, obj_center, obj_m
        :rtype: 3 x (number_of_objs, D_BINS+1) float arrays, 
            3 x (number_of_objs, D_BINS+1, 3) float arrays, 
            (number_of_objs,3) float array, (number_of_objs,) float array"""
        print_status(rank,self.start_time,'Starting getShapeCatGlobal() with snap {0} for obj_type {1}'.format(self.SNAP, obj_type))
        xyz, masses, MIN_NUMBER_PTCS = self.getXYZMasses(obj_type)
        
        if rank == 0:
            self.getIdxCat(obj_type)[0]
            d, q, s, minor, inter, major, obj_centers, obj_masses = self.getShapeCatGlobalBase(xyz, masses, self.r200.base[select[0]:select[1]+1], self.getIdxCat(obj_type)[0][select[0]:select[1]+1], self.getIdxCat(obj_type)[1][select[0]:select[1]+1], self.IT_TOL, self.IT_WALL, self.IT_MIN, reduced)
            del xyz; del masses
            return d, q, s, minor, inter, major, obj_centers, obj_masses
        else:
            del xyz; del masses
            return None, None, None, None, None, None, None, None
        
    def getShapeCatVelLocal(self, list select, bint reduced = False, bint shell_based = False, str obj_type = 'dm'):
        """ Get all relevant local velocity shape data
        
        :param select: index of first and last object to look at in the format [idx_first, idx_last]
        :type select: list containing two integers
        :param reduced: whether or not reduced shape tensor (1/r^2 factor)
        :type reduced: boolean
        :param shell_based: whether shell-based or ellipsoid-based algorithm should be run
        :type shell_based: boolean
        :param obj_type: either 'dm' or 'gx', depending on what catalogue we are looking at
        :type obj_type: string
        :return: d, q, s, minor, inter, major, obj_center, obj_m
        :rtype: 3 x (number_of_objs, D_BINS+1) float arrays,
            3 x (number_of_objs, D_BINS+1, 3) float arrays, 
            (number_of_objs,3) float array, (number_of_objs,) float array"""
        print_status(rank,self.start_time,'Starting getShapeCatVelLocal() with snap {0} for obj_type {1}'.format(self.SNAP, obj_type))
        xyz, masses, MIN_NUMBER_PTCS = self.getXYZMasses(obj_type)
        velxyz = self.getVelXYZ(obj_type)
        if rank == 0:
            self.getIdxCat(obj_type)[0]
            d, q, s, minor, inter, major, obj_centers, obj_masses = self.getShapeCatVelLocalBase(xyz, velxyz, masses, self.r200.base[select[0]:select[1]+1], self.getIdxCat(obj_type)[0][select[0]:select[1]+1], self.getIdxCat(obj_type)[1][select[0]:select[1]+1], self.D_LOGSTART, self.D_LOGEND, self.D_BINS, self.IT_TOL, self.IT_WALL, self.IT_MIN, reduced, shell_based)
            del xyz; del velxyz; del masses
            return d, q, s, minor, inter, major, obj_centers, obj_masses
        else:
            del xyz; del velxyz; del masses
            return None, None, None, None, None, None, None, None
    
    def getShapeCatVelGlobal(self, list select, bint reduced = False, str obj_type = 'dm'):
        """ Get all relevant global velocity shape data
        
        :param select: index of first and last object to look at in the format [idx_first, idx_last]
        :type select: list containing two integers
        :param reduced: whether or not reduced shape tensor (1/r^2 factor)
        :type reduced: boolean
        :param obj_type: either 'dm' or 'gx', depending on what catalogue we are looking at
        :type obj_type: string
        :return: d, q, s, minor, inter, major, obj_center, obj_m
        :rtype: 3 x (number_of_objs, D_BINS+1) float arrays, 
            3 x (number_of_objs, D_BINS+1, 3) float arrays, 
            (number_of_objs,3) float array, (number_of_objs,) float array,
        """
        print_status(rank,self.start_time,'Starting getShapeCatVelGlobal() with snap {0} for obj_type {1}'.format(self.SNAP, obj_type))
        xyz, masses, MIN_NUMBER_PTCS = self.getXYZMasses(obj_type)
        velxyz = self.getVelXYZ(obj_type)
        if rank == 0:
            self.getIdxCat(obj_type)[0]
            d, q, s, minor, inter, major, obj_centers, obj_masses = self.getShapeCatVelGlobalBase(xyz, velxyz, masses, self.r200.base[select[0]:select[1]+1], self.getIdxCat(obj_type)[0][select[0]:select[1]+1], self.getIdxCat(obj_type)[1][select[0]:select[1]+1], self.IT_TOL, self.IT_WALL, self.IT_MIN, self.CENTER, self.SAFE, reduced)
            del xyz; del velxyz; del masses
            return d, q, s, minor, inter, major, obj_centers, obj_masses
        else:
            del xyz; del velxyz; del masses
            return None, None, None, None, None, None, None, None
    
    def vizLocalShapes(self, obj_numbers, str VIZ_DEST, bint reduced = False, bint shell_based = False, str obj_type = 'dm'):
        """ Visualize local shape of objects with numbers ``obj_numbers``
        
        :param obj_numbers: list of object indices for which to visualize local shapes
        :type obj_numbers: list of int
        :param VIZ_DEST: visualization folder
        :type VIZ_DEST: strings
        :param reduced: whether or not reduced shape tensor (1/r^2 factor)
        :type reduced: boolean
        :param shell_based: whether shell-based or ellipsoid-based algorithm should be run
        :type shell_based: boolean
        :param obj_type: either 'dm' or 'gx', depending on what catalogue we are looking at
        :type obj_type: string"""
        print_status(rank,self.start_time,'Starting vizLocalShapes() with snap {0} for obj_type {1}'.format(self.SNAP, obj_type))
        
        xyz, masses, MIN_NUMBER_PTCS = self.getXYZMasses(obj_type)
        suffix = '_{}_'.format(obj_type)
        
        if rank == 0:
            # Retrieve shape information
            d, q, s, minor, inter, major, centers, obj_m = self.getShapeCatLocalBase(xyz, masses, self.r200.base, self.getIdxCat(obj_type)[0], self.getIdxCat(obj_type)[1], self.D_LOGSTART, self.D_LOGEND, self.D_BINS, self.IT_TOL, self.IT_WALL, self.IT_MIN, reduced, shell_based)
            del obj_m
                        
            # Viz all objects under 'obj_numbers'
            for obj_number in obj_numbers:
                if obj_number >= d.shape[0]:
                    print_status(rank, self.start_time, "Given obj_number {} exceeds the maximum number. There are only {} objects that have sufficient resolution. Skip.".format(obj_number, d.shape[0]))
                else:
                    major_obj = major[obj_number]
                    inter_obj = inter[obj_number]
                    minor_obj = minor[obj_number]
                    d_obj = d[obj_number]
                    q_obj = q[obj_number]
                    s_obj = s[obj_number]
                    center = centers[obj_number]
                    obj = np.zeros((len(self.getIdxCat(obj_type)[0][obj_number]),3), dtype = np.float32)
                    masses_obj = np.zeros((len(self.getIdxCat(obj_type)[0][obj_number]),), dtype = np.float32)
                    for idx, ptc in enumerate(self.getIdxCat(obj_type)[0][obj_number]):
                        obj[idx] = xyz[ptc]
                        masses_obj[idx] = masses[ptc]
                    obj = respectPBCNoRef(obj, self.L_BOX)
                    # Plotting
                    fig = pyplot.figure()
                    ax = Axes3D(fig, auto_add_to_figure = False)
                    fig.add_axes(ax)
                    # If obj contains too many particles, choose some randomly for display
                    if len(obj) > 5000:
                        rng = default_rng(seed=42)
                        choose = rng.choice(np.arange(len(obj)), (5000,), replace = False)
                    else:
                        choose = np.arange(len(obj))
                    ax.scatter(obj[choose,0],obj[choose,1],obj[choose,2],s=masses_obj[choose]/np.average(masses_obj[choose])*0.3, label = "Particles")
                    ax.scatter(center[0],center[1],center[2],s=50,c="r", label = "COM")
                    
                    ell = fibonacci_ellipsoid(d_obj[-1], q_obj[-1]*d_obj[-1], s_obj[-1]*d_obj[-1], samples=500)
                    rot_matrix = np.hstack((np.reshape(major_obj[-1]/np.linalg.norm(major_obj[-1]), (3,1)), np.reshape(inter_obj[-1]/np.linalg.norm(inter_obj[-1]), (3,1)), np.reshape(minor_obj[-1]/np.linalg.norm(minor_obj[-1]), (3,1))))
                    for j in range(len(ell)): # Transformation into the principal frame
                        ell[j] = np.dot(rot_matrix, np.array(ell[j]))
                    ell_x = np.array([x[0] for x in ell])
                    ell_y = np.array([x[1] for x in ell])
                    ell_z = np.array([x[2] for x in ell])
                    
                    ax.scatter(ell_x+center[0],ell_y+center[1],ell_z+center[2],s=1, c="g", label = "Inferred; a = {:.2f}, b = {:.2f}, c = {:.2f}".format(d_obj[-1], q_obj[-1]*d_obj[-1], s_obj[-1]*d_obj[-1]))
                    for idx in np.arange(self.D_BINS-self.D_BINS//5, self.D_BINS):
                        if idx == self.D_BINS-1:
                            ax.quiver(*center, major_obj[idx][0], major_obj[idx][1], major_obj[idx][2], length=d_obj[idx], color='m', label= "Major")
                            ax.quiver(*center, inter_obj[idx][0], inter_obj[idx][1], inter_obj[idx][2], length=q_obj[idx]*d_obj[idx], color='c', label = "Intermediate")
                            ax.quiver(*center, minor_obj[idx][0], minor_obj[idx][1], minor_obj[idx][2], length=s_obj[idx]*d_obj[idx], color='y', label = "Minor")
                        else:
                            ax.quiver(*center, major_obj[idx][0], major_obj[idx][1], major_obj[idx][2], length=d_obj[idx], color='m')
                            ax.quiver(*center, inter_obj[idx][0], inter_obj[idx][1], inter_obj[idx][2], length=q_obj[idx]*d_obj[idx], color='c')
                            ax.quiver(*center, minor_obj[idx][0], minor_obj[idx][1], minor_obj[idx][2], length=s_obj[idx]*d_obj[idx], color='y')
                    for special in np.arange(-self.D_BINS//5,-self.D_BINS//5+1):
                        ell = fibonacci_ellipsoid(d_obj[special], q_obj[special]*d_obj[special], s_obj[special]*d_obj[special], samples=500)
                        rot_matrix = np.hstack((np.reshape(major_obj[special]/np.linalg.norm(major_obj[special]), (3,1)), np.reshape(inter_obj[special]/np.linalg.norm(inter_obj[special]), (3,1)), np.reshape(minor_obj[special]/np.linalg.norm(minor_obj[special]), (3,1))))
                        for j in range(len(ell)): # Transformation into the principal frame
                            ell[j] = np.dot(rot_matrix, np.array(ell[j])) 
                        ell_x = np.array([x[0] for x in ell])
                        ell_y = np.array([x[1] for x in ell])
                        ell_z = np.array([x[2] for x in ell])
                        ax.scatter(ell_x+center[0],ell_y+center[1],ell_z+center[2],s=1, c="r", label = "Inferred; a = {:.2f}, b = {:.2f}, c = {:.2f}".format(d_obj[special], q_obj[special]*d_obj[special], s_obj[special]*d_obj[special]))
                        ax.quiver(*center, major_obj[special][0], major_obj[special][1], major_obj[special][2], length=d_obj[special], color='limegreen', label= "Major {0}".format(special))
                        ax.quiver(*center, inter_obj[special][0], inter_obj[special][1], inter_obj[special][2], length=q_obj[special]*d_obj[special], color='darkorange', label = "Intermediate {0}".format(special))
                        ax.quiver(*center, minor_obj[special][0], minor_obj[special][1], minor_obj[special][2], length=s_obj[special]*d_obj[special], color='indigo', label = "Minor {0}".format(special))
                    fontP = FontProperties()
                    fontP.set_size('xx-small')
                    plt.legend(bbox_to_anchor=(0.95, 1), loc='upper right', prop=fontP)        
                    plt.xlabel(r"x (Mpc/h)")
                    plt.ylabel(r"y (Mpc/h)")
                    ax.set_zlabel(r"z (Mpc/h)")
                    ax.set_box_aspect([1,1,1])
                    set_axes_equal(ax)
                    fig.savefig("{}/LocalObj{}{}{}.pdf".format(VIZ_DEST, obj_number, suffix, self.SNAP), bbox_inches='tight')
        
    def vizGlobalShapes(self, obj_numbers, str VIZ_DEST, bint reduced = False, str obj_type = 'dm'):
        """ Visualize global shape of objects with numbers ``obj_numbers``
        
        :param obj_numbers: list of object indices for which to visualize global shapes
        :type obj_numbers: list of ints
        :param VIZ_DEST: visualization folder
        :type VIZ_DEST: string
        :param reduced: whether or not reduced shape tensor (1/r^2 factor)
        :type reduced: boolean
        :param obj_type: either 'dm' or 'gx', depending on what catalogue we are looking at
        :type obj_type: string"""
        print_status(rank,self.start_time,'Starting vizGlobalShapes() with snap {0} for obj_type {1}'.format(self.SNAP, obj_type))

        xyz, masses, MIN_NUMBER_PTCS = self.getXYZMasses(obj_type)
        suffix = '_{}_'.format(obj_type)
        
        if rank == 0:
            # Retrieve shape information
            d, q, s, minor, inter, major, centers, obj_m = self.getShapeCatGlobalBase(xyz, masses, self.r200.base, self.getIdxCat(obj_type)[0], self.getIdxCat(obj_type)[1], self.IT_TOL, self.IT_WALL, self.IT_MIN, reduced)
            del obj_m
                      
            # Viz all objects under 'obj_numbers'
            for obj_number in obj_numbers:
                if obj_number >= d.shape[0]:
                    print_status(rank, self.start_time, "Given obj_number {} exceeds the maximum number. There are only {} objects. Skip.".format(obj_number, d.shape[0]))
                else:
                    major_obj = major[obj_number]
                    inter_obj = inter[obj_number]
                    minor_obj = minor[obj_number]
                    d_obj = d[obj_number]
                    q_obj = q[obj_number]
                    s_obj = s[obj_number]
                    center = centers[obj_number]
                    obj = np.zeros((len(self.getIdxCat(obj_type)[0][obj_number]),3), dtype = np.float32)
                    masses_obj = np.zeros((len(self.getIdxCat(obj_type)[0][obj_number]),), dtype = np.float32)
                    for idx, ptc in enumerate(self.getIdxCat(obj_type)[0][obj_number]):
                        obj[idx] = xyz[ptc]
                        masses_obj[idx] = masses[ptc]
                    obj = respectPBCNoRef(obj, self.L_BOX)
                    # Plotting
                    fig = pyplot.figure()
                    ax = Axes3D(fig, auto_add_to_figure = False)
                    fig.add_axes(ax)
                    # If obj contains too many particles, choose some randomly for display
                    if len(obj) > 5000:
                        rng = default_rng(seed=42)
                        choose = rng.choice(np.arange(len(obj)), (5000,), replace = False)
                    else:
                        choose = np.arange(len(obj))
                    ax.scatter(obj[choose,0],obj[choose,1],obj[choose,2],s=masses_obj[choose]/np.average(masses_obj[choose])*0.3, label = "Particles")
                    ax.scatter(center[0],center[1],center[2],s=50,c="r", label = "COM")
                    
                    ell = fibonacci_ellipsoid(d_obj[-1], q_obj[-1]*d_obj[-1], s_obj[-1]*d_obj[-1], samples=500)
                    rot_matrix = np.hstack((np.reshape(major_obj[-1]/np.linalg.norm(major_obj[-1]), (3,1)), np.reshape(inter_obj[-1]/np.linalg.norm(inter_obj[-1]), (3,1)), np.reshape(minor_obj[-1]/np.linalg.norm(minor_obj[-1]), (3,1))))
                    for j in range(len(ell)): # Transformation into the principal frame
                        ell[j] = np.dot(rot_matrix, np.array(ell[j]))
                    ell_x = np.array([x[0] for x in ell])
                    ell_y = np.array([x[1] for x in ell])
                    ell_z = np.array([x[2] for x in ell])
                    ax.scatter(ell_x+center[0],ell_y+center[1],ell_z+center[2],s=1, c="g", label = "Inferred; a = {:.2f}, b = {:.2f}, c = {:.2f}".format(d_obj[-1], q_obj[-1]*d_obj[-1], s_obj[-1]*d_obj[-1]))
                    ax.quiver(*center, major_obj[0][0], major_obj[0][1], major_obj[0][2], length=d_obj[0], color='m', label= "Major")
                    ax.quiver(*center, inter_obj[0][0], inter_obj[0][1], inter_obj[0][2], length=q_obj[0]*d_obj[0], color='c', label = "Intermediate")
                    ax.quiver(*center, minor_obj[0][0], minor_obj[0][1], minor_obj[0][2], length=s_obj[0]*d_obj[0], color='y', label = "Minor")
                    fontP = FontProperties()
                    fontP.set_size('xx-small')
                    plt.legend(bbox_to_anchor=(0.95, 1), loc='upper right', prop=fontP)  
                    plt.xlabel(r"x (Mpc/h)")
                    plt.ylabel(r"y (Mpc/h)")
                    ax.set_zlabel(r"z (Mpc/h)")
                    ax.set_box_aspect([1,1,1])
                    set_axes_equal(ax)
                    fig.savefig("{}/GlobalObj{}{}{}.pdf".format(VIZ_DEST, obj_number, suffix, self.SNAP), bbox_inches='tight')
    
    def plotGlobalEpsHist(self, HIST_NB_BINS, str VIZ_DEST, list select, str obj_type = 'dm'):
        """ Plot global ellipticity histogram
        
        :param HIST_NB_BINS: number of histogram bins
        :type HIST_NB_BINS: int
        :param VIZ_DEST: visualization folder
        :type VIZ_DEST: string
        :param select: index of first and last object to look at in the format [idx_first, idx_last]
        :type select: list containing two integers
        :param obj_type: either 'dm' or 'gx', depending on what catalogue we are looking at
        :type obj_type: string"""
        print_status(rank,self.start_time,'Starting plotGlobalEpsHist() with snap {0} for obj_type {1}'.format(self.SNAP, obj_type))
        
        xyz, masses, MIN_NUMBER_PTCS = self.getXYZMasses(obj_type)
        if rank != 0:
            del xyz; del masses
        suffix = '_{}_'.format(obj_type)
        
        if rank == 0:
            self.getIdxCat(obj_type)[0]
            getGlobalEpsHist(xyz, masses, self.getIdxCat(obj_type)[0][select[0]:select[1]+1], self.getIdxCat(obj_type)[1][select[0]:select[1]+1], self.L_BOX, self.CENTER, VIZ_DEST, self.SNAP, suffix = suffix, HIST_NB_BINS = HIST_NB_BINS)
            del xyz; del masses

    def plotLocalEpsHist(self, frac_r200, HIST_NB_BINS, str VIZ_DEST, list select, str obj_type = 'dm'):
        """ Plot local ellipticity histogram at depth ``frac_r200``
        
        :param frac_r200: depth of objects to plot ellipticity, in units of R200
        :type frac_r200: float
        :param HIST_NB_BINS: number of histogram bins
        :type HIST_NB_BINS: int
        :param VIZ_DEST: visualization folder
        :type VIZ_DEST: string
        :param select: index of first and last object to look at in the format [idx_first, idx_last]
        :type select: list containing two integers
        :param obj_type: either 'dm' or 'gx', depending on what catalogue we are looking at
        :type obj_type: string"""
        print_status(rank,self.start_time,'Starting plotLocalEpsHist() with snap {0} for obj_type {1}'.format(self.SNAP, obj_type))
        
        xyz, masses, MIN_NUMBER_PTCS = self.getXYZMasses(obj_type)
        if rank != 0:
            del xyz; del masses
        suffix = '_{}_'.format(obj_type)
            
        if rank == 0:
            self.getIdxCat(obj_type)[0]
            getLocalEpsHist(xyz, masses, self.r200.base[select[0]:select[1]+1], self.getIdxCat(obj_type)[0][select[0]:select[1]+1], self.getIdxCat(obj_type)[1][select[0]:select[1]+1], self.L_BOX, self.CENTER, VIZ_DEST, self.SNAP, frac_r200, suffix = suffix, HIST_NB_BINS = HIST_NB_BINS)
            del xyz; del masses
    
    def plotLocalTHist(self, HIST_NB_BINS, str VIZ_DEST, frac_r200, list select, bint reduced = False, bint shell_based = False, str obj_type = 'dm'):
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
        :type shell_based: boolean
        :param obj_type: either 'dm' or 'gx', depending on what catalogue we are looking at
        :type obj_type: string"""
        print_status(rank,self.start_time,'Starting plotLocalTHist() with snap {0}'.format(self.SNAP))
        
        xyz, masses, MIN_NUMBER_PTCS = self.getXYZMasses(obj_type)
        if rank != 0:
            del xyz; del masses
        suffix = '_{}_'.format(obj_type)
            
        if rank == 0:
            self.getIdxCat(obj_type)[0]
            self.plotLocalTHistBase(xyz, masses, self.r200.base[select[0]:select[1]+1], self.getIdxCat(obj_type)[0][select[0]:select[1]+1], self.getIdxCat(obj_type)[1][select[0]:select[1]+1], self.D_LOGSTART, self.D_LOGEND, self.D_BINS, self.IT_TOL, self.IT_WALL, self.IT_MIN, VIZ_DEST, HIST_NB_BINS, frac_r200, reduced, shell_based, suffix = suffix)
            del xyz; del masses
    
    def plotGlobalTHist(self, HIST_NB_BINS, str VIZ_DEST, list select, bint reduced = False, str obj_type = 'dm'):
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
        :type reduced: boolean
        :param obj_type: either 'dm' or 'gx', depending on what catalogue we are looking at
        :type obj_type: string"""
        print_status(rank,self.start_time,'Starting plotGlobalTHist() with snap {0} for obj_type {1}'.format(self.SNAP, obj_type))
        
        xyz, masses, MIN_NUMBER_PTCS = self.getXYZMasses(obj_type)
        if rank != 0:
            del xyz; del masses
        suffix = '_{}_'.format(obj_type)
            
        if rank == 0:
            self.getIdxCat(obj_type)[0]
            self.plotGlobalTHistBase(xyz, masses, self.r200.base[select[0]:select[1]+1], self.getIdxCat(obj_type)[0][select[0]:select[1]+1], self.getIdxCat(obj_type)[1][select[0]:select[1]+1], self.IT_TOL, self.IT_WALL, self.IT_MIN, VIZ_DEST, HIST_NB_BINS, reduced, suffix = suffix)
            del xyz; del masses
        
    def plotShapeProfs(self, int nb_bins, str VIZ_DEST, list select, bint reduced = False, bint shell_based = False, str obj_type = 'dm'):
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
        :type shell_based: boolean
        :param obj_type: either 'dm' or 'gx', depending on what catalogue we are looking at
        :type obj_type: string"""
        print_status(rank,self.start_time,'Starting plotShapeProfs() with snap {0} for obj_type {1}'.format(self.SNAP, obj_type))
        
        xyz, masses, MIN_NUMBER_PTCS = self.getXYZMasses(obj_type)
        if rank != 0:
            del xyz; del masses
        suffix = '_{}_'.format(obj_type)
        
        if rank == 0:
            self.getIdxCat(obj_type)[0]
            self.plotShapeProfsBase(xyz, masses, self.r200.base[select[0]:select[1]+1], self.getIdxCat(obj_type)[0][select[0]:select[1]+1], self.getIdxCat(obj_type)[1][select[0]:select[1]+1], self.D_LOGSTART, self.D_LOGEND, self.D_BINS, self.IT_TOL, self.IT_WALL, self.IT_MIN, VIZ_DEST, reduced, shell_based, nb_bins, suffix = suffix)
            del xyz; del masses

    def dumpShapeCatLocal(self, str CAT_DEST, list select, bint reduced = False, bint shell_based = False, str obj_type = 'dm'):
        """ Dumps all relevant local shape data into ``CAT_DEST``
        
        :param CAT_DEST: catalogue folder
        :type CAT_DEST: string
        :param select: index of first and last object to look at in the format [idx_first, idx_last]
        :type select: list containing two integers
        :param reduced: whether or not reduced shape tensor (1/r^2 factor)
        :type reduced: boolean
        :param shell_based: whether shell-based or ellipsoid-based algorithm should be run
        :type shell_based: boolean
        :param obj_type: either 'dm' or 'gx', depending on what catalogue we are looking at
        :type obj_type: string"""
        print_status(rank,self.start_time,'Starting dumpShapeCatLocal() with snap {0} for obj_type {1}'.format(self.SNAP, obj_type))
        
        xyz, masses, MIN_NUMBER_PTCS = self.getXYZMasses(obj_type)
        if rank != 0:
            del xyz; del masses
        suffix = '_{}_'.format(obj_type)
        
        if rank == 0:
            self.getIdxCat(obj_type)[0]
            self.dumpShapeCatLocalBase(xyz, masses, self.r200.base[select[0]:select[1]+1], self.getIdxCat(obj_type)[0][select[0]:select[1]+1], self.getIdxCat(obj_type)[1][select[0]:select[1]+1], self.D_LOGSTART, self.D_LOGEND, self.D_BINS, self.IT_TOL, self.IT_WALL, self.IT_MIN, CAT_DEST, suffix, reduced, shell_based)
            del xyz; del masses

    def dumpShapeCatGlobal(self, str CAT_DEST, list select, bint reduced = False, str obj_type = 'dm'):
        """ Dumps all relevant global shape data into ``CAT_DEST``
        
        :param CAT_DEST: catalogue folder
        :type CAT_DEST: string
        :param select: index of first and last object to look at in the format [idx_first, idx_last]
        :type select: list containing two integers
        :param reduced: whether or not reduced shape tensor (1/r^2 factor)
        :type reduced: boolean
        :param obj_type: either 'dm' or 'gx', depending on what catalogue we are looking at
        :type obj_type: string"""
        print_status(rank,self.start_time,'Starting dumpShapeCatGlobal() with snap {0} for obj_type {1}'.format(self.SNAP, obj_type))
        
        xyz, masses, MIN_NUMBER_PTCS = self.getXYZMasses(obj_type)
        if rank != 0:
            del xyz; del masses
        suffix = '_{}_'.format(obj_type)
        
        if rank == 0:
            self.getIdxCat(obj_type)[0]
            self.dumpShapeCatGlobalBase(xyz, masses, self.r200.base[select[0]:select[1]+1], self.getIdxCat(obj_type)[0][select[0]:select[1]+1], self.getIdxCat(obj_type)[1][select[0]:select[1]+1], self.IT_TOL, self.IT_WALL, self.IT_MIN, CAT_DEST, suffix, reduced)
            del xyz; del masses

    def dumpShapeVelCatLocal(self, str CAT_DEST, list select, bint reduced = False, bint shell_based = False, str obj_type = 'dm'):
        """ Dumps all relevant local velocity shape data into ``CAT_DEST``
        
        :param CAT_DEST: catalogue folder
        :type CAT_DEST: string
        :param select: index of first and last object to look at in the format [idx_first, idx_last]
        :type select: list containing two integers
        :param reduced: whether or not reduced shape tensor (1/r^2 factor)
        :type reduced: boolean
        :param shell_based: whether shell-based or ellipsoid-based algorithm should be run
        :type shell_based: boolean
        :param obj_type: either 'dm' or 'gx', depending on what catalogue we are looking at
        :type obj_type: string"""
        print_status(rank,self.start_time,'Starting dumpShapeVelCatLocal() with snap {0} for obj_type {1}'.format(self.SNAP, obj_type))
        
        xyz, masses, MIN_NUMBER_PTCS = self.getXYZMasses(obj_type)
        velxyz = self.getVelXYZ(obj_type)
        if rank != 0:
            del xyz; del masses; del velxyz
        suffix = '_v{}_'.format(obj_type)
        
        if rank == 0:
            self.getIdxCat(obj_type)[0]
            self.dumpShapeVelCatLocalBase(xyz, velxyz, masses, self.r200.base[select[0]:select[1]+1], self.getIdxCat(obj_type)[0][select[0]:select[1]+1], self.getIdxCat(obj_type)[1][select[0]:select[1]+1], self.D_LOGSTART, self.D_LOGEND, self.D_BINS, self.IT_TOL, self.IT_WALL, self.IT_MIN, CAT_DEST, suffix, reduced, shell_based)
            del xyz; del velxyz; del masses

    def dumpShapeVelCatGlobal(self, str CAT_DEST, list select, bint reduced = False, str obj_type = 'dm'):
        """ Dumps all relevant global velocity shape data into ``CAT_DEST``
        
        :param CAT_DEST: catalogue folder
        :type CAT_DEST: string
        :param select: index of first and last object to look at in the format [idx_first, idx_last]
        :type select: list containing two integers
        :param reduced: whether or not reduced shape tensor (1/r^2 factor)
        :type reduced: boolean
        :param obj_type: either 'dm' or 'gx', depending on what catalogue we are looking at
        :type obj_type: string"""
        print_status(rank,self.start_time,'Starting dumpShapeVelCatGlobal() with snap {0} for obj_type {1}'.format(self.SNAP, obj_type))
        
        xyz, masses, MIN_NUMBER_PTCS = self.getXYZMasses(obj_type)
        velxyz = self.getVelXYZ(obj_type)
        if rank != 0:
            del xyz; del masses; del velxyz
        suffix = '_v{}_'.format(obj_type)
        
        if rank == 0:
            self.getIdxCat(obj_type)[0]
            self.dumpShapeVelCatGlobalBase(xyz, velxyz, masses, self.r200.base[select[0]:select[1]+1], self.getIdxCat(obj_type)[0][select[0]:select[1]+1], self.getIdxCat(obj_type)[1][select[0]:select[1]+1], self.IT_TOL, self.IT_WALL, self.IT_MIN, CAT_DEST, suffix, reduced)
            del xyz; del velxyz; del masses

    def getObjInfo(self, str obj_type = 'dm'):
        """ Print basic info about the objects used for local shape estimation such as number of converged objects"""
        print_status(rank,self.start_time,'Starting getObjInfoLocal() with snap {0} for obj_type {1}'.format(self.SNAP, obj_type))
        
        xyz, masses, MIN_NUMBER_PTCS = self.getXYZMasses(obj_type)
        if rank != 0:
            del xyz; del masses
        if rank == 0:
            self.getObjInfoBase(xyz, masses, self.getIdxCat(obj_type)[0], obj_type)
            if obj_type == 'dm':
                nb_shs, sh_len, fof_dm_sizes, group_r200, halo_masses = getHDF5SHDMData(self.HDF5_GROUP_DEST, self.WANT_RVIR)
                nb_shs_vec = np.array(nb_shs)
                print_status(rank, self.start_time, "More detailed info on central subhalo catalogue. The total number of halos with > 0 SHs is {0}".format(nb_shs_vec[nb_shs_vec != 0].shape[0]))
                print_status(rank, self.start_time, "The total number of halos is {0}".format(len(nb_shs)))
                print_status(rank, self.start_time, "The total number of SHs (subhalos) is {0}".format(len(sh_len)))
                print_status(rank, self.start_time, "The number of halos that have no SH is {0}".format(nb_shs_vec[nb_shs_vec == 0].shape[0]))
                print_status(rank, self.start_time, "The total number of halos (CSH) that have sufficient resolution is {0}".format(len([x for x in self.getIdxCat(obj_type)[0] if x != []])))
                del nb_shs; del sh_len; del fof_dm_sizes; del group_r200; del halo_masses
            else:
                nb_shs, sh_len_gx, fof_gx_sizes = getHDF5SHGxData(self.HDF5_GROUP_DEST)
                nb_shs_vec = np.array(nb_shs)
                print_status(rank, self.start_time, "More detailed info on galaxy catalogue. The total number of halos with > 0 SHs containing star particles is {0}".format(nb_shs_vec[nb_shs_vec != 0].shape[0]))
                print_status(rank, self.start_time, "The total number of halos is {0}".format(len(nb_shs)))
                print_status(rank, self.start_time, "The total number of SHs (subhalos) containing star particles is {0}".format(len(sh_len_gx)))
                print_status(rank, self.start_time, "The number of halos that have no SH containing star particles is {0}".format(nb_shs_vec[nb_shs_vec == 0].shape[0]))
                print_status(rank, self.start_time, "The number of valid gxs (after discarding low-resolution ones) is {0}.".format(np.array([0 for x in self.getIdxCat(obj_type)[0] if x != []]).shape[0]))
                del nb_shs; del sh_len_gx; del fof_gx_sizes
                
            del xyz; del masses