#cython: language_level=3
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot
from matplotlib.font_manager import FontProperties
import matplotlib
matplotlib.rcParams.update({'font.size': 13})
cimport cython
import inspect
import subprocess
import os
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
from cosmic_profiles.dens_profs.dens_profs_classes cimport DensProfsBase
from cosmic_profiles.common.python_routines import print_status, set_axes_equal, fibonacci_ellipsoid, respectPBCNoRef, isValidSelection, getSubSetIdxCat
from cosmic_profiles.common import config
from cosmic_profiles.shape_profs.shape_profs_tools import getGlobalEpsHist, getLocalEpsHist
from cosmic_profiles.gadget.read_fof import getFoFSHData, getPartType
from cosmic_profiles.gadget import readgadget
from cosmic_profiles.gadget.gen_catalogues import calcObjCat
import time
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
        print_status(rank,self.start_time,'Starting getShapeCatLocal() with snap {0}'.format(self.SNAP))
        if type(obj_numbers) == list:
            obj_numbers = np.int32(obj_numbers)
        if rank == 0:
            nb_objects = len(self.obj_size.base)
            isValidSelection(obj_numbers, nb_objects)
            subset_idx_cat = getSubSetIdxCat(self.idx_cat.base, self.obj_size.base, obj_numbers)
            d, q, s, minor, inter, major, obj_centers, obj_masses = self._getShapeCatLocalBase(self.xyz.base, self.masses.base, self.r200.base[obj_numbers], subset_idx_cat, self.obj_size.base[obj_numbers], self.D_LOGSTART, self.D_LOGEND, self.D_BINS, self.IT_TOL, self.IT_WALL, self.IT_MIN, reduced, shell_based, self.SUFFIX)
            m_curr_over_target = 1.989e43/config.OutUnitMass_in_g
            l_curr_over_target = 3.085678e24/config.OutUnitLength_in_cm
            return d*l_curr_over_target, q, s, minor, inter, major, obj_centers*l_curr_over_target, obj_masses*m_curr_over_target
        else:
            return None, None, None, None, None, None, None, None
    
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
        print_status(rank,self.start_time,'Starting getShapeCatGlobal() with snap {0}'.format(self.SNAP))
        if type(obj_numbers) == list:
            obj_numbers = np.int32(obj_numbers)
        if rank == 0:
            nb_objects = len(self.obj_size.base)
            isValidSelection(obj_numbers, nb_objects)
            subset_idx_cat = getSubSetIdxCat(self.idx_cat.base, self.obj_size.base, obj_numbers)
            d, q, s, minor, inter, major, obj_centers, obj_masses = self._getShapeCatGlobalBase(self.xyz.base, self.masses.base, self.r200.base[obj_numbers], subset_idx_cat, self.obj_size.base[obj_numbers], self.IT_TOL, self.IT_WALL, self.IT_MIN, reduced, self.SUFFIX)
            m_curr_over_target = 1.989e43/config.OutUnitMass_in_g
            l_curr_over_target = 3.085678e24/config.OutUnitLength_in_cm
            return d*l_curr_over_target, q, s, minor, inter, major, obj_centers*l_curr_over_target, obj_masses*m_curr_over_target
        else:
            return None, None, None, None, None, None, None, None
    
    def vizLocalShapes(self, obj_numbers, bint reduced = False, bint shell_based = False): # Public Method
        """ Visualize local shape of objects with numbers ``obj_numbers``
        
        :param obj_numbers: list of object indices for which to visualize local shapes
        :type obj_numbers: list of int
        :param reduced: whether or not reduced shape tensor (1/r^2 factor)
        :type reduced: boolean
        :param shell_based: whether shell-based or ellipsoid-based algorithm should be run
        :type shell_based: boolean"""
        print_status(rank,self.start_time,'Starting vizLocalShapes() with snap {0}'.format(self.SNAP))
        if type(obj_numbers) == list:
            obj_numbers = np.int32(obj_numbers)
        if rank == 0:
            nb_objects = len(self.obj_size.base)
            isValidSelection(obj_numbers, nb_objects)
            subset_idx_cat = getSubSetIdxCat(self.idx_cat.base, self.obj_size.base, obj_numbers)
            offsets = np.int32(np.hstack((np.array([0]), np.cumsum(self.obj_size.base))))
            d, q, s, minor, inter, major, centers, obj_m = self._getShapeCatLocalBase(self.xyz.base, self.masses.base, self.r200.base[obj_numbers], subset_idx_cat, self.obj_size.base[obj_numbers], self.D_LOGSTART, self.D_LOGEND, self.D_BINS, self.IT_TOL, self.IT_WALL, self.IT_MIN, reduced, shell_based, self.SUFFIX)
            del obj_m
                                
            # Create VIZ_DEST if not available
            subprocess.call(['mkdir', '-p', '{}'.format(self.VIZ_DEST)], cwd=os.path.join(currentdir))
            
            # Viz all valid objects under 'obj_numbers'
            for idx_obj, obj_number in enumerate(obj_numbers):
                major_obj = major[idx_obj]
                inter_obj = inter[idx_obj]
                minor_obj = minor[idx_obj]
                d_obj = d[idx_obj]
                q_obj = q[idx_obj]
                s_obj = s[idx_obj]
                center = centers[idx_obj]
                obj = np.zeros((self.obj_size[obj_number],3), dtype = np.float32)
                masses_obj = np.zeros((self.obj_size[obj_number],), dtype = np.float32)
                for idx, ptc in enumerate(self.idx_cat.base[offsets[obj_number]:offsets[obj_number+1]]):
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
                fig.savefig("{}/LocalObj{}{}{}.pdf".format(self.VIZ_DEST, obj_number, self.SUFFIX, self.SNAP), bbox_inches='tight')
        
    def vizGlobalShapes(self, obj_numbers, bint reduced = False): # Public Method
        """ Visualize global shape of objects with numbers ``obj_numbers``
        
        :param obj_numbers: list of object indices for which to visualize global shapes
        :type obj_numbers: list of ints
        :param reduced: whether or not reduced shape tensor (1/r^2 factor)
        :type reduced: boolean"""
        print_status(rank,self.start_time,'Starting vizGlobalShapes() with snap {0}'.format(self.SNAP))
        if type(obj_numbers) == list:
            obj_numbers = np.int32(obj_numbers)
        if rank == 0:
            nb_objects = len(self.obj_size.base)
            isValidSelection(obj_numbers, nb_objects)
            subset_idx_cat = getSubSetIdxCat(self.idx_cat.base, self.obj_size.base, obj_numbers)
            offsets = np.int32(np.hstack((np.array([0]), np.cumsum(self.obj_size.base))))
            d, q, s, minor, inter, major, centers, obj_m = self._getShapeCatGlobalBase(self.xyz.base, self.masses.base, self.r200.base[obj_numbers], subset_idx_cat, self.obj_size.base[obj_numbers], self.IT_TOL, self.IT_WALL, self.IT_MIN, reduced, self.SUFFIX)
            del obj_m
        
            # Create VIZ_DEST if not available
            subprocess.call(['mkdir', '-p', '{}'.format(self.VIZ_DEST)], cwd=os.path.join(currentdir))
            
            # Viz all valid objects under 'obj_numbers'
            for idx_obj, obj_number in enumerate(obj_numbers):
                major_obj = major[idx_obj]
                inter_obj = inter[idx_obj]
                minor_obj = minor[idx_obj]
                d_obj = d[idx_obj]
                q_obj = q[idx_obj]
                s_obj = s[idx_obj]
                center = centers[idx_obj]
                obj = np.zeros((self.obj_size[obj_number],3), dtype = np.float32)
                masses_obj = np.zeros((self.obj_size[obj_number],), dtype = np.float32)
                for idx, ptc in enumerate(self.idx_cat.base[offsets[obj_number]:offsets[obj_number+1]]):
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
                fig.savefig("{}/GlobalObj{}{}{}.pdf".format(self.VIZ_DEST, obj_number, self.SUFFIX, self.SNAP), bbox_inches='tight')
    
    def plotGlobalEpsHist(self, HIST_NB_BINS, obj_numbers): # Public Method
        """ Plot global ellipticity histogram
        
        :param HIST_NB_BINS: number of histogram bins
        :type HIST_NB_BINS: int
        :param obj_numbers: list of object indices of interest
        :type obj_numbers: list of int"""
        print_status(rank,self.start_time,'Starting plotGlobalEpsHist() with snap {0}'.format(self.SNAP))
        if type(obj_numbers) == list:
            obj_numbers = np.int32(obj_numbers)
        if rank == 0:
            nb_objects = len(self.obj_size.base)
            isValidSelection(obj_numbers, nb_objects)
            subset_idx_cat = getSubSetIdxCat(self.idx_cat.base, self.obj_size.base, obj_numbers)
            getGlobalEpsHist(self.xyz.base, self.masses.base, subset_idx_cat, self.obj_size.base[obj_numbers], self.L_BOX, self.CENTER, self.VIZ_DEST, self.SNAP, suffix = self.SUFFIX, HIST_NB_BINS = HIST_NB_BINS)
        
    def plotLocalEpsHist(self, frac_r200, HIST_NB_BINS, obj_numbers): # Public Method
        """ Plot local ellipticity histogram at depth ``frac_r200``
        
        :param frac_r200: depth of objects to plot ellipticity, in units of R200
        :type frac_r200: float
        :param HIST_NB_BINS: number of histogram bins
        :type HIST_NB_BINS: int
        :param obj_numbers: list of object indices of interest
        :type obj_numbers: list of int"""
        print_status(rank,self.start_time,'Starting plotLocalEpsHist() with snap {0}'.format(self.SNAP))
        if type(obj_numbers) == list:
            obj_numbers = np.int32(obj_numbers)
        if rank == 0:
            nb_objects = len(self.obj_size.base)
            isValidSelection(obj_numbers, nb_objects)
            subset_idx_cat = getSubSetIdxCat(self.idx_cat.base, self.obj_size.base, obj_numbers)
            getLocalEpsHist(self.xyz.base, self.masses.base, self.r200.base[obj_numbers], subset_idx_cat, self.obj_size.base[obj_numbers], self.L_BOX, self.CENTER, self.VIZ_DEST, self.SNAP, frac_r200, suffix = self.SUFFIX, HIST_NB_BINS = HIST_NB_BINS)
    
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
        print_status(rank,self.start_time,'Starting plotLocalTHist() with snap {0}'.format(self.SNAP))
        if type(obj_numbers) == list:
            obj_numbers = np.int32(obj_numbers)
        if rank == 0:
            nb_objects = len(self.obj_size.base)
            isValidSelection(obj_numbers, nb_objects)
            subset_idx_cat = getSubSetIdxCat(self.idx_cat.base, self.obj_size.base, obj_numbers)
            self._plotLocalTHistBase(self.xyz.base, self.masses.base, self.r200.base[obj_numbers], subset_idx_cat, self.obj_size.base[obj_numbers], self.D_LOGSTART, self.D_LOGEND, self.D_BINS, self.IT_TOL, self.IT_WALL, self.IT_MIN, HIST_NB_BINS, frac_r200, reduced, shell_based, suffix = self.SUFFIX)
    
    def plotGlobalTHist(self, HIST_NB_BINS, obj_numbers, bint reduced = False): # Public Method
        """ Plot global triaxiality histogram
        
        :param HIST_NB_BINS: number of histogram bins
        :type HIST_NB_BINS: int
        :param obj_numbers: list of object indices of interest
        :type obj_numbers: list of int
        :param reduced: whether or not reduced shape tensor (1/r^2 factor)
        :type reduced: boolean"""
        print_status(rank,self.start_time,'Starting plotGlobalTHist() with snap {0}'.format(self.SNAP))
        if type(obj_numbers) == list:
            obj_numbers = np.int32(obj_numbers)
        if rank == 0:
            nb_objects = len(self.obj_size.base)
            isValidSelection(obj_numbers, nb_objects)
            subset_idx_cat = getSubSetIdxCat(self.idx_cat.base, self.obj_size.base, obj_numbers)
            self._plotGlobalTHistBase(self.xyz.base, self.masses.base, self.r200.base[obj_numbers], subset_idx_cat, self.obj_size.base[obj_numbers], self.IT_TOL, self.IT_WALL, self.IT_MIN, HIST_NB_BINS, reduced, suffix = self.SUFFIX)
    
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
        print_status(rank,self.start_time,'Starting plotShapeProfs() with snap {0}'.format(self.SNAP))
        if type(obj_numbers) == list:
            obj_numbers = np.int32(obj_numbers)
        if rank == 0:
            nb_objects = len(self.obj_size.base)
            isValidSelection(obj_numbers, nb_objects)
            subset_idx_cat = getSubSetIdxCat(self.idx_cat.base, self.obj_size.base, obj_numbers)
            self._plotShapeProfsBase(self.xyz.base, self.masses.base, self.r200.base[obj_numbers], subset_idx_cat, self.obj_size.base[obj_numbers], self.D_LOGSTART, self.D_LOGEND, self.D_BINS, self.IT_TOL, self.IT_WALL, self.IT_MIN, reduced, shell_based, nb_bins, suffix = self.SUFFIX)
    
    def dumpShapeCatLocal(self, obj_numbers, bint reduced = False, bint shell_based = False): # Public Method
        """ Dumps all relevant local shape data into ``CAT_DEST``
        
        :param obj_numbers: list of object indices of interest
        :type obj_numbers: list of int
        :param reduced: whether or not reduced shape tensor (1/r^2 factor)
        :type reduced: boolean
        :param shell_based: whether shell-based or ellipsoid-based algorithm should be run
        :type shell_based: boolean"""
        print_status(rank,self.start_time,'Starting dumpShapeCatLocal() with snap {0}'.format(self.SNAP))
        if type(obj_numbers) == list:
            obj_numbers = np.int32(obj_numbers)
        if rank == 0:
            nb_objects = len(self.obj_size.base)
            isValidSelection(obj_numbers, nb_objects)
            subset_idx_cat = getSubSetIdxCat(self.idx_cat.base, self.obj_size.base, obj_numbers)
            self._dumpShapeCatLocalBase(self.xyz.base, self.masses.base, self.r200.base[obj_numbers], subset_idx_cat, self.obj_size.base[obj_numbers], self.D_LOGSTART, self.D_LOGEND, self.D_BINS, self.IT_TOL, self.IT_WALL, self.IT_MIN, self.SUFFIX, reduced, shell_based)
    
    def dumpShapeCatGlobal(self, obj_numbers, bint reduced = False): # Public Method
        """ Dumps all relevant global shape data into ``CAT_DEST``

        :param obj_numbers: list of object indices of interest
        :type obj_numbers: list of int
        :param reduced: whether or not reduced shape tensor (1/r^2 factor)
        :type reduced: boolean"""
        print_status(rank,self.start_time,'Starting dumpShapeCatGlobal() with snap {0}'.format(self.SNAP))
        if type(obj_numbers) == list:
            obj_numbers = np.int32(obj_numbers)
        if rank == 0:
            nb_objects = len(self.obj_size.base)
            isValidSelection(obj_numbers, nb_objects)
            subset_idx_cat = getSubSetIdxCat(self.idx_cat.base, self.obj_size.base, obj_numbers)
            self._dumpShapeCatGlobalBase(self.xyz.base, self.masses.base, self.r200.base[obj_numbers], subset_idx_cat, self.obj_size.base[obj_numbers], self.IT_TOL, self.IT_WALL, self.IT_MIN, self.SUFFIX, reduced)



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
        print_status(rank,self.start_time,'Starting getShapeCatVelLocal() with snap {0}'.format(self.SNAP))
        xyz, masses = self._getXYZMasses()
        velxyz = self._getVelXYZ()
        idx_cat, obj_size = self.getIdxCat()
        if type(obj_numbers) == list:
            obj_numbers = np.int32(obj_numbers)
        if rank == 0:
            nb_objects = len(obj_size)
            isValidSelection(obj_numbers, nb_objects)
            subset_idx_cat = getSubSetIdxCat(idx_cat, obj_size, obj_numbers)
            suffix = '_v{}_'.format(self.OBJ_TYPE)
            d, q, s, minor, inter, major, obj_centers, obj_masses = self._getShapeCatVelLocalBase(xyz, velxyz, masses, self.r200.base[obj_numbers], subset_idx_cat, obj_size[obj_numbers], self.D_LOGSTART, self.D_LOGEND, self.D_BINS, self.IT_TOL, self.IT_WALL, self.IT_MIN, reduced, shell_based, suffix)
            del xyz; del velxyz; del masses; del idx_cat; del obj_size
            m_curr_over_target = 1.989e43/config.OutUnitMass_in_g
            l_curr_over_target = 3.085678e24/config.OutUnitLength_in_cm
            return d*l_curr_over_target, q, s, minor, inter, major, obj_centers*l_curr_over_target, obj_masses*m_curr_over_target
        else:
            del xyz; del velxyz; del masses; del idx_cat; del obj_size
            return None, None, None, None, None, None, None, None
    
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
        print_status(rank,self.start_time,'Starting getShapeCatVelGlobal() with snap {0}'.format(self.SNAP))
        xyz, masses = self._getXYZMasses()
        velxyz = self._getVelXYZ()
        idx_cat, obj_size = self.getIdxCat()
        if type(obj_numbers) == list:
            obj_numbers = np.int32(obj_numbers)
        if rank == 0:
            nb_objects = len(obj_size)
            isValidSelection(obj_numbers, nb_objects)
            subset_idx_cat = getSubSetIdxCat(idx_cat, obj_size, obj_numbers)
            suffix = '_v{}_'.format(self.OBJ_TYPE)
            d, q, s, minor, inter, major, obj_centers, obj_masses = self._getShapeCatVelGlobalBase(xyz, velxyz, masses, self.r200.base[obj_numbers], subset_idx_cat, obj_size[obj_numbers], self.IT_TOL, self.IT_WALL, self.IT_MIN, self.CENTER, self.SAFE, reduced, suffix)
            del xyz; del velxyz; del masses; del idx_cat; del obj_size
            m_curr_over_target = 1.989e43/config.OutUnitMass_in_g
            l_curr_over_target = 3.085678e24/config.OutUnitLength_in_cm
            return d*l_curr_over_target, q, s, minor, inter, major, obj_centers*l_curr_over_target, obj_masses*m_curr_over_target
        else:
            del xyz; del velxyz; del masses; del idx_cat; del obj_size
            return None, None, None, None, None, None, None, None

    def dumpShapeVelCatLocal(self, obj_numbers, bint reduced = False, bint shell_based = False): # Public Method
        """ Dumps all relevant local velocity shape data into ``CAT_DEST``
        
        :param obj_numbers: list of object indices of interest
        :type obj_numbers: list of int
        :param reduced: whether or not reduced shape tensor (1/r^2 factor)
        :type reduced: boolean
        :param shell_based: whether shell-based or ellipsoid-based algorithm should be run
        :type shell_based: boolean"""
        print_status(rank,self.start_time,'Starting dumpShapeVelCatLocal() with snap {0}'.format(self.SNAP))
        if type(obj_numbers) == list:
            obj_numbers = np.int32(obj_numbers)
        xyz, masses = self._getXYZMasses()
        idx_cat, obj_size = self.getIdxCat()
        if rank == 0:
            idx_cat_len = len(obj_size)
            isValidSelection(obj_numbers, idx_cat_len)
        velxyz = self._getVelXYZ()
        if rank != 0:
            del xyz; del masses; del idx_cat; del obj_size; del velxyz
        suffix = '_v{}_'.format(self.OBJ_TYPE)
        
        if rank == 0:
            subset_idx_cat = getSubSetIdxCat(idx_cat, obj_size, obj_numbers)
            self._dumpShapeVelCatLocalBase(xyz, velxyz, masses, self.r200.base[obj_numbers], subset_idx_cat, obj_size[obj_numbers], self.D_LOGSTART, self.D_LOGEND, self.D_BINS, self.IT_TOL, self.IT_WALL, self.IT_MIN, suffix, reduced, shell_based)
            del xyz; del masses; del idx_cat; del obj_size; del velxyz

    def dumpShapeVelCatGlobal(self, obj_numbers, bint reduced = False): # Public Method
        """ Dumps all relevant global velocity shape data into ``CAT_DEST``
        
        :param obj_numbers: list of object indices of interest
        :type obj_numbers: list of int
        :param reduced: whether or not reduced shape tensor (1/r^2 factor)
        :type reduced: boolean"""
        print_status(rank,self.start_time,'Starting dumpShapeVelCatGlobal() with snap {0}'.format(self.SNAP))
        if type(obj_numbers) == list:
            obj_numbers = np.int32(obj_numbers)
        xyz, masses = self._getXYZMasses()
        idx_cat, obj_size = self.getIdxCat()
        if rank == 0:
            idx_cat_len = len(obj_size)
            isValidSelection(obj_numbers, idx_cat_len)
        velxyz = self._getVelXYZ()
        if rank != 0:
            del xyz; del masses; del idx_cat; del obj_size; del velxyz
        suffix = '_v{}_'.format(self.OBJ_TYPE)
        
        if rank == 0:
            subset_idx_cat = getSubSetIdxCat(idx_cat, obj_size, obj_numbers)
            self._dumpShapeVelCatGlobalBase(xyz, velxyz, masses, self.r200.base[obj_numbers], subset_idx_cat, obj_size[obj_numbers], self.IT_TOL, self.IT_WALL, self.IT_MIN, suffix, reduced)
            del xyz; del masses; del idx_cat; del obj_size; del velxyz
    
    def getXYZMasses(self): # Public Method
        """ Retrieve positions and masses of particles
        
        :return xyz, masses: positions in config.OutUnitLength_in_cm and masses 
            in config.OutUnitMass_in_g
        :rtype: (N2,3) floats, (N2,) floats"""
        xyz = readgadget.read_block(self.SNAP_DEST,"POS ",ptype=[getPartType(self.OBJ_TYPE)])
        masses = readgadget.read_block(self.SNAP_DEST,"MASS",ptype=[getPartType(self.OBJ_TYPE)])
        if rank == 0:
            l_curr_over_target = 3.085678e24/config.OutUnitLength_in_cm
            m_curr_over_target = 1.989e43/config.OutUnitMass_in_g
            return xyz*l_curr_over_target, masses*m_curr_over_target
        else:
            del xyz; del masses
            return None, None
        
    def _getXYZMasses(self):
        """ Retrieve positions and masses of particles
        
        :return xyz, masses: positions in Mpc/h and masses in 10^10*M_sun*h^2/(Mpc)**3
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
            v_curr_over_target = 1e5/config.OutUnitVelocity_in_cm_per_s
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
            l_curr_over_target = 3.085678e24/config.OutUnitLength_in_cm
            return obj_r200*l_curr_over_target
        else:
            del nb_shs; del sh_len; del fof_sizes; del group_r200
            return None
        
    def getIdxCat(self): # Public Method
        """ Fetch catalogue
        
        :return idx_cat: each row contains indices of particles belonging to an object,
            obj_size: number of particles in each object
        :rtype: (N1, N3) integers and (N1,) integers"""
        
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
        
    def getObjInfoHDF5(self): # Public Method
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