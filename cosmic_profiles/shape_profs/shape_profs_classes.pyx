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
from cosmic_profiles.common.python_routines import print_status, set_axes_equal, fibonacci_ellipsoid, respectPBCNoRef, isValidSelection, getSubSetIdxCat, checkKatzConfig, default_katz_config
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
    
    cdef double IT_TOL
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
        print_status(rank,self.start_time,'Starting getShapeCatLocal() with snap {0}'.format(self.SNAP))
        ROverR200, IT_TOL, IT_WALL, IT_MIN, REDUCED, SHELL_BASED = checkKatzConfig(katz_config)
        if type(obj_numbers) == list:
            obj_numbers = np.int32(obj_numbers)
        if rank == 0:
            nb_objects = len(self.obj_size.base)
            isValidSelection(obj_numbers, nb_objects)
            subset_idx_cat = getSubSetIdxCat(self.idx_cat.base, self.obj_size.base, obj_numbers)
            d, q, s, minor, inter, major, obj_centers, obj_masses = self._getShapeCatLocalBase(self.xyz.base, self.masses.base, self.r200.base[obj_numbers], subset_idx_cat, self.obj_size.base[obj_numbers], ROverR200, IT_TOL, IT_WALL, IT_MIN, REDUCED, SHELL_BASED, self.SUFFIX)
            l_internal, m_internal, vel_internal = config.getLMVInternal()
            m_curr_over_target = m_internal/config.OutUnitMass_in_g
            l_curr_over_target = l_internal/config.OutUnitLength_in_cm
            SHAPE_PROF_DTYPE = [("d", "f8"), ("s", "f8"), ("q", "f8"), ("is_conv", "bool"), ("minor", "f8", (3,)), ("inter", "f8", (3,)), ("major", "f8", (3,))]
            shapes = np.zeros((len(obj_numbers), ROverR200.shape[0]), dtype=SHAPE_PROF_DTYPE)
            shapes["d"] = d*l_curr_over_target
            shapes["q"] = q
            shapes["s"] = s
            shapes["minor"] = minor
            shapes["inter"] = inter
            shapes["major"] = major
            shapes["is_conv"] = ~np.isnan(q)
            return shapes
        else:
            return None
    
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
        print_status(rank,self.start_time,'Starting getShapeCatGlobal() with snap {0}'.format(self.SNAP))
        if type(obj_numbers) == list:
            obj_numbers = np.int32(obj_numbers)
        ROverR200, IT_TOL, IT_WALL, IT_MIN, REDUCED, SHELL_BASED = checkKatzConfig(katz_config)
        if rank == 0:
            nb_objects = len(self.obj_size.base)
            isValidSelection(obj_numbers, nb_objects)
            subset_idx_cat = getSubSetIdxCat(self.idx_cat.base, self.obj_size.base, obj_numbers)
            d, q, s, minor, inter, major, obj_centers, obj_masses = self._getShapeCatGlobalBase(self.xyz.base, self.masses.base, self.r200.base[obj_numbers], subset_idx_cat, self.obj_size.base[obj_numbers], IT_TOL, IT_WALL, IT_MIN, REDUCED, self.SUFFIX)
            l_internal, m_internal, vel_internal = config.getLMVInternal()
            m_curr_over_target = m_internal/config.OutUnitMass_in_g
            l_curr_over_target = l_internal/config.OutUnitLength_in_cm
            SHAPE_PROF_DTYPE = [("d", "f8"), ("s", "f8"), ("q", "f8"), ("is_conv", "bool"), ("minor", "f8", (3,)), ("inter", "f8", (3,)), ("major", "f8", (3,))]
            shapes = np.zeros((len(obj_numbers),), dtype=SHAPE_PROF_DTYPE)
            shapes["d"] = d[:,0]*l_curr_over_target
            shapes["q"] = q[:,0]
            shapes["s"] = s[:,0]
            shapes["minor"] = minor[:,0,:]
            shapes["inter"] = inter[:,0,:]
            shapes["major"] = major[:,0,:]
            shapes["is_conv"] = ~np.isnan(q[:,0])
            return shapes
        else:
            return None
    
    def vizLocalShapes(self, obj_numbers, katz_config = default_katz_config): # Public Method
        """ Visualize local shape of objects with numbers ``obj_numbers``
        
        :param obj_numbers: list of object indices of interest
        :type obj_numbers: list of int
        :param katz_config: dictionary with parameters to the Katz algorithm, with fields 'ROverR200', 'IT_TOL', 'IT_WALL', 'IT_MIN', 'REDUCED', 'SHELL_BASED'
        :type katz_config: dictionary"""
        print_status(rank,self.start_time,'Starting vizLocalShapes() with snap {0}'.format(self.SNAP))
        if type(obj_numbers) == list:
            obj_numbers = np.int32(obj_numbers)
        ROverR200, IT_TOL, IT_WALL, IT_MIN, REDUCED, SHELL_BASED = checkKatzConfig(katz_config)
        D_BINS = ROverR200.shape[0]-1
        if rank == 0:
            nb_objects = len(self.obj_size.base)
            isValidSelection(obj_numbers, nb_objects)
            subset_idx_cat = getSubSetIdxCat(self.idx_cat.base, self.obj_size.base, obj_numbers)
            offsets = np.int32(np.hstack((np.array([0]), np.cumsum(self.obj_size.base))))
            d, q, s, minor, inter, major, centers, obj_m = self._getShapeCatLocalBase(self.xyz.base, self.masses.base, self.r200.base[obj_numbers], subset_idx_cat, self.obj_size.base[obj_numbers], ROverR200, IT_TOL, IT_WALL, IT_MIN, REDUCED, SHELL_BASED, self.SUFFIX)
            del obj_m
                                
            # Create VIZ_DEST if not available
            subprocess.call(['mkdir', '-p', '{}'.format(self.VIZ_DEST)], cwd=os.path.join(currentdir))
            
            # Go to outgoing unit system
            d = config.LengthInternalToOut(d)
            centers = config.LengthInternalToOut(centers)
            l_label, m_label, vel_lable = config.LMVLabel()
            
            # Viz all valid objects under 'obj_numbers'
            for idx_obj, obj_number in enumerate(obj_numbers):
                major_obj = major[idx_obj]
                inter_obj = inter[idx_obj]
                minor_obj = minor[idx_obj]
                d_obj = d[idx_obj]
                q_obj = q[idx_obj]
                s_obj = s[idx_obj]
                center = centers[idx_obj]
                obj = np.zeros((self.obj_size[obj_number],3), dtype = np.float64)
                masses_obj = np.zeros((self.obj_size[obj_number],), dtype = np.float64)
                for idx, ptc in enumerate(self.idx_cat.base[offsets[obj_number]:offsets[obj_number+1]]):
                    obj[idx] = self.xyz.base[ptc]
                    masses_obj[idx] = self.masses.base[ptc]
                obj = config.LengthInternalToOut(respectPBCNoRef(obj, self.L_BOX))
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
                for idx in np.arange(D_BINS-D_BINS//5, D_BINS):
                    if idx == D_BINS-1:
                        ax.quiver(*center, major_obj[idx][0], major_obj[idx][1], major_obj[idx][2], length=d_obj[idx], color='m', label= "Major")
                        ax.quiver(*center, inter_obj[idx][0], inter_obj[idx][1], inter_obj[idx][2], length=q_obj[idx]*d_obj[idx], color='c', label = "Intermediate")
                        ax.quiver(*center, minor_obj[idx][0], minor_obj[idx][1], minor_obj[idx][2], length=s_obj[idx]*d_obj[idx], color='y', label = "Minor")
                    else:
                        ax.quiver(*center, major_obj[idx][0], major_obj[idx][1], major_obj[idx][2], length=d_obj[idx], color='m')
                        ax.quiver(*center, inter_obj[idx][0], inter_obj[idx][1], inter_obj[idx][2], length=q_obj[idx]*d_obj[idx], color='c')
                        ax.quiver(*center, minor_obj[idx][0], minor_obj[idx][1], minor_obj[idx][2], length=s_obj[idx]*d_obj[idx], color='y')
                for special in np.arange(-D_BINS//5,-D_BINS//5+1):
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
                plt.xlabel(r"x  [{}]".format(l_label))
                plt.ylabel(r"y  [{}]".format(l_label))
                ax.set_zlabel(r"z  [{}]".format(l_label))
                ax.set_box_aspect([1,1,1])
                set_axes_equal(ax)
                fig.savefig("{}/LocalObj{}{}{}.pdf".format(self.VIZ_DEST, obj_number, self.SUFFIX, self.SNAP), bbox_inches='tight')
        
    def vizGlobalShapes(self, obj_numbers, katz_config = default_katz_config): # Public Method
        """ Visualize global shape of objects with numbers ``obj_numbers``
        
        :param obj_numbers: list of object indices of interest
        :type obj_numbers: list of int
        :param katz_config: dictionary with parameters to the Katz algorithm, with fields 'ROverR200', 'IT_TOL', 'IT_WALL', 'IT_MIN', 'REDUCED', 'SHELL_BASED'
        :type katz_config: dictionary"""
        print_status(rank,self.start_time,'Starting vizGlobalShapes() with snap {0}'.format(self.SNAP))
        if type(obj_numbers) == list:
            obj_numbers = np.int32(obj_numbers)
        ROverR200, IT_TOL, IT_WALL, IT_MIN, REDUCED, SHELL_BASED = checkKatzConfig(katz_config)
        if rank == 0:
            nb_objects = len(self.obj_size.base)
            isValidSelection(obj_numbers, nb_objects)
            subset_idx_cat = getSubSetIdxCat(self.idx_cat.base, self.obj_size.base, obj_numbers)
            offsets = np.int32(np.hstack((np.array([0]), np.cumsum(self.obj_size.base))))
            d, q, s, minor, inter, major, centers, obj_m = self._getShapeCatGlobalBase(self.xyz.base, self.masses.base, self.r200.base[obj_numbers], subset_idx_cat, self.obj_size.base[obj_numbers], IT_TOL, IT_WALL, IT_MIN, REDUCED, self.SUFFIX)
            del obj_m
        
            # Create VIZ_DEST if not available
            subprocess.call(['mkdir', '-p', '{}'.format(self.VIZ_DEST)], cwd=os.path.join(currentdir))
            
            # Go to outgoing unit system
            d = config.LengthInternalToOut(d)
            centers = config.LengthInternalToOut(centers)
            l_label, m_label, vel_lable = config.LMVLabel()
            
            # Viz all valid objects under 'obj_numbers'
            for idx_obj, obj_number in enumerate(obj_numbers):
                major_obj = major[idx_obj]
                inter_obj = inter[idx_obj]
                minor_obj = minor[idx_obj]
                d_obj = d[idx_obj]
                q_obj = q[idx_obj]
                s_obj = s[idx_obj]
                center = centers[idx_obj]
                obj = np.zeros((self.obj_size[obj_number],3), dtype = np.float64)
                masses_obj = np.zeros((self.obj_size[obj_number],), dtype = np.float64)
                for idx, ptc in enumerate(self.idx_cat.base[offsets[obj_number]:offsets[obj_number+1]]):
                    obj[idx] = self.xyz.base[ptc]
                    masses_obj[idx] = self.masses.base[ptc]
                obj = config.LengthInternalToOut(respectPBCNoRef(obj, self.L_BOX))
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
                plt.xlabel(r"x  [{}]".format(l_label))
                plt.ylabel(r"y  [{}]".format(l_label))
                ax.set_zlabel(r"z  [{}]".format(l_label))
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
        print_status(rank,self.start_time,'Starting plotLocalTHist() with snap {0}'.format(self.SNAP))
        if type(obj_numbers) == list:
            obj_numbers = np.int32(obj_numbers)
        ROverR200, IT_TOL, IT_WALL, IT_MIN, REDUCED, SHELL_BASED = checkKatzConfig(katz_config)
        if rank == 0:
            nb_objects = len(self.obj_size.base)
            isValidSelection(obj_numbers, nb_objects)
            subset_idx_cat = getSubSetIdxCat(self.idx_cat.base, self.obj_size.base, obj_numbers)
            self._plotLocalTHistBase(self.xyz.base, self.masses.base, self.r200.base[obj_numbers], subset_idx_cat, self.obj_size.base[obj_numbers], ROverR200, IT_TOL, IT_WALL, IT_MIN, HIST_NB_BINS, frac_r200, REDUCED, SHELL_BASED, suffix = self.SUFFIX)
    
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
        print_status(rank,self.start_time,'Starting plotGlobalTHist() with snap {0}'.format(self.SNAP))
        if type(obj_numbers) == list:
            obj_numbers = np.int32(obj_numbers)
        ROverR200, IT_TOL, IT_WALL, IT_MIN, REDUCED, SHELL_BASED = checkKatzConfig(katz_config)
        if rank == 0:
            nb_objects = len(self.obj_size.base)
            isValidSelection(obj_numbers, nb_objects)
            subset_idx_cat = getSubSetIdxCat(self.idx_cat.base, self.obj_size.base, obj_numbers)
            self._plotGlobalTHistBase(self.xyz.base, self.masses.base, self.r200.base[obj_numbers], subset_idx_cat, self.obj_size.base[obj_numbers], IT_TOL, IT_WALL, IT_MIN, HIST_NB_BINS, REDUCED, suffix = self.SUFFIX)
    
    def plotShapeProfs(self, int nb_bins, obj_numbers, katz_config = default_katz_config): # Public Method
        """ Draws shape profiles, also mass bin-decomposed ones
        
        :param nb_bins: Number of mass bins to plot density profiles for
        :type nb_bins: int
        :param obj_numbers: list of object indices of interest
        :type obj_numbers: list of int
        :param katz_config: dictionary with parameters to the Katz algorithm, with fields 'ROverR200', 'IT_TOL', 'IT_WALL', 'IT_MIN', 'REDUCED', 'SHELL_BASED'
        :type katz_config: dictionary"""
        print_status(rank,self.start_time,'Starting plotShapeProfs() with snap {0}'.format(self.SNAP))
        if type(obj_numbers) == list:
            obj_numbers = np.int32(obj_numbers)
        ROverR200, IT_TOL, IT_WALL, IT_MIN, REDUCED, SHELL_BASED = checkKatzConfig(katz_config)
        if rank == 0:
            nb_objects = len(self.obj_size.base)
            isValidSelection(obj_numbers, nb_objects)
            subset_idx_cat = getSubSetIdxCat(self.idx_cat.base, self.obj_size.base, obj_numbers)
            self._plotShapeProfsBase(self.xyz.base, self.masses.base, self.r200.base[obj_numbers], subset_idx_cat, self.obj_size.base[obj_numbers], ROverR200, IT_TOL, IT_WALL, IT_MIN, REDUCED, SHELL_BASED, nb_bins, suffix = self.SUFFIX)
    
    def dumpShapeCatLocal(self, obj_numbers, katz_config = default_katz_config): # Public Method
        """ Dumps all relevant local shape data into ``CAT_DEST``
        
        :param obj_numbers: list of object indices of interest
        :type obj_numbers: list of int
        :param katz_config: dictionary with parameters to the Katz algorithm, with fields 'ROverR200', 'IT_TOL', 'IT_WALL', 'IT_MIN', 'REDUCED', 'SHELL_BASED'
        :type katz_config: dictionary"""
        print_status(rank,self.start_time,'Starting dumpShapeCatLocal() with snap {0}'.format(self.SNAP))
        if type(obj_numbers) == list:
            obj_numbers = np.int32(obj_numbers)
        ROverR200, IT_TOL, IT_WALL, IT_MIN, REDUCED, SHELL_BASED = checkKatzConfig(katz_config)
        if rank == 0:
            nb_objects = len(self.obj_size.base)
            isValidSelection(obj_numbers, nb_objects)
            subset_idx_cat = getSubSetIdxCat(self.idx_cat.base, self.obj_size.base, obj_numbers)
            self._dumpShapeCatLocalBase(self.xyz.base, self.masses.base, self.r200.base[obj_numbers], subset_idx_cat, self.obj_size.base[obj_numbers], ROverR200, IT_TOL, IT_WALL, IT_MIN, self.SUFFIX, REDUCED, SHELL_BASED)
    
    def dumpShapeCatGlobal(self, obj_numbers, katz_config = default_katz_config): # Public Method
        """ Dumps all relevant global shape data into ``CAT_DEST``

        :param obj_numbers: list of object indices of interest
        :type obj_numbers: list of int
        :param katz_config: dictionary with parameters to the Katz algorithm, with fields 'ROverR200', 'IT_TOL', 'IT_WALL', 'IT_MIN', 'REDUCED', 'SHELL_BASED'
        :type katz_config: dictionary"""
        print_status(rank,self.start_time,'Starting dumpShapeCatGlobal() with snap {0}'.format(self.SNAP))
        if type(obj_numbers) == list:
            obj_numbers = np.int32(obj_numbers)
        ROverR200, IT_TOL, IT_WALL, IT_MIN, REDUCED, SHELL_BASED = checkKatzConfig(katz_config)
        if rank == 0:
            nb_objects = len(self.obj_size.base)
            isValidSelection(obj_numbers, nb_objects)
            subset_idx_cat = getSubSetIdxCat(self.idx_cat.base, self.obj_size.base, obj_numbers)
            self._dumpShapeCatGlobalBase(self.xyz.base, self.masses.base, self.r200.base[obj_numbers], subset_idx_cat, self.obj_size.base[obj_numbers], IT_TOL, IT_WALL, IT_MIN, self.SUFFIX, REDUCED)



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
        print_status(rank,self.start_time,'Starting getShapeCatVelLocal() with snap {0}'.format(self.SNAP))
        xyz, masses = self._getXYZMasses()
        velxyz = self._getVelXYZ()
        idx_cat, obj_size = self.getIdxCat()
        if type(obj_numbers) == list:
            obj_numbers = np.int32(obj_numbers)
        ROverR200, IT_TOL, IT_WALL, IT_MIN, REDUCED, SHELL_BASED = checkKatzConfig(katz_config)
        if rank == 0:
            nb_objects = len(obj_size)
            isValidSelection(obj_numbers, nb_objects)
            subset_idx_cat = getSubSetIdxCat(idx_cat, obj_size, obj_numbers)
            suffix = '_v{}_'.format(self.OBJ_TYPE)
            d, q, s, minor, inter, major, obj_centers, obj_masses = self._getShapeCatVelLocalBase(xyz, velxyz, masses, self.r200.base[obj_numbers], subset_idx_cat, obj_size[obj_numbers], ROverR200, IT_TOL, IT_WALL, IT_MIN, REDUCED, SHELL_BASED, suffix)
            del xyz; del velxyz; del masses; del idx_cat; del obj_size
            l_internal, m_internal, vel_internal = config.getLMVInternal()
            m_curr_over_target = m_internal/config.OutUnitMass_in_g
            l_curr_over_target = l_internal/config.OutUnitLength_in_cm
            SHAPE_PROF_DTYPE = [("d", "f8"), ("s", "f8"), ("q", "f8"), ("is_conv", "bool"), ("minor", "f8", (3,)), ("inter", "f8", (3,)), ("major", "f8", (3,))]
            shapes = np.zeros((len(obj_numbers), ROverR200.shape[0]), dtype=SHAPE_PROF_DTYPE)
            shapes["d"] = d*l_curr_over_target
            shapes["q"] = q
            shapes["s"] = s
            shapes["minor"] = minor
            shapes["inter"] = inter
            shapes["major"] = major
            shapes["is_conv"] = ~np.isnan(q)
            return shapes
        else:
            del xyz; del velxyz; del masses; del idx_cat; del obj_size
            return None
    
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
        print_status(rank,self.start_time,'Starting getShapeCatVelGlobal() with snap {0}'.format(self.SNAP))
        xyz, masses = self._getXYZMasses()
        velxyz = self._getVelXYZ()
        idx_cat, obj_size = self.getIdxCat()
        if type(obj_numbers) == list:
            obj_numbers = np.int32(obj_numbers)
        ROverR200, IT_TOL, IT_WALL, IT_MIN, REDUCED, SHELL_BASED = checkKatzConfig(katz_config)
        if rank == 0:
            nb_objects = len(obj_size)
            isValidSelection(obj_numbers, nb_objects)
            subset_idx_cat = getSubSetIdxCat(idx_cat, obj_size, obj_numbers)
            suffix = '_v{}_'.format(self.OBJ_TYPE)
            d, q, s, minor, inter, major, obj_centers, obj_masses = self._getShapeCatVelGlobalBase(xyz, velxyz, masses, self.r200.base[obj_numbers], subset_idx_cat, obj_size[obj_numbers], IT_TOL, IT_WALL, IT_MIN, self.CENTER, self.SAFE, REDUCED, suffix)
            del xyz; del velxyz; del masses; del idx_cat; del obj_size
            l_internal, m_internal, vel_internal = config.getLMVInternal()
            m_curr_over_target = m_internal/config.OutUnitMass_in_g
            l_curr_over_target = l_internal/config.OutUnitLength_in_cm
            SHAPE_PROF_DTYPE = [("d", "f8"), ("s", "f8"), ("q", "f8"), ("is_conv", "bool"), ("minor", "f8", (3,)), ("inter", "f8", (3,)), ("major", "f8", (3,))]
            shapes = np.zeros((len(obj_numbers),), dtype=SHAPE_PROF_DTYPE)
            shapes["d"] = d[:,0]*l_curr_over_target
            shapes["q"] = q[:,0]
            shapes["s"] = s[:,0]
            shapes["minor"] = minor[:,0,:]
            shapes["inter"] = inter[:,0,:]
            shapes["major"] = major[:,0,:]
            shapes["is_conv"] = ~np.isnan(q[:,0])
            return shapes
        else:
            del xyz; del velxyz; del masses; del idx_cat; del obj_size
            return None

    def dumpShapeVelCatLocal(self, obj_numbers, katz_config = default_katz_config): # Public Method
        """ Dumps all relevant local velocity shape data into ``CAT_DEST``
        
        :param obj_numbers: list of object indices of interest
        :type obj_numbers: list of int
        :param katz_config: dictionary with parameters to the Katz algorithm, with fields 'ROverR200', 'IT_TOL', 'IT_WALL', 'IT_MIN', 'REDUCED', 'SHELL_BASED'
        :type katz_config: dictionary"""
        print_status(rank,self.start_time,'Starting dumpShapeVelCatLocal() with snap {0}'.format(self.SNAP))
        if type(obj_numbers) == list:
            obj_numbers = np.int32(obj_numbers)
        ROverR200, IT_TOL, IT_WALL, IT_MIN, REDUCED, SHELL_BASED = checkKatzConfig(katz_config)
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
            self._dumpShapeVelCatLocalBase(xyz, velxyz, masses, self.r200.base[obj_numbers], subset_idx_cat, obj_size[obj_numbers], ROverR200, IT_TOL, IT_WALL, IT_MIN, suffix, REDUCED, SHELL_BASED)
            del xyz; del masses; del idx_cat; del obj_size; del velxyz

    def dumpShapeVelCatGlobal(self, obj_numbers, katz_config = default_katz_config): # Public Method
        """ Dumps all relevant global velocity shape data into ``CAT_DEST``
        
        :param obj_numbers: list of object indices of interest
        :type obj_numbers: list of int
        :param katz_config: dictionary with parameters to the Katz algorithm, with fields 'ROverR200', 'IT_TOL', 'IT_WALL', 'IT_MIN', 'REDUCED', 'SHELL_BASED'
        :type katz_config: dictionary"""
        print_status(rank,self.start_time,'Starting dumpShapeVelCatGlobal() with snap {0}'.format(self.SNAP))
        if type(obj_numbers) == list:
            obj_numbers = np.int32(obj_numbers)
        ROverR200, IT_TOL, IT_WALL, IT_MIN, REDUCED, SHELL_BASED = checkKatzConfig(katz_config)
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
            self._dumpShapeVelCatGlobalBase(xyz, velxyz, masses, self.r200.base[obj_numbers], subset_idx_cat, obj_size[obj_numbers], IT_TOL, IT_WALL, IT_MIN, suffix, REDUCED)
            del xyz; del masses; del idx_cat; del obj_size; del velxyz
    
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