#!/usr/bin/env python3
# -*- coding: utf-8 -*-
    
import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def getAdmissibleLMV():
    global LittleH
    length_in_cm_d = {'kpc': 3.085678e21, 'Mpc': 3.085678e24, 'Gpc': 3.085678e27, 'kpc/h': 3.085678e21/LittleH, 'Mpc/h': 3.085678e24/LittleH, 'Gpc/h': 3.085678e27/LittleH, 'pc': 3.085678e18, 'pc/h': 3.085678e18/LittleH}
    mass_in_g_d = {'Msun': 1.989e33, 'E+10Msun': 1.989e43, 'Msun/h': 1.989e33/LittleH, 'E+10Msun/h': 1.989e43/LittleH}
    velocity_in_cm_per_s_d = {'km/s': 1e5}
    return length_in_cm_d, mass_in_g_d, velocity_in_cm_per_s_d

def getLMVInternal():
    length_in_cm_d, mass_in_g_d, velocity_in_cm_per_s_d = getAdmissibleLMV()
    return length_in_cm_d['Mpc/h'], mass_in_g_d['E+10Msun/h'], velocity_in_cm_per_s_d['km/s']

def initialize(gbs, maxsize, in_length_in_cm, in_mass_in_g, in_velocity_in_cm_per_s, little_h, out_length_in_cm, out_mass_in_g, out_velocity_in_cm_per_s):
    """ Initialize all global variables, should only be called once.
    
    :param gbs: new memory limit in GBs
    :type gbs: int, or float
    :param maxsize: new maximum cache size (note that multiple functions are decorated, each will have
        new maximum cache size)
    :type maxsize: int
    :param in_length_in_cm: unit for length in cm as used in provided data
    :type in_length_in_cm: float
    :param in_mass_in_g: unit for mass in g as used in provided data
    :type in_mass_in_g: float
    :param in_velocity_in_cm_per_s: unit for velocity in cm/s as used in provided data
    :type in_velocity_in_cm_per_s: float
    :param little_h: scaled Hubble constant
    :type little_h: float
    :param out_length_in_cm: unit for length in cm to be used by CosmicProfiles
    :type out_length_in_cm: float
    :param out_mass_in_g: unit for mass in g to be used by CosmicProfiles
    :type out_mass_in_g: float
    :param out_velocity_in_cm_per_s: unit for velocity in cm/s to be used by CosmicProfiles
    :type out_velocity_in_cm_per_s: float"""
    
    global GBs # Number of gigabytes. If only this much left (according to psutil.virtual_memory().available), cache will be considered full.
               # I.e. if this many GBs of memory available, stop caching to avoid thrashing.

    global CACHE_MAXSIZE # Defines the maximum number of entries before the cache starts evicting old items
                         # Note that since the np_cache_factory decorator is applied to many functions,
                         # CACHE_MAXSIZE refers to the maximum cache size of each function separately.
                         # If CACHE_MAXSIZE == None, then the cache will grow indefinitely, and no entries will be ever evicted.
                         # Unless GBs is 0, CACHE_MAXSIZE has no effect.
                         
    global InUnitLength_in_cm
    global InUnitMass_in_g
    global InUnitVelocity_in_cm_per_s
    global LittleH
    global OutUnitLength_in_cm
    global OutUnitMass_in_g
    global OutUnitVelocity_in_cm_per_s
    
    GBs = gbs
    CACHE_MAXSIZE = maxsize
    InUnitLength_in_cm = in_length_in_cm
    InUnitMass_in_g = in_mass_in_g
    InUnitVelocity_in_cm_per_s = in_velocity_in_cm_per_s
    LittleH = little_h
    OutUnitLength_in_cm = out_length_in_cm
    OutUnitMass_in_g = out_mass_in_g
    OutUnitVelocity_in_cm_per_s = out_velocity_in_cm_per_s
    
def updateCachingMaxGBs(gbs):
    """ Update ``use_memory_up_to`` argument of LRU-caching decorator

    :param gbs: new memory limit in GBs
    :type gbs: int, or float""" 
    global GBs
    GBs = gbs

def updateCachingMaxSize(maxsize):
    """ Update ``maxsize`` argument of LRU-caching decorator

    :param maxsize: new maximum cache size (note that multiple functions are decorated, each will have
        new maximum cache size)
    :type maxsize: int""" 
    global CACHE_MAXSIZE
    CACHE_MAXSIZE = maxsize
    
def updateInUnitSystem(length_in_cm, mass_in_g, velocity_in_cm_per_s, little_h):
    """ Inform CosmicProfiles about system of units employed in snapshot data
    
    :param length_in_cm: unit for length in cm as used in provided data
    :type length_in_cm: float
    :param mass_in_g: unit for mass in g as used in provided data
    :type mass_in_g: float
    :param velocity_in_cm_per_s: unit for velocity in cm/s as used in provided data
    :type velocity_in_cm_per_s: float
    :param little_h: scaled Hubble constant
    :type little_h: float"""
    
    global InUnitLength_in_cm
    global InUnitMass_in_g
    global InUnitVelocity_in_cm_per_s
    global LittleH
    LittleH = little_h
    
    length_in_cm_d, mass_in_g_d, velocity_in_cm_per_s_d = getAdmissibleLMV()
    if type(length_in_cm) == str:
        assert length_in_cm in length_in_cm_d.keys(), "Unit not allowed for length_in_cm. Choose from {} or provide length in cm".format(list(length_in_cm_d.keys()))
        length_in_cm = length_in_cm_d[length_in_cm]
    if type(mass_in_g) == str:
        assert mass_in_g in mass_in_g_d.keys(), "Unit not allowed for mass_in_g. Choose from {} or provide mass in g".format(list(mass_in_g_d.keys()))
        mass_in_g = mass_in_g_d[mass_in_g]
    if type(velocity_in_cm_per_s) == str:
        assert velocity_in_cm_per_s in velocity_in_cm_per_s_d.keys(), "Unit not allowed for velocity_in_cm_per_s. Choose from {} or provide vel in cm/s".format(list(velocity_in_cm_per_s_d.keys()))
        velocity_in_cm_per_s = velocity_in_cm_per_s_d[velocity_in_cm_per_s]
    InUnitLength_in_cm = length_in_cm
    InUnitMass_in_g = mass_in_g
    InUnitVelocity_in_cm_per_s = velocity_in_cm_per_s
    
def updateOutUnitSystem(length_in_cm, mass_in_g, velocity_in_cm_per_s, little_h):
    """ Update unit system to be used by CosmicProfiles in its outputs
    
    :param length_in_cm: unit for length in cm to be used by CosmicProfiles
    :type length_in_cm: float
    :param mass_in_g: unit for mass in g to be used by CosmicProfiles
    :type mass_in_g: float
    :param velocity_in_cm_per_s: unit for velocity in cm/s to be used by CosmicProfiles
    :type velocity_in_cm_per_s: float
    :param little_h: scaled Hubble constant
    :type little_h: float"""
    
    global OutUnitLength_in_cm
    global OutUnitMass_in_g
    global OutUnitVelocity_in_cm_per_s
    global LittleH
    LittleH = little_h
    
    length_in_cm_d, mass_in_g_d, velocity_in_cm_per_s_d = getAdmissibleLMV()
    if type(length_in_cm) == str:
        assert length_in_cm in length_in_cm_d.keys(), "Unit not allowed for length_in_cm. Choose from {} or provide length in cm".format(list(length_in_cm_d.keys()))
        length_in_cm = length_in_cm_d[length_in_cm]
    if type(mass_in_g) == str:
        assert mass_in_g in mass_in_g_d.keys(), "Unit not allowed for mass_in_g. Choose from {} or provide mass in g".format(list(mass_in_g_d.keys()))
        mass_in_g = mass_in_g_d[mass_in_g]
    if type(velocity_in_cm_per_s) == str:
        assert velocity_in_cm_per_s in velocity_in_cm_per_s_d.keys(), "Unit not allowed for velocity_in_cm_per_s. Choose from {} or provide vel in cm/s".format(list(velocity_in_cm_per_s_d.keys()))
        velocity_in_cm_per_s = velocity_in_cm_per_s_d[velocity_in_cm_per_s]
    OutUnitLength_in_cm = length_in_cm
    OutUnitMass_in_g = mass_in_g
    OutUnitVelocity_in_cm_per_s = velocity_in_cm_per_s
    
def LengthInternalToOut(length_arr):
    """ Rescale length_arr from internal length units (Mpc/h) to out length units
    
    :param length_arr: length array in internal units
    :type length_arr: floats"""
    global OutUnitLength_in_cm
    length_in_cm_d, mass_in_g_d, velocity_in_cm_per_s_d = getAdmissibleLMV()
    int_to_out = np.float32(length_in_cm_d['Mpc/h']/OutUnitLength_in_cm)
    return length_arr*int_to_out
    
def MassInternalToOut(mass_arr):
    """ Rescale mass_arr from internal mass units (E+10M_sun/h) to out mass units
    
    :param mass_arr: mass array in internal units
    :type mass_arr: floats"""
    global OutUnitMass_in_g
    length_in_cm_d, mass_in_g_d, velocity_in_cm_per_s_d = getAdmissibleLMV()
    int_to_out = np.float32(mass_in_g_d['E+10M_sun/h']/OutUnitMass_in_g)
    return mass_arr*int_to_out

def VelInternalToOut(vel_arr):
    """ Rescale vel_arr from internal velocity units (km/s) to out velocity units
    
    :param vel_arr: velocity array in internal units
    :type vel_arr: floats"""
    global OutUnitVelocity_in_cm_per_s
    length_in_cm_d, mass_in_g_d, velocity_in_cm_per_s_d = getAdmissibleLMV()
    int_to_out = np.float32(velocity_in_cm_per_s_d['km/s']/OutUnitVelocity_in_cm_per_s)
    return vel_arr*int_to_out

def LengthInToInternal(length_arr):
    """ Rescale length_arr from in length units to internal length units (Mpc/h)
    
    :param length_arr: length array in in units
    :type length_arr: floats"""
    global InUnitLength_in_cm
    length_in_cm_d, mass_in_g_d, velocity_in_cm_per_s_d = getAdmissibleLMV()
    in_to_int = np.float32(InUnitLength_in_cm/length_in_cm_d['Mpc/h'])
    return length_arr*in_to_int
    
def MassInToInternal(mass_arr):
    """ Rescale mass_arr from in mass units to internal mass units (E+10M_sun/h) 
    
    :param mass_arr: mass array in in units
    :type mass_arr: floats"""
    global InUnitMass_in_g
    length_in_cm_d, mass_in_g_d, velocity_in_cm_per_s_d = getAdmissibleLMV()
    in_to_int = np.float32(InUnitMass_in_g/mass_in_g_d['E+10M_sun/h'])
    return mass_arr*in_to_int

def VelInToInternal(vel_arr):
    """ Rescale vel_arr from in velocity units to internal velocity units (km/s) 
    
    :param vel_arr: velocity array in in units
    :type vel_arr: floats"""
    global InUnitVelocity_in_cm_per_s
    length_in_cm_d, mass_in_g_d, velocity_in_cm_per_s_d = getAdmissibleLMV()
    in_to_int = np.float32(InUnitVelocity_in_cm_per_s/velocity_in_cm_per_s_d['km/s'])
    return vel_arr*in_to_int
    
def LMVLabel():
    """ Return strings that can be used for plots, i.e. in out units"""
    global OutUnitLength_in_cm
    global OutUnitMass_in_g
    global OutUnitVelocity_in_cm_per_s
    length_in_cm_d, mass_in_g_d, velocity_in_cm_per_s_d = getAdmissibleLMV()
    if OutUnitLength_in_cm in length_in_cm_d.values():
        l_label = list(length_in_cm_d.keys())[np.argmin(abs(np.fromiter(length_in_cm_d.values(), dtype=float)-OutUnitLength_in_cm))]
    else:
        l_label = "{:.2e}kpc".format(OutUnitLength_in_cm/length_in_cm_d['kpc'])
    if OutUnitMass_in_g in mass_in_g_d.values():
        m_label = list(mass_in_g_d.keys())[np.argmin(abs(np.fromiter(mass_in_g_d.values(), dtype=float)-OutUnitMass_in_g))]
    else:
        m_label = "{:.2e}Msun".format(OutUnitMass_in_g/mass_in_g_d['Msun'])
    if OutUnitVelocity_in_cm_per_s in velocity_in_cm_per_s_d.values():
        vel_label = list(velocity_in_cm_per_s_d.keys())[np.argmin(abs(np.fromiter(velocity_in_cm_per_s_d.values(), dtype=float)-OutUnitVelocity_in_cm_per_s))]
    else:
        vel_label = "{:.2e}km/s".format(OutUnitVelocity_in_cm_per_s/velocity_in_cm_per_s_d['km/s'])
    return l_label, m_label, vel_label
    
initialize(gbs=2, maxsize=128, in_length_in_cm = 3.085678e21, in_mass_in_g = 1.989e43, in_velocity_in_cm_per_s = 1e5, little_h = 0.6774, out_length_in_cm = 3.085678e24, out_mass_in_g = 1.989e33, out_velocity_in_cm_per_s = 1e5)