#!/usr/bin/env python3
# -*- coding: utf-8 -*-
    
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def initialize(gbs, maxsize, in_unit_length_in_cm, in_unit_mass_in_g, in_unit_velocity_in_cm_per_s, out_unit_length_in_cm, out_unit_mass_in_g, out_unit_velocity_in_cm_per_s):
    """ Initialize all global variables, should only be called once.
    
    :param gbs: new memory limit in GBs
    :type gbs: int, or float
    :param maxsize: new maximum cache size (note that multiple functions are decorated, each will have
        new maximum cache size)
    :type maxsize: int
    :param in_unit_length_in_cm: unit for length in cm as used in provided data
    :type in_unit_length_in_cm: float
    :param in_unit_mass_in_g: unit for mass in g as used in provided data
    :type in_unit_mass_in_g: float
    :param in_unit_velocity_in_cm_per_s: unit for velocity in cm/s as used in provided data
    :type in_unit_velocity_in_cm_per_s: float
    :param out_unit_length_in_cm: unit for length in cm to be used by CosmicProfiles
    :type out_unit_length_in_cm: float
    :param out_unit_mass_in_g: unit for mass in g to be used by CosmicProfiles
    :type out_unit_mass_in_g: float
    :param out_unit_velocity_in_cm_per_s: unit for velocity in cm/s to be used by CosmicProfiles
    :type out_unit_velocity_in_cm_per_s: float"""
    
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
    global OutUnitLength_in_cm
    global OutUnitMass_in_g
    global OutUnitVelocity_in_cm_per_s
    
    GBs = gbs
    CACHE_MAXSIZE = maxsize
    InUnitLength_in_cm = in_unit_length_in_cm
    InUnitMass_in_g = in_unit_mass_in_g
    InUnitVelocity_in_cm_per_s = in_unit_velocity_in_cm_per_s
    OutUnitLength_in_cm = out_unit_length_in_cm
    OutUnitMass_in_g = out_unit_mass_in_g
    OutUnitVelocity_in_cm_per_s = out_unit_velocity_in_cm_per_s
    
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
    
def updateInUnitSystem(in_unit_length_in_cm, in_unit_mass_in_g, in_unit_velocity_in_cm_per_s):
    """ Inform CosmicProfiles about system of units employed in snapshot data
    
    :param in_unit_length_in_cm: unit for length in cm as used in provided data
    :type in_unit_length_in_cm: float
    :param in_unit_mass_in_g: unit for mass in g as used in provided data
    :type in_unit_mass_in_g: float
    :param in_unit_velocity_in_cm_per_s: unit for velocity in cm/s as used in provided data
    :type in_unit_velocity_in_cm_per_s: float"""
    
    global InUnitLength_in_cm
    global InUnitMass_in_g
    global InUnitVelocity_in_cm_per_s
    
    InUnitLength_in_cm = in_unit_length_in_cm
    InUnitMass_in_g = in_unit_mass_in_g
    InUnitVelocity_in_cm_per_s = in_unit_velocity_in_cm_per_s
    
def updateOutUnitSystem(out_unit_length_in_cm, out_unit_mass_in_g, out_unit_velocity_in_cm_per_s):
    """ Update unit system to be used by CosmicProfiles in its outputs
    
    :param out_unit_length_in_cm: unit for length in cm to be used by CosmicProfiles
    :type out_unit_length_in_cm: float
    :param out_unit_mass_in_g: unit for mass in g to be used by CosmicProfiles
    :type out_unit_mass_in_g: float
    :param out_unit_velocity_in_cm_per_s: unit for velocity in cm/s to be used by CosmicProfiles
    :type out_unit_velocity_in_cm_per_s: float"""
    
    global OutUnitLength_in_cm
    global OutUnitMass_in_g
    global OutUnitVelocity_in_cm_per_s
    
    OutUnitLength_in_cm = out_unit_length_in_cm
    OutUnitMass_in_g = out_unit_mass_in_g
    OutUnitVelocity_in_cm_per_s = out_unit_velocity_in_cm_per_s
    
initialize(gbs=2, maxsize=128, in_unit_length_in_cm = 3.085678e21, in_unit_mass_in_g = 1.989e43, in_unit_velocity_in_cm_per_s = 1e5, out_unit_length_in_cm = 3.085678e24, out_unit_mass_in_g = 1.989e33, out_unit_velocity_in_cm_per_s = 1e5)