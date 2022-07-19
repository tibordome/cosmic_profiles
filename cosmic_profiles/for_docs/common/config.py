#!/usr/bin/env python3
# -*- coding: utf-8 -*-
    
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def initialize(GB, MAXSIZE):
    """ Initialize all global variables, should only be called once.

    :param GB: new memory limit
    :type GB: int, or float
    :param MAXSIZE: new maximum size of each cache
    :type MAXSIZE: int"""
    
    global GBs # Number of gigabytes. If only this much left (according to psutil.virtual_memory().available), cache will be considered full.
               # I.e. if this many GBs of memory available, stop caching to avoid thrashing.

    global CACHE_MAXSIZE # Defines the maximum number of entries before the cache starts evicting old items
                         # Note that since the np_cache_factory decorator is applied to many functions,
                         # CACHE_MAXSIZE refers to the maximum cache size of each function separately.
                         # If CACHE_MAXSIZE == None, then the cache will grow indefinitely, and no entries will be ever evicted.
                         # Unless GBs is 0, CACHE_MAXSIZE has no effect.
    
    GBs = GB
    CACHE_MAXSIZE = MAXSIZE
    
def updateCachingMaxGBs(GB):
    """ Update ``use_memory_up_to`` argument of LRU-caching decorator

    :param GB: new memory limit
    :type GB: int, or float""" 
    global GBs
    GBs = GB

def updateCachingMaxSize(MAXSIZE):
    """ Update ``maxsize`` argument of LRU-caching decorator

    :param MAXSIZE: new maximum size of each cache
    :type MAXSIZE: int""" 
    global CACHE_MAXSIZE
    CACHE_MAXSIZE = MAXSIZE
    
initialize(GB=2, MAXSIZE=128)
