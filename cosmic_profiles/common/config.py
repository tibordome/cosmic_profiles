#!/usr/bin/env python3
# -*- coding: utf-8 -*-
    
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def initialize(GB):
    """ Initialize all global variables, should only be called once."""
    
    global GBs # Number of gigabytes. If only this much left (according to psutil.virtual_memory().available), cache will be considered full.
    
    GBs = GB
    
def updateCachingMaxGBs(GB):
    global GBs
    GBs = GB
    
initialize(GB=1) # I.e. if only 1 GB of memory available, stop caching to avoid thrashing.