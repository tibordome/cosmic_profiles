#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import h5py
import glob
from mpi4py import MPI
import re
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def getPartType(OBJ_TYPE):
    """ Return particle type number
    
    :param OBJ_TYPE: which simulation particles to consider, 'dm', 'gas' or 'stars'
    :type OBJ_TYPE: str
    :returns: particle type number
    :rtype: int"""
    if OBJ_TYPE == 'dm':
        return 1
    elif OBJ_TYPE == 'stars':
        return 4
    else:
        assert OBJ_TYPE == 'gas', "Please specify either 'dm', 'gas' or 'stars' for OBJ_TYPE"
        return 0

def getFoFSHData(HDF5_GROUP_DEST, RVIR_OR_R200, PART_TYPE):
    """ Retrieve FoF/SH-related DM HDF5 data from the simulation box
    
    :param HDF5_GROUP_DEST: path to snapshot, halo/subhalo data
    :type HDF5_GROUP_DEST: string
    :param RVIR_OR_R200: 'Rvir' if we want quantities (e.g. D_LOGSTART) to be expressed 
        with respect to the virial radius R_vir, 'R200' for the overdensity radius R_200
    :type RVIR_OR_R200: str
    :param PART_TYPE: which simulation particles to consider, 0 for gas, 1 for DM,
        4 for stars
    :type PART_TYPE: int
    :return: nb_shs (# subhalos in each FoF-halo), sh_len (size of each SH), 
        fof_sizes (size of each FoF-halo), group_r200 (R200 radius of each FoF-halo
        in units of cMpc/h)
    :rtype: float and int arrays"""
    return