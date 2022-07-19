#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import itertools
import h5py
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def getHDF5Data(HDF5_SNAP_DEST, HDF5_GROUP_DEST, SNAP_MAX, SNAP):
    """ Retrieve all relevant HDF5 data from the simulation box, both DM and stars
    
    :param HDF5_SNAP_DEST: path to snapshot, particle data
    :type HDF5_SNAP_DEST: string
    :param HDF5_GROUP_DEST: path to snapshot, halo/subhalo data
    :type HDF5_GROUP_DEST: string
    :param SNAP_MAX: number of files per snapshot
    :type SNAP_MAX: int
    :param SNAP: snap number of interest, e.g. '000' or '038'
    :type SNAP: string
    :return: dm_xyz (DM ptc positions), star_xyz (star ptc positions), 
        sh_com (SH COMs), nb_shs (# subhalos in each FoF-halo), 
        sh_len (size of each SH), fof_dm_sizes (size of each FoF-halo), 
        dm_masses (mass of each DM ptc), star_masses (mass of each star ptc),
        fof_com (COM of each FoF-halo), fof_masses (mass of each FoF-halo)
    :rtype: float and int arrays"""
    return

def getHDF5GxData(HDF5_SNAP_DEST, HDF5_GROUP_DEST, SNAP_MAX, SNAP):
    """ Retrieve stars-related HDF5 data from the simulation box
    
    Note that Type4 refers to stars
    
    :param HDF5_SNAP_DEST: path to snapshot, particle data
    :type HDF5_SNAP_DEST: string
    :param HDF5_GROUP_DEST: path to snapshot, halo/subhalo data
    :type HDF5_GROUP_DEST: string
    :param SNAP_MAX: number of files per snapshot
    :type SNAP_MAX: int
    :param SNAP: snap number of interest, e.g. '000' or '038'
    :type SNAP: string
    :return: star_xyz (star ptc positions), fof_com (COM of each FoF-halo), 
        sh_com (SH COMs), nb_shs (# subhalos in each FoF-halo), 
        star_masses (mass of each star ptc)
    :rtype: float and int arrays"""
    return

def getHDF5SHDMData(HDF5_GROUP_DEST, SNAP_MAX, SNAP, WANT_RVIR):
    """ Retrieve FoF/SH-related HDF5 data from the simulation box
        
    :param HDF5_GROUP_DEST: path to snapshot, halo/subhalo data
    :type HDF5_GROUP_DEST: string
    :param SNAP_MAX: number of files per snapshot
    :type SNAP_MAX: int
    :param SNAP: snap number of interest, e.g. '000' or '038'
    :type SNAP: string
    :param WANT_RVIR: Whether or not we want quantities (e.g. D_LOGSTART) expressed 
            with respect to the virial radius R_vir or the overdensity radius R_200
    :type WANT_RVIR: boolean
    :return: nb_shs (# subhalos in each FoF-halo), sh_len (size of each SH), 
        fof_dm_sizes (size of each FoF-halo), group_r200 (R200 radius of each FoF-halo), 
        fof_masses (mass of each FoF-halo), fof_com (COM of each FoF-halo)
    :rtype: float and int arrays"""
    return

def getHDF5SHGxData(HDF5_GROUP_DEST, SNAP_MAX, SNAP):
    """ Retrieve FoF/SH-related HDF5 data from the simulation box
        
    :param HDF5_GROUP_DEST: path to snapshot, halo/subhalo data
    :type HDF5_GROUP_DEST: string
    :param SNAP_MAX: number of files per snapshot
    :type SNAP_MAX: int
    :param SNAP: snap number of interest, e.g. '000' or '038'
    :type SNAP: string
    :return: nb_shs (# subhalos in each FoF-halo), sh_len (star particle size of each SH), 
        fof_gx_sizes (star particle size of each FoF-halo)
    :rtype: int arrays"""
    return

def getHDF5DMData(HDF5_SNAP_DEST, SNAP_MAX, SNAP):
    """ Retrieve FoF/SH-related HDF5 data from the simulation box
        
    :param HDF5_GROUP_DEST: path to snapshot, halo/subhalo data
    :type HDF5_GROUP_DEST: string
    :param SNAP_MAX: number of files per snapshot
    :type SNAP_MAX: int
    :param SNAP: snap number of interest, e.g. '000' or '038'
    :type SNAP: string
    :return: dm_xyz (DM ptc positions), dm_masses (mass of each DM ptc),
        dm_velxyz (velocity of each DM ptc)
    :rtype: float and int arrays"""
    return
