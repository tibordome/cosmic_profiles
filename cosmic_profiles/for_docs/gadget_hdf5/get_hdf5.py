#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def getHDF5SHData(HDF5_GROUP_DEST, RVIR_OR_R200, PART_TYPE):
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

def getHDF5ObjData(HDF5_SNAP_DEST, PART_TYPE):
    """ Retrieve DM HDF5 data from the simulation box
        
    :param HDF5_SNAP_DEST: path to snapshot, particle data
    :type HDF5_SNAP_DEST: string
    :param PART_TYPE: which simulation particles to consider, 0 for gas, 1 for DM,
        4 for stars
    :type PART_TYPE: int
    :return: obj_xyz (ptc positions in units of cMpc/h), obj_masses (mass of each ptc
        in units of 10^10 M_sun/h), obj_velxyz (velocity of each ptc in units of km/s)
    :rtype: float arrays"""
    return