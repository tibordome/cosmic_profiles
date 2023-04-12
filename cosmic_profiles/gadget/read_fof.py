#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import h5py
from cosmic_profiles.common.caching import np_cache_factory
import glob
from mpi4py import MPI
from cosmic_profiles.common import config
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
    def inner(HDF5_GROUP_DEST, RVIR_OR_R200, PART_TYPE):
        fof_sizes = np.empty(0, dtype = np.int32)
        nb_shs = np.empty(0, dtype = np.int32)
        sh_len = np.empty(0, dtype = np.int32)
        group_r200 = np.empty(0, dtype = np.float32)
        hdf5GroupFilenamesList = glob.glob('{}/*.hdf5'.format(HDF5_GROUP_DEST))
        # Raise Error message if empty
        if len(hdf5GroupFilenamesList) == 0:
            raise ValueError("Could not find any HDF5 files. Please double-check provided paths, specifically HDF5_GROUP_DEST.")
        groupfile_argsort = np.argsort(np.array([np.int32(re.split('(\d+)', file)[-4]) for file in hdf5GroupFilenamesList]))
        hdf5GroupFilenamesList = np.array(hdf5GroupFilenamesList)[groupfile_argsort]
        nb_jobs_to_do = len(hdf5GroupFilenamesList)
        perrank = nb_jobs_to_do//size + (nb_jobs_to_do//size == 0)*1
        do_sth = rank <= nb_jobs_to_do-1
        if size <= nb_jobs_to_do:
            last = rank == size - 1 # Whether or not last process
        else:
            last = rank == nb_jobs_to_do - 1
        count_fof = 0
        count_sh = 0
        l_target_over_curr = 3.085678e24/config.InUnitLength_in_cm
        for snap_run in range(rank*perrank, rank*perrank+do_sth*(perrank+last*(nb_jobs_to_do-(rank+1)*perrank))):
            g = h5py.File(r'{}'.format(hdf5GroupFilenamesList[snap_run]), 'r')
            if 'Group/GroupLenType' in g:
                fof_sizes = np.hstack((fof_sizes, np.int32([np.int32(g['Group/GroupLenType'][i,PART_TYPE]) for i in range(g['Group/GroupLenType'].shape[0])])))
                if RVIR_OR_R200 == 'Rvir':
                    group_r200 = np.hstack((group_r200, np.float32(g['Group/Group_R_TopHat200'][:]/l_target_over_curr)))
                elif RVIR_OR_R200 == 'R200':
                    group_r200 = np.hstack((group_r200, np.float32(g['Group/Group_R_Mean200'][:]/l_target_over_curr)))
                else:
                    raise ValueError("RVIR_OR_R200 should be either 'Rvir' or 'R200'. Please modify the provided RVIR_OR_R200 variable.")
                nb_shs = np.hstack((nb_shs, np.int32([np.int32(g['Group/GroupNsubs'][i]) for i in range(g['Group/GroupNsubs'].shape[0])])))
                count_fof += g['Group/GroupLenType'].shape[0]
            if 'Subhalo/SubhaloLenType' in g:
                sh_len = np.hstack((sh_len, np.int32([np.int32(g['Subhalo/SubhaloLenType'][i,PART_TYPE]) for i in range(g['Subhalo/SubhaloLenType'].shape[0])])))        
                count_sh += g['Subhalo/SubhaloLenType'].shape[0]
        
        count_fof_new = comm.gather(count_fof, root=0)
        count_fof_new = comm.bcast(count_fof_new, root = 0)
        nb_fof = np.sum(np.array(count_fof_new))
        comm.Barrier()
        recvcounts_fof = np.array(count_fof_new)
        rdispls_fof = np.zeros_like(recvcounts_fof)
        for j in range(rdispls_fof.shape[0]):
            rdispls_fof[j] = np.sum(recvcounts_fof[:j])
        count_sh_new = comm.gather(count_sh, root=0)
        count_sh_new = comm.bcast(count_sh_new, root = 0)
        nb_sh = np.sum(np.array(count_sh_new))
        comm.Barrier()
        recvcounts_sh = np.array(count_sh_new)
        rdispls_sh = np.zeros_like(recvcounts_sh)
        for j in range(rdispls_sh.shape[0]):
            rdispls_sh[j] = np.sum(recvcounts_sh[:j])
            
        fof_sizes_total = np.empty(nb_fof, dtype = np.int32)
        group_r200_total = np.empty(nb_fof, dtype = np.float32)
        nb_shs_total = np.empty(nb_fof, dtype = np.int32)
        sh_len_total = np.empty(nb_sh, dtype = np.int32)
        
        comm.Gatherv(fof_sizes, [fof_sizes_total, recvcounts_fof, rdispls_fof, MPI.INT], root = 0)
        comm.Gatherv(group_r200, [group_r200_total, recvcounts_fof, rdispls_fof, MPI.FLOAT], root = 0)
        comm.Gatherv(nb_shs, [nb_shs_total, recvcounts_fof, rdispls_fof, MPI.INT], root = 0)
        comm.Gatherv(sh_len, [sh_len_total, recvcounts_sh, rdispls_sh, MPI.INT], root = 0)
        
        pieces = 1 + (nb_fof>=3*10**8)*nb_fof//(3*10**8) # Not too high since this is a slow-down!
        chunk = nb_fof//pieces
        fof_sizes = np.empty(0, dtype = np.int32)
        group_r200 = np.empty(0, dtype = np.float32)
        nb_shs = np.empty(0, dtype = np.int32)
        for i in range(pieces):
            to_bcast = fof_sizes_total[i*chunk:(i+1)*chunk+(i==(pieces-1))*(nb_fof-pieces*chunk)]
            comm.Bcast(to_bcast, root=0)
            fof_sizes = np.hstack((fof_sizes, to_bcast))
            to_bcast = group_r200_total[i*chunk:(i+1)*chunk+(i==(pieces-1))*(nb_fof-pieces*chunk)]
            comm.Bcast(to_bcast, root=0)
            group_r200 = np.hstack((group_r200, to_bcast))
            to_bcast = nb_shs_total[i*chunk:(i+1)*chunk+(i==(pieces-1))*(nb_fof-pieces*chunk)]
            comm.Bcast(to_bcast, root=0)
            nb_shs = np.hstack((nb_shs, to_bcast))
            
        pieces = 1 + (nb_sh>=3*10**8)*nb_sh//(3*10**8) # Not too high since this is a slow-down!
        chunk = nb_sh//pieces
        sh_len = np.empty(0, dtype = np.int32)
        for i in range(pieces):
            to_bcast = sh_len_total[i*chunk:(i+1)*chunk+(i==(pieces-1))*(nb_sh-pieces*chunk)]
            comm.Bcast(to_bcast, root=0)
            sh_len = np.hstack((sh_len, to_bcast))
        
        return nb_shs, sh_len, fof_sizes, group_r200
    if(not hasattr(getFoFSHData, "inner")):
        getFoFSHData.inner = np_cache_factory(0,0)(inner)
    getFoFSHData.inner(HDF5_GROUP_DEST, RVIR_OR_R200, PART_TYPE)
    return getFoFSHData.inner(HDF5_GROUP_DEST, RVIR_OR_R200, PART_TYPE)