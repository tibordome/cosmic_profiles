#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import itertools
import h5py
from cosmic_profiles.common.caching import np_cache_factory
import glob
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def getHDF5Data(HDF5_SNAP_DEST, HDF5_GROUP_DEST):
    """ Retrieve all relevant HDF5 data from the simulation box, both DM and stars
    
    :param HDF5_SNAP_DEST: path to snapshot, particle data
    :type HDF5_SNAP_DEST: string
    :param HDF5_GROUP_DEST: path to snapshot, halo/subhalo data
    :type HDF5_GROUP_DEST: string
    :return: dm_xyz (DM ptc positions), star_xyz (star ptc positions), 
        nb_shs (# subhalos in each FoF-halo), 
        sh_len (size of each SH), fof_dm_sizes (size of each FoF-halo), 
        dm_masses (mass of each DM ptc), star_masses (mass of each star ptc),
        fof_masses (mass of each FoF-halo)
    :rtype: float and int arrays"""
    def inner(HDF5_SNAP_DEST, HDF5_GROUP_DEST):  
        dm_x = np.empty(0, dtype = np.float32)
        dm_y = np.empty(0, dtype = np.float32)
        dm_z = np.empty(0, dtype = np.float32)
        star_masses = np.empty(0, dtype = np.float32)
        star_x = np.empty(0, dtype = np.float32)
        star_y = np.empty(0, dtype = np.float32)
        star_z = np.empty(0, dtype = np.float32)
        fof_masses = np.empty(0, dtype = np.float32)
        fof_dm_sizes = []
        nb_shs = []
        sh_len = []
        hdf5SnapFilenamesList = glob.glob('{}/*.hdf5'.format(HDF5_SNAP_DEST))
        hdf5GroupFilenamesList = glob.glob('{}/*.hdf5'.format(HDF5_GROUP_DEST))
        for fname in hdf5GroupFilenamesList:
            g = h5py.File(r'{}'.format(fname), 'r')
            if 'Group/GroupLenType' in g:
                to_be_appended = [np.float32(g['Group/GroupLenType'][i,1]) for i in range(g['Group/GroupLenType'].shape[0])]
                fof_dm_sizes.append(to_be_appended)
                nb_shs.append([np.float32(g['Group/GroupNsubs'][i]) for i in range(g['Group/GroupNsubs'].shape[0])])
                fof_masses = np.hstack((fof_masses, np.float32(g['Group/GroupMassType'][:,1])))
            if 'Subhalo/SubhaloLenType' in g:
                sh_len.append([np.float32(g['Subhalo/SubhaloLenType'][i,1]) for i in range(g['Subhalo/SubhaloLenType'].shape[0])])
        for fname in hdf5SnapFilenamesList:
            f = h5py.File(r'{}'.format(fname), 'r')
            dm_x = np.hstack((dm_x, np.float32(f['PartType1/Coordinates'][:,0]/1000))) # in Mpc = 3.085678e+27 cm
            dm_y = np.hstack((dm_y, np.float32(f['PartType1/Coordinates'][:,1]/1000))) 
            dm_z = np.hstack((dm_z, np.float32(f['PartType1/Coordinates'][:,2]/1000))) 
            if 'PartType4/Coordinates' in f:
                is_star = np.where(f['PartType4/GFM_StellarFormationTime'][:]>0)[0] # Wind particle?
                star_x = np.hstack((star_x, np.float32(f['PartType4/Coordinates'][is_star][:,0]/1000))) # in Mpc = 3.085678e+27 cm
                star_y = np.hstack((star_y, np.float32(f['PartType4/Coordinates'][is_star][:,1]/1000))) 
                star_z = np.hstack((star_z, np.float32(f['PartType4/Coordinates'][is_star][:,2]/1000))) 
                star_masses = np.hstack((star_masses, np.float32(f['PartType4/Masses'][is_star][:]))) # in 1.989e+43 g
                
        dm_xyz = np.hstack((np.reshape(dm_x, (dm_x.shape[0],1)), np.reshape(dm_y, (dm_y.shape[0],1)), np.reshape(dm_z, (dm_z.shape[0],1))))
        dm_masses = np.ones((dm_xyz.shape[0],), dtype=np.float32)*np.float32(f['Header'].attrs['MassTable'][1]) # in 1.989e+43 g
        fof_dm_sizes = list(itertools.chain.from_iterable(fof_dm_sizes)) # Simple list, not nested list
        nb_shs = list(itertools.chain.from_iterable(nb_shs))
        sh_len = list(itertools.chain.from_iterable(sh_len))
        star_xyz = np.hstack((np.reshape(star_x, (star_x.shape[0],1)), np.reshape(star_y, (star_y.shape[0],1)), np.reshape(star_z, (star_z.shape[0],1))))
        
        return dm_xyz, star_xyz, nb_shs, sh_len, fof_dm_sizes, dm_masses, star_masses, fof_masses
    if(not hasattr(getHDF5Data, "inner")):
        getHDF5Data.inner = np_cache_factory(0,0)(inner)
    getHDF5Data.inner(HDF5_SNAP_DEST, HDF5_GROUP_DEST)
    return getHDF5Data.inner(HDF5_SNAP_DEST, HDF5_GROUP_DEST)

def getHDF5GxData(HDF5_SNAP_DEST, HDF5_GROUP_DEST):
    """ Retrieve stars-related HDF5 data from the simulation box
    
    Note that Type4 refers to stars
    
    :param HDF5_SNAP_DEST: path to snapshot, particle data
    :type HDF5_SNAP_DEST: string
    :param HDF5_GROUP_DEST: path to snapshot, halo/subhalo data
    :type HDF5_GROUP_DEST: string
    :return: star_xyz (star ptc positions),
        nb_shs (# subhalos in each FoF-halo), 
        star_masses (mass of each star ptc),
        star_velxyz (star ptc velocities),
        is_star (whether or not particle is star particle or wind particle)
    :rtype: float and int arrays"""
    def inner(HDF5_SNAP_DEST, HDF5_GROUP_DEST):
        is_star = np.empty(0, dtype = bool)
        star_masses = np.empty(0, dtype = np.float32)
        star_x = np.empty(0, dtype = np.float32)
        star_y = np.empty(0, dtype = np.float32)
        star_z = np.empty(0, dtype = np.float32)
        star_velx = np.empty(0, dtype = np.float32)
        star_vely = np.empty(0, dtype = np.float32)
        star_velz = np.empty(0, dtype = np.float32)
        nb_shs_l = []
        
        # Snap data
        hdf5SnapFilenamesList = glob.glob('{}/*.hdf5'.format(HDF5_SNAP_DEST))
        nb_jobs_to_do = len(hdf5SnapFilenamesList)
        perrank = nb_jobs_to_do//size + (nb_jobs_to_do//size == 0)*1
        do_sth = rank <= nb_jobs_to_do-1
        count = 0
        if size <= nb_jobs_to_do:
            last = rank == size - 1 # Whether or not last process
        else:
            last = rank == nb_jobs_to_do - 1
        for snap_run in range(rank*perrank, rank*perrank+do_sth*(perrank+last*(nb_jobs_to_do-(rank+1)*perrank))):
            f = h5py.File(r'{}'.format(hdf5SnapFilenamesList[snap_run]), 'r')
            if 'PartType4/Coordinates' in f:
                is_star = np.hstack((is_star, f['PartType4/GFM_StellarFormationTime'][:] > 0)) # Wind particle?
                star_x = np.hstack((star_x, np.float32(f['PartType4/Coordinates'][:,0]/1000))) # in Mpc = 3.085678e+27 cm
                star_y = np.hstack((star_y, np.float32(f['PartType4/Coordinates'][:,1]/1000))) 
                star_z = np.hstack((star_z, np.float32(f['PartType4/Coordinates'][:,2]/1000))) 
                star_velx = np.hstack((star_velx, np.float32(f['PartType4/Velocities'][:,0]))) # in km/s
                star_vely = np.hstack((star_vely, np.float32(f['PartType4/Velocities'][:,1]))) 
                star_velz = np.hstack((star_velz, np.float32(f['PartType4/Velocities'][:,2]))) 
                star_masses = np.hstack((star_masses, np.float32(f['PartType4/Masses'][:]))) # in 1.989e+43 g
                count += f['PartType4/Coordinates'][:].shape[0]
                    
        count_new = comm.gather(count, root=0)
        count_new = comm.bcast(count_new, root = 0)
        nb_star_ptcs = np.sum(np.array(count_new))
        comm.Barrier()
        
        recvcounts = np.array(count_new)
        rdispls = np.zeros_like(recvcounts)
        for j in range(rdispls.shape[0]):
            rdispls[j] = np.sum(recvcounts[:j])
        star_x_total = np.empty(nb_star_ptcs, dtype = np.float32)
        star_y_total = np.empty(nb_star_ptcs, dtype = np.float32)
        star_z_total = np.empty(nb_star_ptcs, dtype = np.float32)
        star_velx_total = np.empty(nb_star_ptcs, dtype = np.float32)
        star_vely_total = np.empty(nb_star_ptcs, dtype = np.float32)
        star_velz_total = np.empty(nb_star_ptcs, dtype = np.float32)
        is_star_total = np.empty(nb_star_ptcs, dtype = np.float32)
        star_masses_total = np.empty(nb_star_ptcs, dtype = np.float32)
        
        comm.Gatherv(star_x, [star_x_total, recvcounts, rdispls, MPI.FLOAT], root = 0)
        comm.Gatherv(star_y, [star_y_total, recvcounts, rdispls, MPI.FLOAT], root = 0)
        comm.Gatherv(star_z, [star_z_total, recvcounts, rdispls, MPI.FLOAT], root = 0)
        comm.Gatherv(star_velx, [star_velx_total, recvcounts, rdispls, MPI.FLOAT], root = 0)
        comm.Gatherv(star_vely, [star_vely_total, recvcounts, rdispls, MPI.FLOAT], root = 0)
        comm.Gatherv(star_velz, [star_velz_total, recvcounts, rdispls, MPI.FLOAT], root = 0)
        comm.Gatherv(is_star, [is_star_total, recvcounts, rdispls, MPI.BOOL], root = 0)
        comm.Gatherv(star_masses, [star_masses_total, recvcounts, rdispls, MPI.FLOAT], root = 0)
        
        pieces = 1 + (nb_star_ptcs>=3*10**8)*nb_star_ptcs//(3*10**8) # Not too high since this is a slow-down!
        chunk = nb_star_ptcs//pieces
        star_x = np.empty(0, dtype = np.float32)
        star_y = np.empty(0, dtype = np.float32)
        star_z = np.empty(0, dtype = np.float32)
        star_velx = np.empty(0, dtype = np.float32)
        star_vely = np.empty(0, dtype = np.float32)
        star_velz = np.empty(0, dtype = np.float32)
        is_star = np.empty(0, dtype = bool)
        star_masses = np.empty(0, dtype = np.float32)
        for i in range(pieces):
            to_bcast = star_x_total[i*chunk:(i+1)*chunk+(i==(pieces-1))*(nb_star_ptcs-pieces*chunk)]
            comm.Bcast(to_bcast, root=0)
            star_x = np.hstack((star_x, to_bcast))
            to_bcast = star_y_total[i*chunk:(i+1)*chunk+(i==(pieces-1))*(nb_star_ptcs-pieces*chunk)]
            comm.Bcast(to_bcast, root=0)
            star_y = np.hstack((star_y, to_bcast))
            to_bcast = star_z_total[i*chunk:(i+1)*chunk+(i==(pieces-1))*(nb_star_ptcs-pieces*chunk)]
            comm.Bcast(to_bcast, root=0)
            star_z = np.hstack((star_z, to_bcast))
            to_bcast = star_velx_total[i*chunk:(i+1)*chunk+(i==(pieces-1))*(nb_star_ptcs-pieces*chunk)]
            comm.Bcast(to_bcast, root=0)
            star_velx = np.hstack((star_velx, to_bcast))
            to_bcast = star_vely_total[i*chunk:(i+1)*chunk+(i==(pieces-1))*(nb_star_ptcs-pieces*chunk)]
            comm.Bcast(to_bcast, root=0)
            star_vely = np.hstack((star_vely, to_bcast))
            to_bcast = star_velz_total[i*chunk:(i+1)*chunk+(i==(pieces-1))*(nb_star_ptcs-pieces*chunk)]
            comm.Bcast(to_bcast, root=0)
            star_velz = np.hstack((star_velz, to_bcast))
            to_bcast = is_star_total[i*chunk:(i+1)*chunk+(i==(pieces-1))*(nb_star_ptcs-pieces*chunk)]
            comm.Bcast(to_bcast, root=0)
            is_star = np.hstack((is_star, to_bcast))
            to_bcast = star_masses_total[i*chunk:(i+1)*chunk+(i==(pieces-1))*(nb_star_ptcs-pieces*chunk)]
            comm.Bcast(to_bcast, root=0)
            star_masses = np.hstack((star_masses, to_bcast))
        
        star_xyz = np.hstack((np.reshape(star_x, (star_x.shape[0],1)), np.reshape(star_y, (star_y.shape[0],1)), np.reshape(star_z, (star_z.shape[0],1))))
        star_velxyz = np.hstack((np.reshape(star_velx, (star_velx.shape[0],1)), np.reshape(star_vely, (star_vely.shape[0],1)), np.reshape(star_velz, (star_velz.shape[0],1))))
        
        # Group data
        hdf5GroupFilenamesList = glob.glob('{}/*.hdf5'.format(HDF5_GROUP_DEST))
        nb_jobs_to_do = len(hdf5GroupFilenamesList)
        perrank = nb_jobs_to_do//size + (nb_jobs_to_do//size == 0)*1
        do_sth = rank <= nb_jobs_to_do-1
        count = 0
        count_sh_l = 0
        if size <= nb_jobs_to_do:
            last = rank == size - 1 # Whether or not last process
        else:
            last = rank == nb_jobs_to_do - 1
        for snap_run in range(rank*perrank, rank*perrank+do_sth*(perrank+last*(nb_jobs_to_do-(rank+1)*perrank))):
            g = h5py.File(r'{}'.format(hdf5GroupFilenamesList[snap_run]), 'r')
            if 'Group/GroupCM' in g:
                nb_shs_l.append([np.int32(g['Group/GroupNsubs'][i]) for i in range(g['Group/GroupNsubs'].shape[0])])
                count_sh_l += 1
        
        count_new_sh_l = comm.gather(count_sh_l, root=0)
        nb_shs_l = comm.gather(nb_shs_l, root=0)
        comm.Barrier()
        
        if rank == 0:
            nb_shs_l = [nb_shs_l[i][j] for i in range(size) for j in range(count_new_sh_l[i])]
        nb_shs_l = comm.bcast(nb_shs_l, root = 0)
        nb_shs_l = list(itertools.chain.from_iterable(nb_shs_l))
        
        return star_xyz, nb_shs_l, star_masses, star_velxyz, is_star
    if(not hasattr(getHDF5GxData, "inner")):
        getHDF5GxData.inner = np_cache_factory(0,0)(inner)
    getHDF5GxData.inner(HDF5_SNAP_DEST, HDF5_GROUP_DEST)
    return getHDF5GxData.inner(HDF5_SNAP_DEST, HDF5_GROUP_DEST)

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
        fof_sizes (size of each FoF-halo), group_r200 (R200 radius of each FoF-halo)
    :rtype: float and int arrays"""
    def inner(HDF5_GROUP_DEST, RVIR_OR_R200, PART_TYPE):
        fof_sizes = np.empty(0, dtype = np.int32)
        nb_shs = np.empty(0, dtype = np.int32)
        sh_len = np.empty(0, dtype = np.int32)
        group_r200 = np.empty(0, dtype = np.float32)
        hdf5GroupFilenamesList = glob.glob('{}/*.hdf5'.format(HDF5_GROUP_DEST))
        nb_jobs_to_do = len(hdf5GroupFilenamesList)
        perrank = nb_jobs_to_do//size + (nb_jobs_to_do//size == 0)*1
        do_sth = rank <= nb_jobs_to_do-1
        if size <= nb_jobs_to_do:
            last = rank == size - 1 # Whether or not last process
        else:
            last = rank == nb_jobs_to_do - 1
        count_fof = 0
        count_sh = 0
        for snap_run in range(rank*perrank, rank*perrank+do_sth*(perrank+last*(nb_jobs_to_do-(rank+1)*perrank))):
            g = h5py.File(r'{}'.format(hdf5GroupFilenamesList[snap_run]), 'r')
            if 'Group/GroupLenType' in g:
                fof_sizes = np.hstack((fof_sizes, np.int32([np.int32(g['Group/GroupLenType'][i,1]) for i in range(g['Group/GroupLenType'].shape[0])])))
                if RVIR_OR_R200 == 'Rvir':
                    group_r200 = np.hstack((group_r200, np.float32(g['Group/Group_R_TopHat200'][:]/1000)))
                elif RVIR_OR_R200 == 'R200':
                    group_r200 = np.hstack((group_r200, np.float32(g['Group/Group_R_Mean200'][:]/1000)))
                else:
                    raise ValueError("RVIR_OR_R200 should be either 'Rvir' or 'R200'. Please modify the provided RVIR_OR_R200 variable.")
                nb_shs = np.hstack((nb_shs, np.int32([np.int32(g['Group/GroupNsubs'][i]) for i in range(g['Group/GroupNsubs'].shape[0])])))
                count_fof += g['Group/GroupLenType'].shape[0]
            if 'Subhalo/SubhaloLenType' in g:
                sh_len = np.hstack((sh_len, np.int32([np.int32(g['Subhalo/SubhaloLenType'][i,1]) for i in range(g['Subhalo/SubhaloLenType'].shape[0])])))        
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
    if(not hasattr(getHDF5SHData, "inner")):
        getHDF5SHData.inner = np_cache_factory(0,0)(inner)
    getHDF5SHData.inner(HDF5_GROUP_DEST, RVIR_OR_R200, PART_TYPE)
    return getHDF5SHData.inner(HDF5_GROUP_DEST, RVIR_OR_R200, PART_TYPE)

def getHDF5DMData(HDF5_SNAP_DEST):
    """ Retrieve DM HDF5 data from the simulation box
        
    :param HDF5_SNAP_DEST: path to snapshot, particle data
    :type HDF5_SNAP_DEST: string
    :return: dm_xyz (DM ptc positions), dm_masses (mass of each DM ptc),
        dm_velxyz (velocity of each DM ptc)
    :rtype: float and int arrays"""
    def inner(HDF5_SNAP_DEST):
        dm_x = np.empty(0, dtype = np.float32)
        dm_y = np.empty(0, dtype = np.float32)
        dm_z = np.empty(0, dtype = np.float32)
        dm_velx = np.empty(0, dtype = np.float32)
        dm_vely = np.empty(0, dtype = np.float32)
        dm_velz = np.empty(0, dtype = np.float32)
        dm_masses = np.empty(0, dtype = np.float32)
        hdf5SnapFilenamesList = glob.glob('{}/*.hdf5'.format(HDF5_SNAP_DEST))
        nb_jobs_to_do = len(hdf5SnapFilenamesList)
        perrank = nb_jobs_to_do//size + (nb_jobs_to_do//size == 0)*1
        do_sth = rank <= nb_jobs_to_do-1
        count = 0
        if size <= nb_jobs_to_do:
            last = rank == size - 1 # Whether or not last process
        else:
            last = rank == nb_jobs_to_do - 1
        for snap_run in range(rank*perrank, rank*perrank+do_sth*(perrank+last*(nb_jobs_to_do-(rank+1)*perrank))):
            f = h5py.File(r'{}'.format(hdf5SnapFilenamesList[snap_run]), 'r')
            dm_x = np.hstack((dm_x, np.float32(f['PartType1/Coordinates'][:,0]/1000))) # in Mpc = 3.085678e+27 cm
            dm_y = np.hstack((dm_y, np.float32(f['PartType1/Coordinates'][:,1]/1000)))
            dm_z = np.hstack((dm_z, np.float32(f['PartType1/Coordinates'][:,2]/1000)))
            dm_velx = np.hstack((dm_velx, np.float32(f['PartType1/Velocities'][:,0]))) # in km/s
            dm_vely = np.hstack((dm_vely, np.float32(f['PartType1/Velocities'][:,1])))
            dm_velz = np.hstack((dm_velz, np.float32(f['PartType1/Velocities'][:,2])))
            dm_masses = np.hstack((dm_masses, np.ones((f['PartType1/Coordinates'][:].shape[0],), dtype=np.float32)*np.float32(f['Header'].attrs['MassTable'][1]))) # in 1.989e+43 g
            count += f['PartType1/Coordinates'][:].shape[0]
        count_new = comm.gather(count, root=0)
        count_new = comm.bcast(count_new, root = 0)
        nb_dm_ptcs = np.sum(np.array(count_new))
        comm.Barrier()
        recvcounts = np.array(count_new)
        rdispls = np.zeros_like(recvcounts)
        for j in range(rdispls.shape[0]):
            rdispls[j] = np.sum(recvcounts[:j])
        dm_x_total = np.empty(nb_dm_ptcs, dtype = np.float32)
        dm_y_total = np.empty(nb_dm_ptcs, dtype = np.float32)
        dm_z_total = np.empty(nb_dm_ptcs, dtype = np.float32)
        dm_velx_total = np.empty(nb_dm_ptcs, dtype = np.float32)
        dm_vely_total = np.empty(nb_dm_ptcs, dtype = np.float32)
        dm_velz_total = np.empty(nb_dm_ptcs, dtype = np.float32)
        dm_masses_total = np.empty(nb_dm_ptcs, dtype = np.float32)
        
        comm.Gatherv(dm_x, [dm_x_total, recvcounts, rdispls, MPI.FLOAT], root = 0)
        comm.Gatherv(dm_y, [dm_y_total, recvcounts, rdispls, MPI.FLOAT], root = 0)
        comm.Gatherv(dm_z, [dm_z_total, recvcounts, rdispls, MPI.FLOAT], root = 0)
        comm.Gatherv(dm_velx, [dm_velx_total, recvcounts, rdispls, MPI.FLOAT], root = 0)
        comm.Gatherv(dm_vely, [dm_vely_total, recvcounts, rdispls, MPI.FLOAT], root = 0)
        comm.Gatherv(dm_velz, [dm_velz_total, recvcounts, rdispls, MPI.FLOAT], root = 0)
        comm.Gatherv(dm_masses, [dm_masses_total, recvcounts, rdispls, MPI.FLOAT], root = 0)
        
        pieces = 1 + (nb_dm_ptcs>=3*10**8)*nb_dm_ptcs//(3*10**8) # Not too high since this is a slow-down!
        chunk = nb_dm_ptcs//pieces
        dm_x = np.empty(0, dtype = np.float32)
        dm_y = np.empty(0, dtype = np.float32)
        dm_z = np.empty(0, dtype = np.float32)
        dm_velx = np.empty(0, dtype = np.float32)
        dm_vely = np.empty(0, dtype = np.float32)
        dm_velz = np.empty(0, dtype = np.float32)
        dm_masses = np.empty(0, dtype = np.float32)
        for i in range(pieces):
            to_bcast = dm_x_total[i*chunk:(i+1)*chunk+(i==(pieces-1))*(nb_dm_ptcs-pieces*chunk)]
            comm.Bcast(to_bcast, root=0)
            dm_x = np.hstack((dm_x, to_bcast))
            to_bcast = dm_y_total[i*chunk:(i+1)*chunk+(i==(pieces-1))*(nb_dm_ptcs-pieces*chunk)]
            comm.Bcast(to_bcast, root=0)
            dm_y = np.hstack((dm_y, to_bcast))
            to_bcast = dm_z_total[i*chunk:(i+1)*chunk+(i==(pieces-1))*(nb_dm_ptcs-pieces*chunk)]
            comm.Bcast(to_bcast, root=0)
            dm_z = np.hstack((dm_z, to_bcast))
            to_bcast = dm_velx_total[i*chunk:(i+1)*chunk+(i==(pieces-1))*(nb_dm_ptcs-pieces*chunk)]
            comm.Bcast(to_bcast, root=0)
            dm_velx = np.hstack((dm_velx, to_bcast))
            to_bcast = dm_vely_total[i*chunk:(i+1)*chunk+(i==(pieces-1))*(nb_dm_ptcs-pieces*chunk)]
            comm.Bcast(to_bcast, root=0)
            dm_vely = np.hstack((dm_vely, to_bcast))
            to_bcast = dm_velz_total[i*chunk:(i+1)*chunk+(i==(pieces-1))*(nb_dm_ptcs-pieces*chunk)]
            comm.Bcast(to_bcast, root=0)
            dm_velz = np.hstack((dm_velz, to_bcast))
            to_bcast = dm_masses_total[i*chunk:(i+1)*chunk+(i==(pieces-1))*(nb_dm_ptcs-pieces*chunk)]
            comm.Bcast(to_bcast, root=0)
            dm_masses = np.hstack((dm_masses, to_bcast))
    
        dm_xyz = np.hstack((np.reshape(dm_x, (dm_x.shape[0],1)), np.reshape(dm_y, (dm_y.shape[0],1)), np.reshape(dm_z, (dm_z.shape[0],1))))
        dm_velxyz = np.hstack((np.reshape(dm_velx, (dm_velx.shape[0],1)), np.reshape(dm_vely, (dm_vely.shape[0],1)), np.reshape(dm_velz, (dm_velz.shape[0],1))))
    
        return dm_xyz, dm_masses, dm_velxyz
    if(not hasattr(getHDF5DMData, "inner")):
        getHDF5DMData.inner = np_cache_factory(0,0)(inner)
    getHDF5DMData.inner(HDF5_SNAP_DEST)
    return getHDF5DMData.inner(HDF5_SNAP_DEST)