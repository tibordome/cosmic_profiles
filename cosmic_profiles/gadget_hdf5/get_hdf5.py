#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import itertools
import h5py
from cosmic_profiles.common.caching import np_cache_factory
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

@np_cache_factory(0, 0)
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
        
    sh_x = np.empty(0, dtype = np.float32)
    sh_y = np.empty(0, dtype = np.float32)
    sh_z = np.empty(0, dtype = np.float32)
    dm_x = np.empty(0, dtype = np.float32)
    dm_y = np.empty(0, dtype = np.float32)
    dm_z = np.empty(0, dtype = np.float32)
    star_masses = np.empty(0, dtype = np.float32)
    star_x = np.empty(0, dtype = np.float32)
    star_y = np.empty(0, dtype = np.float32)
    star_z = np.empty(0, dtype = np.float32)
    fof_x = np.empty(0, dtype = np.float32)
    fof_y = np.empty(0, dtype = np.float32)
    fof_z = np.empty(0, dtype = np.float32)
    fof_masses = np.empty(0, dtype = np.float32)
    fof_dm_sizes = []
    nb_shs = []
    sh_len = []
    for snap_run in range(SNAP_MAX):
        f = h5py.File(r'{0}/snap_{1}.{2}.hdf5'.format(HDF5_SNAP_DEST, SNAP, snap_run), 'r')
        g = h5py.File(r'{0}/fof_subhalo_tab_{1}.{2}.hdf5'.format(HDF5_GROUP_DEST, SNAP, snap_run), 'r')
        if 'Group/GroupLenType' in g:
            to_be_appended = [np.float32(g['Group/GroupLenType'][i,1]) for i in range(g['Group/GroupLenType'].shape[0])]
            fof_dm_sizes.append(to_be_appended)
            nb_shs.append([np.float32(g['Group/GroupNsubs'][i]) for i in range(g['Group/GroupNsubs'].shape[0])])
            fof_x = np.hstack((fof_x, np.float32(g['Group/GroupCM'][:,0]/1000)))
            fof_y = np.hstack((fof_y, np.float32(g['Group/GroupCM'][:,1]/1000)))
            fof_z = np.hstack((fof_z, np.float32(g['Group/GroupCM'][:,2]/1000)))
            fof_masses = np.hstack((fof_masses, np.float32(g['Group/GroupMassType'][:,1])))
        if 'Subhalo/SubhaloLenType' in f:
            sh_len.append([np.float32(g['Subhalo/SubhaloLenType'][i,1]) for i in range(g['Subhalo/SubhaloLenType'].shape[0])])
            sh_x = np.hstack((sh_x, np.float32(g['Subhalo/SubhaloCM'][:,0]/1000))) 
            sh_y = np.hstack((sh_y, np.float32(g['Subhalo/SubhaloCM'][:,1]/1000))) 
            sh_z = np.hstack((sh_z, np.float32(g['Subhalo/SubhaloCM'][:,2]/1000))) 
        dm_x = np.hstack((dm_x, np.float32(f['PartType1/Coordinates'][:,0]/1000))) # in Mpc = 3.085678e+27 cm
        dm_y = np.hstack((dm_y, np.float32(f['PartType1/Coordinates'][:,1]/1000))) 
        dm_z = np.hstack((dm_z, np.float32(f['PartType1/Coordinates'][:,2]/1000))) 
        if 'PartType4/Coordinates' in f:
            is_star = np.where(f['PartType4/GFM_StellarFormationTime'][:]>0)[0] # Discard wind particles
            star_x = np.hstack((star_x, np.float32(f['PartType4/Coordinates'][is_star][:,0]/1000))) # in Mpc = 3.085678e+27 cm
            star_y = np.hstack((star_y, np.float32(f['PartType4/Coordinates'][is_star][:,1]/1000))) 
            star_z = np.hstack((star_z, np.float32(f['PartType4/Coordinates'][is_star][:,2]/1000))) 
            star_masses = np.hstack((star_masses, np.float32(f['PartType4/Masses'][is_star][:]))) # in 1.989e+43 g
        
    dm_xyz = np.hstack((np.reshape(dm_x, (dm_x.shape[0],1)), np.reshape(dm_y, (dm_y.shape[0],1)), np.reshape(dm_z, (dm_z.shape[0],1))))
    fof_com = np.hstack((np.reshape(fof_x, (fof_x.shape[0],1)), np.reshape(fof_y, (fof_y.shape[0],1)), np.reshape(fof_z, (fof_z.shape[0],1))))
    dm_masses = np.ones((dm_xyz.shape[0],), dtype=np.float32)*np.float32(f['Header'].attrs['MassTable'][1]) # in 1.989e+43 g
    sh_com = np.hstack((np.reshape(sh_x, (sh_x.shape[0],1)), np.reshape(sh_y, (sh_y.shape[0],1)), np.reshape(sh_z, (sh_z.shape[0],1))))
    fof_dm_sizes = list(itertools.chain.from_iterable(fof_dm_sizes)) # Simple list, not nested list
    nb_shs = list(itertools.chain.from_iterable(nb_shs))
    sh_len = list(itertools.chain.from_iterable(sh_len))
    star_xyz = np.hstack((np.reshape(star_x, (star_x.shape[0],1)), np.reshape(star_y, (star_y.shape[0],1)), np.reshape(star_z, (star_z.shape[0],1))))

    return dm_xyz, star_xyz, sh_com, nb_shs, sh_len, fof_dm_sizes, dm_masses, star_masses, fof_com, fof_masses

@np_cache_factory(0, 0)
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
    fof_x = np.empty(0, dtype = np.float32)
    fof_y = np.empty(0, dtype = np.float32)
    fof_z = np.empty(0, dtype = np.float32)
    sh_x = np.empty(0, dtype = np.float32)
    sh_y = np.empty(0, dtype = np.float32)
    sh_z = np.empty(0, dtype = np.float32)
    is_star = np.empty(0, dtype = bool)
    star_masses = np.empty(0, dtype = np.float32)
    star_x = np.empty(0, dtype = np.float32)
    star_y = np.empty(0, dtype = np.float32)
    star_z = np.empty(0, dtype = np.float32)
    star_velx = np.empty(0, dtype = np.float32)
    star_vely = np.empty(0, dtype = np.float32)
    star_velz = np.empty(0, dtype = np.float32)
    nb_shs_l = []
    nb_jobs_to_do = SNAP_MAX
    perrank = nb_jobs_to_do//size + (nb_jobs_to_do//size == 0)*1
    do_sth = rank <= nb_jobs_to_do-1
    count = 0
    count_fof = 0
    count_sh = 0
    count_sh_l = 0
    if size <= nb_jobs_to_do:
        last = rank == size - 1 # Whether or not last process
    else:
        last = rank == nb_jobs_to_do - 1
    for snap_run in range(rank*perrank, rank*perrank+do_sth*(perrank+last*(nb_jobs_to_do-(rank+1)*perrank))):
        f = h5py.File(r'{0}/snap_{1}.{2}.hdf5'.format(HDF5_SNAP_DEST, SNAP, snap_run), 'r')
        g = h5py.File(r'{0}/fof_subhalo_tab_{1}.{2}.hdf5'.format(HDF5_GROUP_DEST, SNAP, snap_run), 'r')
        if 'Group/GroupCM' in g:
            fof_x = np.hstack((fof_x, np.float32(g['Group/GroupCM'][:,0]/1000)))
            fof_y = np.hstack((fof_y, np.float32(g['Group/GroupCM'][:,1]/1000)))
            fof_z = np.hstack((fof_z, np.float32(g['Group/GroupCM'][:,2]/1000)))
            nb_shs_l.append([np.int32(g['Group/GroupNsubs'][i]) for i in range(g['Group/GroupNsubs'].shape[0])])
            count_sh_l += 1
            count_fof += g['Group/GroupCM'][:].shape[0]
        if 'PartType4/Coordinates' in f:
            is_star = np.hstack((is_star, f['PartType4/GFM_StellarFormationTime'][:] > 0)) # Discard wind particles
            star_x = np.hstack((star_x, np.float32(f['PartType4/Coordinates'][:,0]/1000))) # in Mpc = 3.085678e+27 cm
            star_y = np.hstack((star_y, np.float32(f['PartType4/Coordinates'][:,1]/1000))) 
            star_z = np.hstack((star_z, np.float32(f['PartType4/Coordinates'][:,2]/1000))) 
            star_velx = np.hstack((star_velx, np.float32(f['PartType4/Velocities'][:,0]))) # in km/s
            star_vely = np.hstack((star_vely, np.float32(f['PartType4/Velocities'][:,1]))) 
            star_velz = np.hstack((star_velz, np.float32(f['PartType4/Velocities'][:,2]))) 
            star_masses = np.hstack((star_masses, np.float32(f['PartType4/Masses'][:]))) # in 1.989e+43 g
            count += f['PartType4/Coordinates'][:].shape[0]
        if 'Subhalo/SubhaloCM' in g:
            sh_x = np.hstack((sh_x, np.float32(g['Subhalo/SubhaloCM'][:,0]/1000))) 
            sh_y = np.hstack((sh_y, np.float32(g['Subhalo/SubhaloCM'][:,1]/1000))) 
            sh_z = np.hstack((sh_z, np.float32(g['Subhalo/SubhaloCM'][:,2]/1000)))
            count_sh += g['Subhalo/SubhaloCM'][:].shape[0]
                
    count_new = comm.gather(count, root=0)
    count_new = comm.bcast(count_new, root = 0)
    nb_star_ptcs = np.sum(np.array(count_new))
    count_new_fof = comm.gather(count_fof, root=0)
    count_new_fof = comm.bcast(count_new_fof, root = 0)
    nb_fofs = np.sum(np.array(count_new_fof))
    count_new_sh = comm.gather(count_sh, root=0)
    count_new_sh = comm.bcast(count_new_sh, root = 0)
    nb_shs = np.sum(np.array(count_new_sh))
    count_new_sh_l = comm.gather(count_sh_l, root=0)
    nb_shs_l = comm.gather(nb_shs_l, root=0)
    comm.Barrier()
    
    recvcounts = np.array(count_new)
    rdispls = np.zeros_like(recvcounts)
    for j in range(rdispls.shape[0]):
        rdispls[j] = np.sum(recvcounts[:j])
    recvcounts_fof = np.array(count_new_fof)
    rdispls_fof = np.zeros_like(recvcounts_fof)
    for j in range(rdispls_fof.shape[0]):
        rdispls_fof[j] = np.sum(recvcounts_fof[:j])
    recvcounts_sh = np.array(count_new_sh)
    rdispls_sh = np.zeros_like(recvcounts_sh)
    for j in range(rdispls_sh.shape[0]):
        rdispls_sh[j] = np.sum(recvcounts_sh[:j])
    fof_x_total = np.empty(nb_fofs, dtype = np.float32)
    fof_y_total = np.empty(nb_fofs, dtype = np.float32)
    fof_z_total = np.empty(nb_fofs, dtype = np.float32)
    sh_x_total = np.empty(nb_shs, dtype = np.float32)
    sh_y_total = np.empty(nb_shs, dtype = np.float32)
    sh_z_total = np.empty(nb_shs, dtype = np.float32)
    star_x_total = np.empty(nb_star_ptcs, dtype = np.float32)
    star_y_total = np.empty(nb_star_ptcs, dtype = np.float32)
    star_z_total = np.empty(nb_star_ptcs, dtype = np.float32)
    star_velx_total = np.empty(nb_star_ptcs, dtype = np.float32)
    star_vely_total = np.empty(nb_star_ptcs, dtype = np.float32)
    star_velz_total = np.empty(nb_star_ptcs, dtype = np.float32)
    is_star_total = np.empty(nb_star_ptcs, dtype = np.float32)
    star_masses_total = np.empty(nb_star_ptcs, dtype = np.float32)

    comm.Gatherv(fof_x, [fof_x_total, recvcounts_fof, rdispls_fof, MPI.FLOAT], root = 0)
    comm.Gatherv(fof_y, [fof_y_total, recvcounts_fof, rdispls_fof, MPI.FLOAT], root = 0)
    comm.Gatherv(fof_z, [fof_z_total, recvcounts_fof, rdispls_fof, MPI.FLOAT], root = 0)
    comm.Gatherv(sh_x, [sh_x_total, recvcounts_sh, rdispls_sh, MPI.FLOAT], root = 0)
    comm.Gatherv(sh_y, [sh_y_total, recvcounts_sh, rdispls_sh, MPI.FLOAT], root = 0)
    comm.Gatherv(sh_z, [sh_z_total, recvcounts_sh, rdispls_sh, MPI.FLOAT], root = 0)
    comm.Gatherv(star_x, [star_x_total, recvcounts, rdispls, MPI.FLOAT], root = 0)
    comm.Gatherv(star_y, [star_y_total, recvcounts, rdispls, MPI.FLOAT], root = 0)
    comm.Gatherv(star_z, [star_z_total, recvcounts, rdispls, MPI.FLOAT], root = 0)
    comm.Gatherv(star_velx, [star_velx_total, recvcounts, rdispls, MPI.FLOAT], root = 0)
    comm.Gatherv(star_vely, [star_vely_total, recvcounts, rdispls, MPI.FLOAT], root = 0)
    comm.Gatherv(star_velz, [star_velz_total, recvcounts, rdispls, MPI.FLOAT], root = 0)
    comm.Gatherv(is_star, [is_star_total, recvcounts, rdispls, MPI.BOOL], root = 0)
    comm.Gatherv(star_masses, [star_masses_total, recvcounts, rdispls, MPI.FLOAT], root = 0)
    
    pieces = 1 + (nb_fofs>=3*10**8)*nb_fofs//(3*10**8) # Not too high since this is a slow-down!
    chunk = nb_fofs//pieces
    fof_x = np.empty(0, dtype = np.float32)
    fof_y = np.empty(0, dtype = np.float32)
    fof_z = np.empty(0, dtype = np.float32)
    sh_x = np.empty(0, dtype = np.float32)
    sh_y = np.empty(0, dtype = np.float32)
    sh_z = np.empty(0, dtype = np.float32)
    for i in range(pieces):
        to_bcast = fof_x_total[i*chunk:(i+1)*chunk+(i==(pieces-1))*(nb_fofs-pieces*chunk)]
        comm.Bcast(to_bcast, root=0)
        fof_x = np.hstack((fof_x, to_bcast))
        to_bcast = fof_y_total[i*chunk:(i+1)*chunk+(i==(pieces-1))*(nb_fofs-pieces*chunk)]
        comm.Bcast(to_bcast, root=0)
        fof_y = np.hstack((fof_y, to_bcast))
        to_bcast = fof_z_total[i*chunk:(i+1)*chunk+(i==(pieces-1))*(nb_fofs-pieces*chunk)]
        comm.Bcast(to_bcast, root=0)
        fof_z = np.hstack((fof_z, to_bcast))
    pieces = 1 + (nb_shs>=3*10**8)*nb_shs//(3*10**8) # Not too high since this is a slow-down!
    chunk = nb_shs//pieces
    sh_x = np.empty(0, dtype = np.float32)
    sh_y = np.empty(0, dtype = np.float32)
    sh_z = np.empty(0, dtype = np.float32)
    for i in range(pieces):
        to_bcast = sh_x_total[i*chunk:(i+1)*chunk+(i==(pieces-1))*(nb_shs-pieces*chunk)]
        comm.Bcast(to_bcast, root=0)
        sh_x = np.hstack((sh_x, to_bcast))
        to_bcast = sh_y_total[i*chunk:(i+1)*chunk+(i==(pieces-1))*(nb_shs-pieces*chunk)]
        comm.Bcast(to_bcast, root=0)
        sh_y = np.hstack((sh_y, to_bcast))
        to_bcast = sh_z_total[i*chunk:(i+1)*chunk+(i==(pieces-1))*(nb_shs-pieces*chunk)]
        comm.Bcast(to_bcast, root=0)
        sh_z = np.hstack((sh_z, to_bcast))
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
    
    if rank == 0:
        nb_shs_l = [nb_shs_l[i][j] for i in range(size) for j in range(count_new_sh_l[i])]
    nb_shs_l = comm.bcast(nb_shs_l, root = 0)

    fof_com = np.hstack((np.reshape(fof_x, (fof_x.shape[0],1)), np.reshape(fof_y, (fof_y.shape[0],1)), np.reshape(fof_z, (fof_z.shape[0],1))))
    sh_com = np.hstack((np.reshape(sh_x, (sh_x.shape[0],1)), np.reshape(sh_y, (sh_y.shape[0],1)), np.reshape(sh_z, (sh_z.shape[0],1))))
    star_xyz = np.hstack((np.reshape(star_x, (star_x.shape[0],1)), np.reshape(star_y, (star_y.shape[0],1)), np.reshape(star_z, (star_z.shape[0],1))))
    star_velxyz = np.hstack((np.reshape(star_velx, (star_velx.shape[0],1)), np.reshape(star_vely, (star_vely.shape[0],1)), np.reshape(star_velz, (star_velz.shape[0],1))))
    nb_shs_l = list(itertools.chain.from_iterable(nb_shs_l))

    return star_xyz, fof_com, sh_com, nb_shs_l, star_masses, star_velxyz, is_star

@np_cache_factory(0, 0)
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
    
    fof_dm_sizes = []
    nb_shs = []
    sh_len = []
    group_r200 = np.empty(0, dtype = np.float32)
    fof_masses = np.empty(0, dtype = np.float32)
    fof_com = np.empty(0, dtype = np.float32)
    for snap_run in range(SNAP_MAX):
        g = h5py.File(r'{0}/fof_subhalo_tab_{1}.{2}.hdf5'.format(HDF5_GROUP_DEST, SNAP, snap_run), 'r')
        if 'Group/GroupLenType' in g:
            to_be_appended = [np.int32(g['Group/GroupLenType'][i,1]) for i in range(g['Group/GroupLenType'].shape[0])]
            if WANT_RVIR:
                group_r200 = np.hstack((group_r200, np.float32(g['Group/Group_R_TopHat200'][:]/1000)))
            else:
                group_r200 = np.hstack((group_r200, np.float32(g['Group/Group_R_Mean200'][:]/1000)))
            fof_masses = np.hstack((fof_masses, np.float32(g['Group/GroupMassType'][:,1])))
            fof_com = np.hstack((fof_com, np.float32(g['Group/GroupCM'][:]/1000).flatten()))
            nb_shs.append([np.int32(g['Group/GroupNsubs'][i]) for i in range(g['Group/GroupNsubs'].shape[0])])
        else:
            to_be_appended = []
        if 'Subhalo/SubhaloLenType' in g:
            sh_len.append([np.int32(g['Subhalo/SubhaloLenType'][i,1]) for i in range(g['Subhalo/SubhaloLenType'].shape[0])])
        fof_dm_sizes.append(to_be_appended)
    fof_dm_sizes = list(itertools.chain.from_iterable(fof_dm_sizes)) # Simple list, not nested list
    nb_shs = list(itertools.chain.from_iterable(nb_shs))
    sh_len = list(itertools.chain.from_iterable(sh_len))

    return nb_shs, sh_len, fof_dm_sizes, group_r200, fof_masses, fof_com

@np_cache_factory(0, 0)
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
    
    nb_shs = []
    fof_gx_sizes = []
    sh_len_gx = []
    for snap_run in range(SNAP_MAX):
        g = h5py.File(r'{0}/fof_subhalo_tab_{1}.{2}.hdf5'.format(HDF5_GROUP_DEST, SNAP, snap_run), 'r')
        if 'Group/GroupLenType' in g:
            nb_shs.append([np.int32(g['Group/GroupNsubs'][i]) for i in range(g['Group/GroupNsubs'].shape[0])])
            to_be_appended = [np.int32(g['Group/GroupLenType'][i,4]) for i in range(g['Group/GroupLenType'].shape[0])]
        else:
            to_be_appended = []
        if 'Subhalo/SubhaloLenType' in g:
            sh_len_gx.append([np.int32(g['Subhalo/SubhaloLenType'][i,4]) for i in range(g['Subhalo/SubhaloLenType'].shape[0])])
        fof_gx_sizes.append(to_be_appended)
    nb_shs = list(itertools.chain.from_iterable(nb_shs))
    fof_gx_sizes = list(itertools.chain.from_iterable(fof_gx_sizes)) # Simple list, not nested list
    sh_len_gx = list(itertools.chain.from_iterable(sh_len_gx))

    return nb_shs, sh_len_gx, fof_gx_sizes

@np_cache_factory(0, 0)
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
    
    dm_x = np.empty(0, dtype = np.float32)
    dm_y = np.empty(0, dtype = np.float32)
    dm_z = np.empty(0, dtype = np.float32)
    dm_velx = np.empty(0, dtype = np.float32)
    dm_vely = np.empty(0, dtype = np.float32)
    dm_velz = np.empty(0, dtype = np.float32)
    dm_masses = np.empty(0, dtype = np.float32)
    nb_jobs_to_do = SNAP_MAX
    perrank = nb_jobs_to_do//size + (nb_jobs_to_do//size == 0)*1
    do_sth = rank <= nb_jobs_to_do-1
    count = 0
    if size <= nb_jobs_to_do:
        last = rank == size - 1 # Whether or not last process
    else:
        last = rank == nb_jobs_to_do - 1
    for snap_run in range(rank*perrank, rank*perrank+do_sth*(perrank+last*(nb_jobs_to_do-(rank+1)*perrank))):
        f = h5py.File(r'{0}/snap_{1}.{2}.hdf5'.format(HDF5_SNAP_DEST, SNAP, snap_run), 'r')
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