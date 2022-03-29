#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 12:12:12 2022

@author: tibor
"""

import numpy as np
import itertools
import h5py
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def getHDF5Data(HDF5_SNAP_DEST, HDF5_GROUP_DEST, SNAP_MAX, SNAP):
    """ Retrieve all relevant HDF5 data, both DM and stars"""
        
    sh_x = np.empty(0, dtype = np.float32)
    sh_y = np.empty(0, dtype = np.float32)
    sh_z = np.empty(0, dtype = np.float32)
    dm_x = np.empty(0, dtype = np.float32)
    dm_y = np.empty(0, dtype = np.float32)
    dm_z = np.empty(0, dtype = np.float32)
    dm_smoothing = np.empty(0, dtype = np.float32)
    star_masses = np.empty(0, dtype = np.float32)
    star_smoothing = np.empty(0, dtype = np.float32)
    star_x = np.empty(0, dtype = np.float32)
    star_y = np.empty(0, dtype = np.float32)
    star_z = np.empty(0, dtype = np.float32)
    group_x = np.empty(0, dtype = np.float32)
    group_y = np.empty(0, dtype = np.float32)
    group_z = np.empty(0, dtype = np.float32)
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
            group_x = np.hstack((group_x, np.float32(g['Group/GroupCM'][:,0]/1000)))
            group_y = np.hstack((group_y, np.float32(g['Group/GroupCM'][:,1]/1000)))
            group_z = np.hstack((group_z, np.float32(g['Group/GroupCM'][:,2]/1000)))
            fof_masses = np.hstack((fof_masses, np.float32(g['Group/GroupMassType'][:,1])))
        if 'Subhalo/SubhaloLenType' in f:
            sh_len.append([np.float32(g['Subhalo/SubhaloLenType'][i,1]) for i in range(g['Subhalo/SubhaloLenType'].shape[0])])
            sh_x = np.hstack((sh_x, np.float32(g['Subhalo/SubhaloCM'][:,0]/1000))) 
            sh_y = np.hstack((sh_y, np.float32(g['Subhalo/SubhaloCM'][:,1]/1000))) 
            sh_z = np.hstack((sh_z, np.float32(g['Subhalo/SubhaloCM'][:,2]/1000))) 
        dm_x = np.hstack((dm_x, np.float32(f['PartType1/Coordinates'][:,0]/1000))) # in Mpc = 3.085678e+27 cm
        dm_y = np.hstack((dm_y, np.float32(f['PartType1/Coordinates'][:,1]/1000))) 
        dm_z = np.hstack((dm_z, np.float32(f['PartType1/Coordinates'][:,2]/1000))) 
        dm_smoothing = np.hstack((dm_smoothing, np.float32(f['PartType1/SubfindHsml'][:]/1000))) 
        if 'PartType4/Coordinates' in f:
            is_star = np.where(f['PartType4/GFM_StellarFormationTime'][:]>0)[0] # Discard wind particles
            star_x = np.hstack((star_x, np.float32(f['PartType4/Coordinates'][is_star][:,0]/1000))) # in Mpc = 3.085678e+27 cm
            star_y = np.hstack((star_y, np.float32(f['PartType4/Coordinates'][is_star][:,1]/1000))) 
            star_z = np.hstack((star_z, np.float32(f['PartType4/Coordinates'][is_star][:,2]/1000))) 
            star_smoothing = np.hstack((star_smoothing, np.float32(f['PartType4/SubfindHsml'][is_star][:]/1000))) # in Mpc = 3.085678e+27 cm
            star_masses = np.hstack((star_masses, np.float32(f['PartType4/Masses'][is_star][:]))) # in 1.989e+43 g
        
    dm_xyz = np.hstack((np.reshape(dm_x, (dm_x.shape[0],1)), np.reshape(dm_y, (dm_y.shape[0],1)), np.reshape(dm_z, (dm_z.shape[0],1))))
    group_xyz = np.hstack((np.reshape(group_x, (group_x.shape[0],1)), np.reshape(group_y, (group_y.shape[0],1)), np.reshape(group_z, (group_z.shape[0],1))))
    dm_masses = np.ones((dm_xyz.shape[0],), dtype=np.float32)*np.float32(f['Header'].attrs['MassTable'][1]) # in 1.989e+43 g
    sh_com = np.hstack((np.reshape(sh_x, (sh_x.shape[0],1)), np.reshape(sh_y, (sh_y.shape[0],1)), np.reshape(sh_z, (sh_z.shape[0],1))))
    fof_dm_sizes = list(itertools.chain.from_iterable(fof_dm_sizes)) # Simple list, not nested list
    nb_shs = list(itertools.chain.from_iterable(nb_shs))
    sh_len = list(itertools.chain.from_iterable(sh_len))
    star_xyz = np.hstack((np.reshape(star_x, (star_x.shape[0],1)), np.reshape(star_y, (star_y.shape[0],1)), np.reshape(star_z, (star_z.shape[0],1))))

    return dm_xyz, star_xyz, sh_com, nb_shs, sh_len, fof_dm_sizes, dm_masses, dm_smoothing, star_masses, star_smoothing, group_xyz, fof_masses

def getHDF5GxData(HDF5_SNAP_DEST, HDF5_GROUP_DEST, SNAP_MAX, SNAP):
    """ Retrieve all relevant HDF5 data, Type4: stars only"""
    fof_x = np.empty(0, dtype = np.float32)
    fof_y = np.empty(0, dtype = np.float32)
    fof_z = np.empty(0, dtype = np.float32)
    sh_x = np.empty(0, dtype = np.float32)
    sh_y = np.empty(0, dtype = np.float32)
    sh_z = np.empty(0, dtype = np.float32)
    star_masses = np.empty(0, dtype = np.float32)
    star_smoothing = np.empty(0, dtype = np.float32)
    star_x = np.empty(0, dtype = np.float32)
    star_y = np.empty(0, dtype = np.float32)
    star_z = np.empty(0, dtype = np.float32)
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
            is_star = np.where(f['PartType4/GFM_StellarFormationTime'][:]>0)[0] # Discard wind particles
            star_x = np.hstack((star_x, np.float32(f['PartType4/Coordinates'][is_star][:,0]/1000))) # in Mpc = 3.085678e+27 cm
            star_y = np.hstack((star_y, np.float32(f['PartType4/Coordinates'][is_star][:,1]/1000))) 
            star_z = np.hstack((star_z, np.float32(f['PartType4/Coordinates'][is_star][:,2]/1000))) 
            star_smoothing = np.hstack((star_smoothing, np.float32(f['PartType4/SubfindHsml'][is_star][:]/1000))) # in Mpc = 3.085678e+27 cm
            star_masses = np.hstack((star_masses, np.float32(f['PartType4/Masses'][is_star][:]))) # in 1.989e+43 g
            count += f['PartType4/Coordinates'][is_star][:].shape[0]
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
    star_smoothing_total = np.empty(nb_star_ptcs, dtype = np.float32)
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
    comm.Gatherv(star_smoothing, [star_smoothing_total, recvcounts, rdispls, MPI.FLOAT], root = 0)
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
    star_masses = np.empty(0, dtype = np.float32)
    star_smoothing = np.empty(0, dtype = np.float32)
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
        to_bcast = star_masses_total[i*chunk:(i+1)*chunk+(i==(pieces-1))*(nb_star_ptcs-pieces*chunk)]
        comm.Bcast(to_bcast, root=0)
        star_masses = np.hstack((star_masses, to_bcast))
        to_bcast = star_smoothing_total[i*chunk:(i+1)*chunk+(i==(pieces-1))*(nb_star_ptcs-pieces*chunk)]
        comm.Bcast(to_bcast, root=0)
        star_smoothing = np.hstack((star_smoothing, to_bcast))
    
    if rank == 0:
        nb_shs_l = [nb_shs_l[i][j] for i in range(size) for j in range(count_new_sh_l[i])]
    nb_shs_l = comm.bcast(nb_shs_l, root = 0)

    fof_com = np.hstack((np.reshape(fof_x, (fof_x.shape[0],1)), np.reshape(fof_y, (fof_y.shape[0],1)), np.reshape(fof_z, (fof_z.shape[0],1))))
    sh_com = np.hstack((np.reshape(sh_x, (sh_x.shape[0],1)), np.reshape(sh_y, (sh_y.shape[0],1)), np.reshape(sh_z, (sh_z.shape[0],1))))
    star_xyz = np.hstack((np.reshape(star_x, (star_x.shape[0],1)), np.reshape(star_y, (star_y.shape[0],1)), np.reshape(star_z, (star_z.shape[0],1))))
    nb_shs_l = list(itertools.chain.from_iterable(nb_shs_l))

    return star_xyz, fof_com, sh_com, nb_shs_l, star_masses, star_smoothing

    
def getHDF5SHData(HDF5_GROUP_DEST, SNAP_MAX, SNAP):
    """ Retrieve all relevant FOF/SH data"""
    
    fof_dm_sizes = []
    nb_shs = []
    sh_len = []
    group_r200 = np.empty(0, dtype = np.float32)
    fof_masses = np.empty(0, dtype = np.float32)
    fof_coms = np.empty(0, dtype = np.float32)
    for snap_run in range(SNAP_MAX):
        g = h5py.File(r'{0}/fof_subhalo_tab_{1}.{2}.hdf5'.format(HDF5_GROUP_DEST, SNAP, snap_run), 'r')
        if 'Group/GroupLenType' in g:
            to_be_appended = [np.int32(g['Group/GroupLenType'][i,1]) for i in range(g['Group/GroupLenType'].shape[0])]
            group_r200 = np.hstack((group_r200, np.float32(g['Group/Group_R_Mean200'][:]/1000)))
            fof_masses = np.hstack((fof_masses, np.float32(g['Group/GroupMassType'][:,1])))
            fof_coms = np.hstack((fof_coms, np.float32(g['Group/GroupCM'][:]/1000).flatten()))
            nb_shs.append([np.int32(g['Group/GroupNsubs'][i]) for i in range(g['Group/GroupNsubs'].shape[0])])
        else:
            to_be_appended = []
        if 'Subhalo/SubhaloLenType' in g:
            sh_len.append([np.int32(g['Subhalo/SubhaloLenType'][i,1]) for i in range(g['Subhalo/SubhaloLenType'].shape[0])])
        fof_dm_sizes.append(to_be_appended)
    fof_dm_sizes = list(itertools.chain.from_iterable(fof_dm_sizes)) # Simple list, not nested list
    nb_shs = list(itertools.chain.from_iterable(nb_shs))
    sh_len = list(itertools.chain.from_iterable(sh_len))

    return nb_shs, sh_len, fof_dm_sizes, group_r200, fof_masses, fof_coms

def getHDF5DMData(HDF5_SNAP_DEST, SNAP_MAX, SNAP):
    """ Retrieve all relevant HDF5 data, Type1: DM only"""
    
    dm_x = np.empty(0, dtype = np.float32)
    dm_y = np.empty(0, dtype = np.float32)
    dm_z = np.empty(0, dtype = np.float32)
    dm_velx = np.empty(0, dtype = np.float32)
    dm_vely = np.empty(0, dtype = np.float32)
    dm_velz = np.empty(0, dtype = np.float32)
    dm_smoothing = np.empty(0, dtype = np.float32)
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
        dm_smoothing = np.hstack((dm_smoothing, np.float32(f['PartType1/SubfindHsml'][:]/1000)))
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
    dm_smoothing_total = np.empty(nb_dm_ptcs, dtype = np.float32)
    dm_masses_total = np.empty(nb_dm_ptcs, dtype = np.float32)
    
    comm.Gatherv(dm_x, [dm_x_total, recvcounts, rdispls, MPI.FLOAT], root = 0)
    comm.Gatherv(dm_y, [dm_y_total, recvcounts, rdispls, MPI.FLOAT], root = 0)
    comm.Gatherv(dm_z, [dm_z_total, recvcounts, rdispls, MPI.FLOAT], root = 0)
    comm.Gatherv(dm_velx, [dm_velx_total, recvcounts, rdispls, MPI.FLOAT], root = 0)
    comm.Gatherv(dm_vely, [dm_vely_total, recvcounts, rdispls, MPI.FLOAT], root = 0)
    comm.Gatherv(dm_velz, [dm_velz_total, recvcounts, rdispls, MPI.FLOAT], root = 0)
    comm.Gatherv(dm_smoothing, [dm_smoothing_total, recvcounts, rdispls, MPI.FLOAT], root = 0)
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
    dm_smoothing = np.empty(0, dtype = np.float32)
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
        to_bcast = dm_smoothing_total[i*chunk:(i+1)*chunk+(i==(pieces-1))*(nb_dm_ptcs-pieces*chunk)]
        comm.Bcast(to_bcast, root=0)
        dm_smoothing = np.hstack((dm_smoothing, to_bcast))

    dm_xyz = np.hstack((np.reshape(dm_x, (dm_x.shape[0],1)), np.reshape(dm_y, (dm_y.shape[0],1)), np.reshape(dm_z, (dm_z.shape[0],1))))
    dm_velxyz = np.hstack((np.reshape(dm_velx, (dm_velx.shape[0],1)), np.reshape(dm_vely, (dm_vely.shape[0],1)), np.reshape(dm_velz, (dm_velz.shape[0],1))))

    return dm_xyz, dm_masses, dm_smoothing, dm_velxyz