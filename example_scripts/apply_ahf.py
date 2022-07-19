#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 18:23:43 2021
"""

from mpi4py import MPI
import matplotlib.pyplot as plt
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
import pynbody
import os
import sys
import h5py
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.append(os.path.join(currentdir, '..')) # Only needed if cosmic_profiles is not installed
import numpy as np
from cosmic_profiles import DensShapeProfs, getMassDMParticle
from nbodykit.lab import cosmology, LogNormalCatalog

def createLogNormUni(L_BOX, nbar, redshift, Nmesh, UNIT_MASS):
    """ Create mock simulation box by Poisson-sampling a lognormal density distribution
    
    The Poisson-sampled distribution is evolved according to the Zeldovich (1LPT) prescription
    up until redshift ``redshift`` under the constraint of an 'EisensteinHu' power spectrum
    
    :param L_BOX: size of to-be-obtained simulation box
    :type L_BOX: float
    :param nbar: number density of points (i.e. sampling density / resolution) in box, units: 1/(Mpc/h)**3
        Note: ``nbar`` is assumed to be constant across the box
    :type nbar: float
    :param redshift: redshift of interest
    :type redshift: float
    :param Nmesh: the mesh size to use when generating the density and displacement fields, 
        which are Poisson-sampled to particles
    :type Nmesh: int
    :param UNIT_MASS: in units of solar masses / h. Returned masses will have units UNIT_MASS*(solar_mass)/h
    :type UNIT_MASS: float
    :return: total number of particles, xyz-coordinates of DM particles, xyz-values of DM particle velocities, 
        masses of the DM particles (all identical)
    :rtype: int, (N,) floats, (N,) floats, (N,) floats, (N,) floats, (N,) floats, (N,) floats, (N,) floats"""
    print('Starting createLogNormUni()')
        
    if rank == 0:
        # Generating LogNormal Catalog
        redshift = redshift
        cosmo = cosmology.Planck15
        Plin = cosmology.LinearPower(cosmo, redshift, transfer='EisensteinHu')
        
        cat = LogNormalCatalog(Plin=Plin, nbar=nbar, BoxSize=L_BOX, Nmesh=Nmesh, bias=2.0, seed=42)
        x_vec = np.float32(np.array(cat['Position'][:,0])) # Mpc/h
        y_vec = np.float32(np.array(cat['Position'][:,1]))
        z_vec = np.float32(np.array(cat['Position'][:,2]))
        
        x_vel = np.float32(np.array(cat['Velocity'][:,0]))
        y_vel = np.float32(np.array(cat['Velocity'][:,1]))
        z_vel = np.float32(np.array(cat['Velocity'][:,2]))
        
        N = int(round(len(x_vec)**(1/3)))
        N_tot = len(x_vec)
        dm_mass = getMassDMParticle(N, L_BOX)/UNIT_MASS
        return N_tot, x_vec, y_vec, z_vec, x_vel, y_vel, z_vel, np.ones((len(x_vec),),dtype = np.float32)*dm_mass
    else:
        return None, None, None, None, None, None, None, None

############## Parameters ####################################################################################
L_BOX = np.float32(10) # Mpc/h
nbar = 8e+3 # If too small, e.g. 5e+3: pynbody later yields OSError: Corrupt header record. If too large, need many GBs of RAM.
Nmesh = 256
redshift = 5.5
h = 0.6774
UNIT_MASS = 10**10 # in M_sun
SNAP = '025'
D_LOGSTART = -2
D_LOGEND = 1
D_BINS = 30 # If D_LOGSTART == -2 D_LOGEND == 1, 60 corresponds to shell width of 0.05 dex
M_TOL = np.float32(1e-2)
N_WALL = 100
N_MIN = 10
MIN_NUMBER_DM_PTCS = 200
CENTER = 'mode'
CAT_DEST = "./cat"
VIZ_DEST = "./viz"
#############################################################################################################

def AHFEx():
    ############## Generate mock universe #######################################################################
    N_tot, dm_x, dm_y, dm_z, vel_x, vel_y, vel_z, mass_array = createLogNormUni(L_BOX, nbar, redshift, Nmesh, UNIT_MASS) # mass_array in 10**10 solar masses * h^-1
    #############################################################################################################
    
    
    ############## Save mock universe to Gadget 2 file #########################################################
    f = h5py.File('{}/snap_{}.hdf5'.format(CAT_DEST, SNAP), 'w')
    config = f.create_group('Config')
    header = f.create_group('Header')
    param = f.create_group('Parameters')
    ptype1 = f.create_group('PartType1')
    # Populate Config
    config.attrs['NTYPES'] = 6.0
    # Populate Header
    header.attrs['MassTable'] = np.array([0, 1, 0, 0, 0, 0])
    header.attrs['Redshift'] = redshift
    header.attrs['BoxSize'] = L_BOX # in Mpc/h
    header.attrs['Omega0'] = 0.3089 # Planck 2015
    header.attrs['OmegaBaryon'] = 0.048
    header.attrs['OmegaLambda'] = 0.6911
    header.attrs['UnitLength_in_cm'] = 3.085678e+24 # 1 Mpc
    header.attrs['UnitVelocity_in_cm_per_s'] = 100000.0 # 1 km/s
    header.attrs['UnitMass_in_g'] = 1.989e+43 # 10**10 M_sun
    header.attrs['HubbleParam'] = h
    header.attrs['Time'] = 1/(redshift+1)
    header.attrs['NumFilesPerSnapshot'] = 1
    header.attrs['NumPart_ThisFile'] = np.array([0, N_tot, 0, 0, 0, 0])
    header.attrs['NumPart_Total'] = np.array([0, N_tot, 0, 0, 0, 0])
    header.attrs['MassTable'] = np.array([0, mass_array[0], 0, 0, 0, 0])
    # Populate Parameters
    param.attrs['MassTable'] = np.array([0, mass_array[0], 0, 0, 0, 0])
    param.attrs['Redshift'] = redshift
    param.attrs['BoxSize'] = L_BOX # in Mpc/h
    param.attrs['Omega0'] = 0.3089 # Planck 2015
    param.attrs['OmegaBaryon'] = 0.048
    param.attrs['OmegaLambda'] = 0.6911
    param.attrs['UnitLength_in_cm'] = 3.085678e+24 # 1 Mpc
    param.attrs['UnitVelocity_in_cm_per_s'] = 100000.0 # 1 km/s
    param.attrs['UnitMass_in_g'] = 1.989e+43 # 10**10 M_sun
    param.attrs['HubbleParam'] = h
    param.attrs['NumFilesPerSnapshot'] = 1
    param.attrs['SofteningComovingType1'] = 0.19/1000 # Mpc
    param.attrs['SofteningMaxPhysType1'] = 0.19/1000
    param.attrs['SofteningTypeOfPartType1'] = 1
    # Populate PartType1
    dm_xyz = np.hstack((np.reshape(dm_x, (dm_x.shape[0],1)), np.reshape(dm_y, (dm_y.shape[0],1)), np.reshape(dm_z, (dm_z.shape[0],1))))
    vel_xyz = np.hstack((np.reshape(vel_x, (vel_x.shape[0],1)), np.reshape(vel_y, (vel_y.shape[0],1)), np.reshape(vel_z, (vel_z.shape[0],1))))
    ptype1_coords = ptype1.create_dataset('Coordinates', dtype = np.float32, data = dm_xyz) # Mpc/h
    ptype1_coords.attrs['a_scaling'] = 1.0
    ptype1_coords.attrs['h_scaling'] = -1.0
    ptype1_ids = ptype1.create_dataset('ParticleIDs', dtype = np.uint32, data = np.arange(N_tot, dtype = np.uint32))
    ptype1_ids.attrs['a_scaling'] = 0.0
    ptype1_ids.attrs['h_scaling'] = 0.0
    ptype1_vels = ptype1.create_dataset('Velocities', dtype = np.float32, data = vel_xyz)
    ptype1_vels.attrs['a_scaling'] = 0.5
    ptype1_vels.attrs['h_scaling'] = 0.0
    f.close()
    s = pynbody.load('{}/snap_{}.hdf5'.format(CAT_DEST, SNAP))
    s['pos']; s['mass']; s['vel']; s['iord']
    s.write(fmt=pynbody.gadget.GadgetSnap,
                    filename='{}/snap_{}'.format(CAT_DEST, SNAP))
    #############################################################################################################
    
    ############## Load Gadget 2 file with pynbody ###############################################################
    s = pynbody.load('{}/snap_{}'.format(CAT_DEST, SNAP))
    print("The snaphost properties read", s.properties)
    #############################################################################################################
    
    ############## Find halos with e.g. AHF #####################################################################
    s['eps'] = 0.01 # The smaller, the larger AHF's LgridMax: SpatialResolution ≃ BoxSize/LgridMax
    halos = s.halos() # Install e.g. AHF and place executable in ~/bin (or extend $PATH variable)
    # Modify /halo/ahf.py (Gadget, not Tipsy) and config.ini (Gadget, not Tipsy)
    #############################################################################################################
    
    ############## Viz some halos and retrieve basic properties #################################################
    print("The length of halo 1 and 2 are {} and {}, respectively.".format(len(halos[1]), len(halos[2])))
    print("The second halo has mass {} 1e12 Msol".format(halos[2]['mass'].sum().in_units('1e12 Msol')))
    print("The first 5 particles of halo 2 are located at (in kpc)", halos[2]['pos'].in_units('kpc')[:5])
    pynbody.analysis.halo.center(halos[2]) # Modify cen_size="1 kpc" argument in analysis/halo.py's center() method to cen_size="10 kpc" if resolution is too low (i.e. nbar too low)
    im = pynbody.plot.image(halos[2].d, width = '500 kpc', cmap=plt.cm.Greys, units = 'Msol kpc^-2')
    plt.savefig('{}/RhoHalo2.pdf'.format(VIZ_DEST))
    #############################################################################################################
    
    ############## Extract R_vir, halo indices and halo sizes for e.g. first 5 halos ############################
    ahf_halos = np.loadtxt('{}/snap_{}.z{}.AHF_halos'.format(CAT_DEST, SNAP, format(redshift, '.3f')), unpack=True)
    h_sizes = np.int32(ahf_halos[4])[:5] # 5th column
    r_vir = np.float32(ahf_halos[11])[:5] # 12th column
    ahf_ptcs = np.loadtxt('{}/snap_{}.z{}.AHF_particles'.format(CAT_DEST, SNAP, format(redshift, '.3f')), skiprows=2, unpack = True, dtype = int)[0]
    h_indices = [[] for i in range(len(h_sizes))]
    offset = 0
    for h_idx in range(len(h_sizes)):
        h_indices[h_idx].extend(ahf_ptcs[offset:offset+h_sizes[h_idx]]) # True indices, no plus 1 etc
        offset += h_sizes[h_idx] + 1 # Skip line containing number of particles in halo
    #############################################################################################################
    
    ############## Run cosmic_profiles: define DensShapeProfs object ############################################
    cprofiles = DensShapeProfs(dm_xyz, mass_array, h_indices, r_vir, SNAP, L_BOX, MIN_NUMBER_DM_PTCS, D_LOGSTART, D_LOGEND, D_BINS, M_TOL, N_WALL, N_MIN, CENTER)
    
    ############## Create local halo shape catalogue ############################################################
    cprofiles.dumpShapeCatLocal(CAT_DEST, reduced = True, shell_based = True)
    #############################################################################################################
    
    ############## Viz first halo ###############################################################################
    cprofiles.vizLocalShapes(obj_numbers = [0], VIZ_DEST = VIZ_DEST, reduced = True, shell_based = True)
    #############################################################################################################
    
AHFEx()
