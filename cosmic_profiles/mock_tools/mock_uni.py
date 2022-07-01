#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
import numpy as np
from cosmic_profiles.common.python_routines import getMassDMParticle
from nbodykit.lab import cosmology, LogNormalCatalog

def createLogNormUni(BoxSize, nbar, redshift, Nmesh, UNIT_MASS):
    """ Create mock simulation box by Poisson-sampling a lognormal density distribution
    
    The Poisson-sampled distribution is evolved according to the Zeldovich (1LPT) prescription
    up until redshift ``redshift`` under the constraint of an 'EisensteinHu' power spectrum
    
    :param BoxSize: size of to-be-obtained simulation box
    :type BoxSize: float
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
        
        cat = LogNormalCatalog(Plin=Plin, nbar=nbar, BoxSize=BoxSize, Nmesh=Nmesh, bias=2.0, seed=42)
        x_vec = np.float32(np.array(cat['Position'][:,0])) # Mpc/h
        y_vec = np.float32(np.array(cat['Position'][:,1]))
        z_vec = np.float32(np.array(cat['Position'][:,2]))
        
        x_vel = np.float32(np.array(cat['Velocity'][:,0]))
        y_vel = np.float32(np.array(cat['Velocity'][:,1]))
        z_vel = np.float32(np.array(cat['Velocity'][:,2]))
        
        N = int(round(len(x_vec)**(1/3)))
        N_tot = len(x_vec)
        dm_mass = getMassDMParticle(N, BoxSize)/UNIT_MASS
        return N_tot, x_vec, y_vec, z_vec, x_vel, y_vel, z_vel, np.ones((len(x_vec),),dtype = np.float32)*dm_mass
    else:
        return None, None, None, None, None, None, None, None
