#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def genHalo(tot_mass, res, model_pars, method, a, b, c):
    """ Mock halo generator
    
    Create mock halo of mass ``tot_mass`` consisting of approximately ``res`` particles. The ``model_pars``
    array contains the parameters for the profile model given in ``method``. 
    
    :param tot_mass: total target mass of halo, in units of M_sun*h^2/Mpc^3
    :type tot_mass: float
    :param res: halo resolution
    :type res: int
    :param model_pars: parameters (except for ``rho_s`` which will be deduced from ``tot_mass``)
        in density profile model
    :type model_pars: dictionary of length 4, 2 or 1
    :param a: major axis array
    :type a: float array, units are Mpc/h
    :param b: intermediate axis array
    :type b: float array, units are Mpc/h
    :param c: minor axis array
    :type c: float array, units are Mpc/h
    :return: halo_x, halo_y, halo_z: arrays containing positions of halo particles, 
        mass_ptc: mass of each DM ptc in units of M_sun/h, rho_s: ``rho_s`` parameter in profile model
    :rtype: 3 (N,) float arrays, 2 floats
    """
    return
