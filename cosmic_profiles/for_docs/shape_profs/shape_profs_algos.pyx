#cython: language_level=3
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

cimport openmp
import numpy as np
cimport cython
from libc.math cimport sqrt

@cython.embedsignature(True)
def runEllShellAlgo(float[:] morph_info, float[:,:] xyz, float[:,:] xyz_princ, float[:] masses, int[:] shell, float[:] center, complex[::1,:] shape_tensor, double[::1] eigval, complex[::1,:] eigvec, float d, float delta_d, float M_TOL, int N_WALL, int N_MIN):
    """ S1 algorithm for halos/galaxies at elliptical radius ``d`` with shell width ``delta_d``
    
    Calculates the axis ratios at a distance ``d`` from the center of the entire particle distro.\n
    Note that before and during the iteration, ``d`` is defined with respect to the center of 
    the entire particle distro, not the center of the initial spherical volume as in Katz 1991.\n
    Differential version of E1.\n
    Shells can cross (except 2nd shell with 1st), and a shell is assumed to be equally thick everywhere.\n
    Whether we adopt the last assumption or let the thickness float (Tomassetti et al 2016) barely makes 
    any difference in terms of shapes found, but the convergence properties improve for the version with fixated thickness.
    For 1st shell: ``delta_d`` is ``d``
    
    :param morph_info: Array to be filled with morphological info. 1st entry: d,
        2nd entry: q, 3rd entry: s, 4th to 6th: normalized major axis, 7th to 9th: normalized intermediate axis,
        10th to 12th: normalized minor axis
    :type morph_info: (12,) floats
    :param xyz: position array
    :type xyz: (N x 3) floats
    :param xyz_princ: position arrays transformed into principal frame (varies from iteration to iteration)
    :type xyz_princ: (N x 3) floats, zeros
    :param masses: mass array
    :type masses: (N x 1) floats
    :param shell: indices of points that fall into shell (varies from iteration to iteration)
    :type shell: (N,) ints, zeros
    :param center: center of point cloud
    :type center: (3,) floats
    :param shape_tensor: shape tensor array to be filled
    :type shape_tensor: (3,3) complex, zeros
    :param eigval: eigenvalue array to be filled
    :type eigval: (3,) double, zeros
    :param eigvec: eigenvector array to be filled
    :type eigvec: (3,3) double, zeros
    :param d: distance from the center, kept fixed during iterative procedure
    :type d: float
    :param delta_d: thickness of the shell in real space (constant across shells in logarithmic space)
    :type delta_d: float
    :param M_TOL: convergence tolerance, eigenvalue fractions must differ by less than ``M_TOL``
        for iteration to stop
    :type M_TOL: float
    :param N_WALL: maximum permissible number of iterations
    :type N_WALL: float
    :param N_MIN: minimum number of particles (DM or star particle) in any iteration; 
        if undercut, shape is unclassified
    :type N_MIN: int
    :param CENTER: shape quantities will be calculated with respect to CENTER = 'mode' (point of highest density)
        or 'com' (center of mass) of each halo
    :type CENTER: str
    :return: ``morph_info`` containing d, q, s, eigframe info
    :rtype: (12,) float array"""
    return


@cython.embedsignature(True)
def runEllAlgo(float[:] morph_info, float[:,:] xyz, float[:,:] xyz_princ, float[:] masses, int[:] ellipsoid, float[:] center, complex[::1,:] shape_tensor, double[::1] eigval, complex[::1,:] eigvec, float d, float M_TOL, int N_WALL, int N_MIN):
    """ Katz-Dubinski ellipsoid-based algorithm for halos/galaxies at elliptical radius ``d``
    
    Calculates the axis ratios at a distance ``d`` from the center of the entire particle distro.\n
    Note that before and during the iteration, ``d`` is defined with respect to the center of 
    the entire particle distro, not the center of the initial spherical volume as in Katz 1991.\n
    
    :param morph_info: Array to be filled with morphological info. 1st entry: d,
        2nd entry: q, 3rd entry: s, 4th to 6th: normalized major axis, 7th to 9th: normalized intermediate axis,
        10th to 12th: normalized minor axis
    :type morph_info: (12,) floats
    :param xyz: position array
    :type xyz: (N x 3) floats
    :param xyz_princ: position arrays transformed into principal frame (varies from iteration to iteration)
    :type xyz_princ: (N x 3) floats, zeros
    :param masses: mass array
    :type masses: (N x 1) floats
    :param ellipsoid: indices of points that fall into ellipsoid (varies from iteration to iteration)
    :type ellipsoid: (N,) ints, zeros
    :param center: center of point cloud
    :type center: (3,) floats
    :param shape_tensor: shape tensor array to be filled
    :type shape_tensor: (3,3) complex, zeros
    :param eigval: eigenvalue array to be filled
    :type eigval: (3,) double, zeros
    :param eigvec: eigenvector array to be filled
    :type eigvec: (3,3) double, zeros
    :param d: distance from the center, kept fixed during iterative procedure
    :type d: float
    :param M_TOL: convergence tolerance, eigenvalue fractions must differ by less than ``M_TOL``
        for iteration to stop
    :type M_TOL: float
    :param N_WALL: maximum permissible number of iterations
    :type N_WALL: float
    :param N_MIN: minimum number of particles (DM or star particle) in any iteration; 
        if undercut, shape is unclassified
    :type N_MIN: int
    :param CENTER: shape quantities will be calculated with respect to CENTER = 'mode' (point of highest density)
        or 'com' (center of mass) of each halo
    :type CENTER: str
    :return: ``morph_info`` containing d, q, s, eigframe info
    :rtype: (12,) float array"""
    return morph_info

@cython.embedsignature(True)
def runEllVDispAlgo(float[:] morph_info, float[:,:] xyz, float[:,:] vxyz, float[:,:] xyz_princ, float[:] masses, int[:] ellipsoid, float[:] center, float[:] vcenter, complex[::1,:] shape_tensor, double[::1] eigval, complex[::1,:] eigvec, float d, float M_TOL, int N_WALL, int N_MIN):
    """ Similar to ``runEllAlgo`` algorithm for halos/galaxies but for velocity dispersion tensor
    
    Calculates the axis ratios at a distance ``d`` from the center of the entire particle distro.\n
    Note that before and during the iteration, ``d`` is defined with respect to the center of 
    the entire particle distro, not the center of the initial spherical volume as in Katz 1991.\n
    
    :param morph_info: Array to be filled with morphological info. 1st entry: d,
        2nd entry: q, 3rd entry: s, 4th to 6th: normalized major axis, 7th to 9th: normalized intermediate axis,
        10th to 12th: normalized minor axis
    :type morph_info: (12,) floats
    :param xyz: position array
    :type xyz: (N x 3) floats
    :param vxyz: velocity array
    :type vxyz: (N x 3) floats
    :param xyz_princ: position arrays transformed into principal frame (varies from iteration to iteration)
    :type xyz_princ: (N x 3) floats, zeros
    :param masses: mass array
    :type masses: (N x 1) floats
    :param ellipsoid: indices of points that fall into ellipsoid (varies from iteration to iteration)
    :type ellipsoid: (N,) ints, zeros
    :param center: center of point cloud
    :type center: (3,) floats
    :param vcenter: velocity-center of point cloud
    :type vcenter: (3,) floats
    :param shape_tensor: shape tensor array to be filled
    :type shape_tensor: (3,3) complex, zeros
    :param eigval: eigenvalue array to be filled
    :type eigval: (3,) double, zeros
    :param eigvec: eigenvector array to be filled
    :type eigvec: (3,3) double, zeros
    :param d: distance from the center, kept fixed during iterative procedure
    :type d: float
    :param M_TOL: convergence tolerance, eigenvalue fractions must differ by less than ``M_TOL``
        for iteration to stop
    :type M_TOL: float
    :param N_WALL: maximum permissible number of iterations
    :type N_WALL: float
    :param N_MIN: minimum number of particles (DM or star particle) in any iteration; 
        if undercut, shape is unclassified
    :type N_MIN: int
    :param CENTER: shape quantities will be calculated with respect to CENTER = 'mode' (point of highest density)
        or 'com' (center of mass) of each halo
    :type CENTER: str
    :return: ``morph_info`` containing d, q, s, eigframe info
    :rtype: (12,) float array"""
    return
