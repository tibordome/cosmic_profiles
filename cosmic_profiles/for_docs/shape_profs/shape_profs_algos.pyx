#cython: language_level=3
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

cimport openmp
import numpy as np
cimport cython
from libc.math cimport sqrt

@cython.embedsignature(True)
cdef runShellAlgo(float[:] morph_info, float[:,:] xyz, float[:,:] xyz_princ, float[:] masses, int[:] shell, float[:] r_ell, float[:] center, complex[::1,:] shape_tensor, double[::1] eigval, complex[::1,:] eigvec, float d, float delta_d, float IT_TOL, int IT_WALL, int IT_MIN, bint reduced):
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
    :param r_ell: semi-major axis a of the ellipsoid surface on which each particle lies (varies from iteration to iteration)
    :type r_ell: (N,) floats, zeros
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
    :param IT_TOL: convergence tolerance, eigenvalue fractions must differ by less than ``IT_TOL``
        for iteration to stop
    :type IT_TOL: float
    :param IT_WALL: maximum permissible number of iterations
    :type IT_WALL: float
    :param IT_MIN: minimum number of particles (DM or star particle) in any iteration; 
        if undercut, shape is unclassified
    :type IT_MIN: int
    :param reduced: whether or not reduced shape tensor (1/r^2 factor)
    :type reduced: boolean
    :return: ``morph_info`` containing d, q, s, eigframe info
    :rtype: (12,) float array"""
    return

@cython.embedsignature(True)
cdef runEllAlgo(float[:] morph_info, float[:,:] xyz, float[:,:] xyz_princ, float[:] masses, int[:] ellipsoid, float[:] r_ell, float[:] center, complex[::1,:] shape_tensor, double[::1] eigval, complex[::1,:] eigvec, float d, float IT_TOL, int IT_WALL, int IT_MIN, bint reduced):
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
    :param r_ell: semi-major axis a of the ellipsoid surface on which each particle lies (varies from iteration to iteration)
    :type r_ell: (N,) floats, zeros
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
    :param IT_TOL: convergence tolerance, eigenvalue fractions must differ by less than ``IT_TOL``
        for iteration to stop
    :type IT_TOL: float
    :param IT_WALL: maximum permissible number of iterations
    :type IT_WALL: float
    :param IT_MIN: minimum number of particles (DM or star particle) in any iteration; 
        if undercut, shape is unclassified
    :type IT_MIN: int
    :param reduced: whether or not reduced shape tensor (1/r^2 factor)
    :type reduced: boolean
    :return: ``morph_info`` containing d, q, s, eigframe info
    :rtype: (12,) float array"""
    return

@cython.embedsignature(True)
cdef runEllVDispAlgo(float[:] morph_info, float[:,:] xyz, float[:,:] vxyz, float[:,:] xyz_princ, float[:] masses, int[:] ellipsoid, float[:] r_ell, float[:] center, float[:] vcenter, complex[::1,:] shape_tensor, double[::1] eigval, complex[::1,:] eigvec, float d, float IT_TOL, int IT_WALL, int IT_MIN, bint reduced):
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
    :param r_ell: semi-major axis a of the ellipsoid surface on which each particle lies (varies from iteration to iteration)
    :type r_ell: (N,) floats, zeros
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
    :param IT_TOL: convergence tolerance, eigenvalue fractions must differ by less than ``IT_TOL``
        for iteration to stop
    :type IT_TOL: float
    :param IT_WALL: maximum permissible number of iterations
    :type IT_WALL: float
    :param IT_MIN: minimum number of particles (DM or star particle) in any iteration; 
        if undercut, shape is unclassified
    :type IT_MIN: int
    :param reduced: whether or not reduced shape tensor (1/r^2 factor)
    :type reduced: boolean
    :return: ``morph_info`` containing d, q, s, eigframe info
    :rtype: (12,) float array"""
    return

@cython.embedsignature(True)
cdef runShellVDispAlgo(float[:] morph_info, float[:,:] xyz, float[:,:] vxyz, float[:,:] xyz_princ, float[:] masses, int[:] shell, float[:] r_ell, float[:] center, float[:] vcenter, complex[::1,:] shape_tensor, double[::1] eigval, complex[::1,:] eigvec, float d, float delta_d, float IT_TOL, int IT_WALL, int IT_MIN, bint reduced):
    """ Similar to ``runShellAlgo`` algorithm for halos/galaxies but for velocity dispersion tensor
    
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
    :param shell: indices of points that fall into shell (varies from iteration to iteration)
    :type shell: (N,) ints, zeros
    :param r_ell: semi-major axis a of the ellipsoid surface on which each particle lies (varies from iteration to iteration)
    :type r_ell: (N,) floats, zeros
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
    :param delta_d: thickness of the shell in real space (constant across shells in logarithmic space)
    :type delta_d: float
    :param IT_TOL: convergence tolerance, eigenvalue fractions must differ by less than ``IT_TOL``
        for iteration to stop
    :type IT_TOL: float
    :param IT_WALL: maximum permissible number of iterations
    :type IT_WALL: float
    :param IT_MIN: minimum number of particles (DM or star particle) in any iteration; 
        if undercut, shape is unclassified
    :type IT_MIN: int
    :param reduced: whether or not reduced shape tensor (1/r^2 factor)
    :type reduced: boolean
    :return: ``morph_info`` containing d, q, s, eigframe info
    :rtype: (12,) float array"""
    return

@cython.embedsignature(True)
def calcMorphLocal(double[:,:] xyz, double[:] masses, double[:] r200, int[:] idx_cat, int[:] obj_size, double L_BOX, double[:] r_over_r200, double IT_TOL, int IT_WALL, int IT_MIN, str CENTER, bint reduced, bint shell_based):
    """ Calculates the local shape catalogue
    
    Calls ``calcObjMorphLocal()`` in a parallelized manner.\n
    Calculates the axis ratios for the range [ ``r200`` x r_over_r200[0], ``r200`` x r_over_r200[-1]] from the centers, for each object.
    
    :param xyz: positions of all (DM or star) particles in simulation box
    :type xyz: (N2 x 3) floats
    :param masses: masses of the particles expressed in unit mass
    :type masses: (N2 x 1) floats
    :param r200: R_200 radii of the parent halos
    :type r200: (N1,) floats
    :param idx_cat: contains indices of particles belonging to an object
    :type idx_cat: (N3,) integers
    :param obj_size: indicates how many particles are in each object
    :type obj_size: (N1,) integers
    :param L_BOX: simulation box side length
    :type L_BOX: float, units: Mpc/h
    :param r_over_r200: normalized radii at which shape profiles should be estimated
    :type r_over_r200: (r_res,) floats
    :param IT_TOL: convergence tolerance, eigenvalue fractions must differ by less than ``IT_TOL``
        for iteration to stop
    :type IT_TOL: float
    :param IT_WALL: maximum permissible number of iterations
    :type IT_WALL: float
    :param IT_MIN: minimum number of particles (DM or star particle) in any iteration; 
        if undercut, shape is unclassified
    :type IT_MIN: int
    :param CENTER: shape quantities will be calculated with respect to CENTER = 'mode' (point of highest density)
        or 'com' (center of mass) of each halo
    :type CENTER: str
    :param reduced: whether or not reduced shape tensor (1/r^2 factor)
    :type reduced: boolean
    :param shell_based: whether shell-based or ellipsoid-based algorithm should be run
    :type shell_based: boolean
    :return: d, q, s, eigframe, centers, masses, l_succeed: list of object indices for which morphology could be determined at R200 (length: N3)
    :rtype: (N3, ``D_BINS`` + 1) floats (for d, q, s, eigframe (x3)), (N3, 3) floats (for centers), (N3,) floats (for masses), N3-list of ints for l_succeed
    """
    return
    
@cython.embedsignature(True)
def calcMorphGlobal(double[:,:] xyz, double[:] masses, double[:] r200, int[:] idx_cat, int[:] obj_size, double L_BOX, double IT_TOL, int IT_WALL, int IT_MIN, str CENTER, double SAFE, bint reduced):
    """ Calculates the overall shape catalogue
    
    Calls ``calcObjMorphGlobal()`` in a parallelized manner.\n
    Calculates the overall axis ratios and eigenframe for each object.
    
    :param xyz: positions of all (DM or star) particles in simulation box
    :type xyz: (N2 x 3) floats
    :param masses: masses of the particles expressed in unit mass
    :type masses: (N2 x 1) floats
    :param r200: R_200 radii of the parent halos
    :type r200: (N1,) floats
    :param idx_cat: contains indices of particles belonging to an object
    :type idx_cat: (N3,) integers
    :param obj_size: indicates how many particles are in each object
    :type obj_size: (N1,) integers
    :param L_BOX: simulation box side length
    :type L_BOX: float, units: Mpc/h
    :param IT_TOL: convergence tolerance, eigenvalue fractions must differ by less than ``IT_TOL``
        for iteration to stop
    :type IT_TOL: float
    :param IT_WALL: maximum permissible number of iterations
    :type IT_WALL: float
    :param IT_MIN: minimum number of particles (DM or star particle) in any iteration; 
        if undercut, shape is unclassified
    :type IT_MIN: int
    :param CENTER: shape quantities will be calculated with respect to CENTER = 'mode' (point of highest density)
        or 'com' (center of mass) of each halo
    :type CENTER: str
    :param SAFE: ellipsoidal radius will be maxdist(COM,point)+SAFE where point is any point in the point cloud. 
        The larger the better.
    :type SAFE: float
    :param reduced: whether or not reduced shape tensor (1/r^2 factor)
    :type reduced: boolean
    :return: d, q, s, eigframe, centers, masses
    :rtype: (N3,) floats (for d, q, s, eigframe (x3)), (N3, 3) floats (for centers), (N3,) floats (for masses)
    """
    return
    
@cython.embedsignature(True)
def calcMorphLocalVelDisp(double[:,:] xyz, double[:,:] vxyz, double[:] masses, double[:] r200, int[:] idx_cat, int[:] obj_size, double L_BOX, double[:] r_over_r200, double IT_TOL, int IT_WALL, int IT_MIN, str CENTER, bint reduced, bint shell_based):
    """ Calculates the local velocity dispersion shape catalogue
    
    Calls ``calcObjMorphLocalVelDisp()`` in a parallelized manner.\n
    Calculates the overall axis ratios and eigenframe for each object.
    
    :param xyz: positions of all (DM or star) particles in simulation box
    :type xyz: (N2 x 3) floats
    :param vxyz: velocities of all (DM or star) particles in simulation box
    :type vxyz: (N2 x 3) floats
    :param masses: masses of the particles expressed in unit mass
    :type masses: (N2 x 1) floats
    :param r200: R_200 radii of the parent halos
    :type r200: (N1,) floats
    :param idx_cat: contains indices of particles belonging to an object
    :type idx_cat: (N3,) integers
    :param obj_size: indicates how many particles are in each object
    :type obj_size: (N1,) integers
    :param L_BOX: simulation box side length
    :type L_BOX: float, units: Mpc/h
    :param MIN_NUMBER_PTCS: minimum number of particles for object to qualify for morphology calculation
    :type MIN_NUMBER_PTCS: int
    :param r_over_r200: normalized radii at which shape profiles should be estimated
    :type r_over_r200: (r_res,) floats
    :param IT_TOL: convergence tolerance, eigenvalue fractions must differ by less than ``IT_TOL``
        for iteration to stop
    :type IT_TOL: float
    :param IT_WALL: maximum permissible number of iterations
    :type IT_WALL: float
    :param IT_MIN: minimum number of particles (DM or star particle) in any iteration; 
        if undercut, shape is unclassified
    :type IT_MIN: int
    :param CENTER: shape quantities will be calculated with respect to CENTER = 'mode' (point of highest density)
        or 'com' (center of mass) of each halo
    :type CENTER: str
    :param reduced: whether or not reduced shape tensor (1/r^2 factor)
    :type reduced: boolean
    :param shell_based: whether shell-based or ellipsoid-based algorithm should be run
    :type shell_based: boolean
    :return: d, q, s, eigframe, centers, masses, l_succeed: list of object indices for which morphology could be determined at R200 (length: N3)
    :rtype: (N3, ``D_BINS`` + 1) floats (for d, q, s, eigframe (x3)), (N3, 3) floats (for centers), (N3,) floats (for masses), N3-list of ints for l_succeed
    """
    return
    
@cython.embedsignature(True)
def calcMorphGlobalVelDisp(double[:,:] xyz, double[:,:] vxyz, double[:] masses, double[:] r200, int[:] idx_cat, int[:] obj_size, double L_BOX, double IT_TOL, int IT_WALL, int IT_MIN, str CENTER, double SAFE, bint reduced):
    """ Calculates the global velocity dipsersion shape catalogue
    
    Calls ``calcObjMorphGlobalVelDisp()`` in a parallelized manner.\n
    Calculates the overall axis ratios and eigenframe for each object.
    
    :param xyz: positions of all (DM or star) particles in simulation box
    :type xyz: (N2 x 3) floats
    :param vxyz: velocities of all (DM or star) particles in simulation box
    :type vxyz: (N2 x 3) floats
    :param masses: masses of the particles expressed in unit mass
    :type masses: (N2 x 1) floats
    :param r200: R_200 radii of the parent halos
    :type r200: (N1,) floats
    :param idx_cat: contains indices of particles belonging to an object
    :type idx_cat: (N3,) integers
    :param obj_size: indicates how many particles are in each object
    :type obj_size: (N1,) integers
    :param L_BOX: simulation box side length
    :type L_BOX: float, units: Mpc/h
    :param MIN_NUMBER_PTCS: minimum number of particles for object to qualify for morphology calculation
    :type MIN_NUMBER_PTCS: int
    :param IT_TOL: convergence tolerance, eigenvalue fractions must differ by less than ``IT_TOL``
        for iteration to stop
    :type IT_TOL: float
    :param IT_WALL: maximum permissible number of iterations
    :type IT_WALL: float
    :param IT_MIN: minimum number of particles (DM or star particle) in any iteration; 
        if undercut, shape is unclassified
    :type IT_MIN: int
    :param CENTER: shape quantities will be calculated with respect to CENTER = 'mode' (point of highest density)
        or 'com' (center of mass) of each halo
    :type CENTER: str
    :param SAFE: ellipsoidal radius will be maxdist(COM,point)+SAFE where point is any point in the point cloud. 
        The larger the better.
    :type SAFE: float
    :param reduced: whether or not reduced shape tensor (1/r^2 factor)
    :type reduced: boolean
    :return: d, q, s, eigframe, centers, masses
    :rtype: (N3, ``D_BINS`` + 1) floats (for d, q, s, eigframe (x3)), (N3, 3) floats (for centers), (N3,) floats (for masses)
    """
    return

@cython.embedsignature(True)
cdef calcObjMorphLocal(double[:,:] morph_info, double r200, double[:] log_d, double[:,:] xyz, double[:,:] xyz_princ, double[:] masses, int[:] shell, double[:] r_ell, double[:] center, complex[::1,:] shape_tensor, double[::1] eigval, complex[::1,:] eigvec, double IT_TOL, int IT_WALL, int IT_MIN, bint reduced, bint shell_based):
    """ Calculates the local axis ratios
    
    The local morphology is calculated for the ellipsoidal radius range [ ``r200`` x ``log_d`` [0], ``r200`` x ``log_d`` [-1]] 
    from the center of the point cloud
    
    :param morph_info: Array to be filled with morphological info. For each column, 1st entry: d,
        2nd entry: q, 3rd entry: s, 4th to 6th: normalized major axis, 7th to 9th: normalized intermediate axis,
        10th to 12th: normalized minor axis
    :type morph_info: (12,N) floats
    :param r200: R_200 radius of the parent halo
    :type r200: float
    :param log_d: logarithmically equally spaced ellipsoidal radius array of interest, in units of R_200 
        radius of the parent halo, e.g. np.logspace(-2,1,100)
    :type log_d: (N,) floats
    :param xyz: positions of particles in point cloud
    :type xyz: (N1 x 3) floats
    :param xyz_princ: position arrays transformed into principal frame (varies from iteration to iteration)
    :type xyz_princ: (N1 x 3) floats, zeros
    :param masses: masses of the particles expressed in unit mass
    :type masses: (N1 x 1) floats
    :param shell: indices of points that fall into shell (varies from iteration to iteration)
    :type shell: (N1,) ints, zeros
    :param r_ell: semi-major axis a of the ellipsoid surface on which each particle lies (varies from iteration to iteration)
    :type r_ell: (N1,) floats, zeros
    :param center: center of point cloud
    :type center: (3,) floats
    :param shape_tensor: shape tensor array to be filled
    :type shape_tensor: (3,3) complex, zeros
    :param eigval: eigenvalue array to be filled
    :type eigval: (3,) double, zeros
    :param eigvec: eigenvector array to be filled
    :type eigvec: (3,3) double, zeros
    :param IT_TOL: convergence tolerance, eigenvalue fractions must differ by less than ``IT_TOL``
        for iteration to stop
    :type IT_TOL: float
    :param IT_WALL: maximum permissible number of iterations
    :type IT_WALL: float
    :param IT_MIN: minimum number of particles (DM or star particle) in any iteration; 
        if undercut, shape is unclassified
    :type IT_MIN: int
    :param reduced: whether or not reduced shape tensor (1/r^2 factor)
    :type reduced: boolean
    :param shell_based: whether shell-based or ellipsoid-based algorithm should be run
    :type shell_based: boolean
    :return: ``morph_info`` containing d, q, s, eigframe info in each column, for each ellipsoidal radius
    :rtype: (12,N) double array"""
    return

@cython.embedsignature(True)
cdef calcObjMorphGlobal(double[:] morph_info, double r200, double[:,:] xyz, double[:,:] xyz_princ, double[:] masses, int[:] ellipsoid, double[:] r_ell, double[:] center, complex[::1,:] shape_tensor, double[::1] eigval, complex[::1,:] eigvec, double IT_TOL, int IT_WALL, int IT_MIN, double SAFE, bint reduced):
    """ Calculates the global axis ratios and eigenframe of the point cloud
    
    :param morph_info: Array to be filled with morphological info. 1st entry: d,
        2nd entry: q, 3rd entry: s, 4th to 6th: normalized major axis, 7th to 9th: normalized intermediate axis,
        10th to 12th: normalized minor axis
    :type morph_info: (12,) floats
    :param r200: R_200 radius of the parent halo
    :type r200: float
    :param xyz: positions of particles in point cloud
    :type xyz: (N1 x 3) floats
    :param xyz_princ: position arrays transformed into principal frame (varies from iteration to iteration)
    :type xyz_princ: (N1 x 3) floats, zeros
    :param masses: masses of the particles expressed in unit mass
    :type masses: (N1 x 1) floats
    :param ellipsoid: indices of points that fall into ellipsoid (varies from iteration to iteration)
    :type ellipsoid: (N1,) ints, zeros
    :param r_ell: semi-major axis a of the ellipsoid surface on which each particle lies (varies from iteration to iteration)
    :type r_ell: (N1,) floats, zeros
    :param center: center of point cloud
    :type center: (3,) floats
    :param shape_tensor: shape tensor array to be filled
    :type shape_tensor: (3,3) complex, zeros
    :param eigval: eigenvalue array to be filled
    :type eigval: (3,) double, zeros
    :param eigvec: eigenvector array to be filled
    :type eigvec: (3,3) double, zeros
    :param IT_TOL: convergence tolerance, eigenvalue fractions must differ by less than ``IT_TOL``
        for iteration to stop
    :type IT_TOL: float
    :param IT_WALL: maximum permissible number of iterations
    :type IT_WALL: float
    :param IT_MIN: minimum number of particles (DM or star particle) in any iteration; 
        if undercut, shape is unclassified
    :type IT_MIN: int
    :param SAFE: ellipsoidal radius will be maxdist(COM,point)+SAFE where point is any point in the point cloud. 
        The larger the better.
    :type SAFE: float
    :param reduced: whether or not reduced shape tensor (1/r^2 factor)
    :type reduced: boolean
    :return: ``morph_info`` containing d, q, s, eigframe info
    :rtype: (12,) double array"""
    return

@cython.embedsignature(True)
cdef calcObjMorphLocalVelDisp(double[:,:] morph_info, double r200, double[:] log_d, double[:,:] xyz, double[:,:] vxyz, double[:,:] xyz_princ, double[:] masses, int[:] shell, double[:] r_ell, double[:] center, double[:] vcenter, complex[::1,:] shape_tensor, double[::1] eigval, complex[::1,:] eigvec, double IT_TOL, int IT_WALL, int IT_MIN, bint reduced, bint shell_based):
    """ Calculates the local axis ratios of the velocity dispersion tensor 
    
    The local morphology is calculated for the ellipsoidal radius range [ ``r200`` x ``log_d`` [0], ``r200`` x ``log_d`` [-1]] 
    from the center of the point cloud
    
    :param morph_info: Array to be filled with morphological info. For each column, 1st entry: d,
        2nd entry: q, 3rd entry: s, 4th to 6th: normalized major axis, 7th to 9th: normalized intermediate axis,
        10th to 12th: normalized minor axis
    :type morph_info: (12,N) floats
    :param r200: R_200 radius of the parent halo
    :type r200: float
    :param log_d: logarithmically equally spaced ellipsoidal radius array of interest, in units of R_200 
        radius of the parent halo, e.g. np.logspace(-2,1,100)
    :type log_d: (N,) floats
    :param xyz: positions of particles in point cloud
    :type xyz: (N1 x 3) floats
    :param vxyz: velocity array
    :type vxyz: (N1 x 3) floats
    :param xyz_princ: position arrays transformed into principal frame (varies from iteration to iteration)
    :type xyz_princ: (N1 x 3) floats, zeros
    :param masses: masses of the particles expressed in unit mass
    :type masses: (N1 x 1) floats
    :param shell: indices of points that fall into shell (varies from iteration to iteration)
    :type shell: (N1,) ints, zeros
    :param r_ell: semi-major axis a of the ellipsoid surface on which each particle lies (varies from iteration to iteration)
    :type r_ell: (N1,) floats, zeros
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
    :param IT_TOL: convergence tolerance, eigenvalue fractions must differ by less than ``IT_TOL``
        for iteration to stop
    :type IT_TOL: float
    :param IT_WALL: maximum permissible number of iterations
    :type IT_WALL: float
    :param IT_MIN: minimum number of particles (DM or star particle) in any iteration; 
        if undercut, shape is unclassified
    :type IT_MIN: int
    :param reduced: whether or not reduced shape tensor (1/r^2 factor)
    :type reduced: boolean
    :param shell_based: whether shell-based or ellipsoid-based algorithm should be run
    :type shell_based: boolean
    :return: ``morph_info`` containing d (= ``r200``), q, s, eigframe info
    :rtype: (12,N) double array"""
    return

@cython.embedsignature(True)
cdef calcObjMorphGlobalVelDisp(double[:] morph_info, double r200, double[:,:] xyz, double[:,:] vxyz, double[:,:] xyz_princ, double[:] masses, int[:] ellipsoid, double[:] r_ell, double[:] center, double[:] vcenter, complex[::1,:] shape_tensor, double[::1] eigval, complex[::1,:] eigvec, double IT_TOL, int IT_WALL, int IT_MIN, double SAFE, bint reduced):
    """ Calculates the global axis ratios and eigenframe of the velocity dispersion tensor
    
    :param morph_info: Array to be filled with morphological info. 1st entry: d,
        2nd entry: q, 3rd entry: s, 4th to 6th: normalized major axis, 7th to 9th: normalized intermediate axis,
        10th to 12th: normalized minor axis
    :type morph_info: (12,) floats
    :param r200: R_200 radius of the parent halo
    :type r200: float
    :param xyz: positions of particles in point cloud
    :type xyz: (N1 x 3) floats
    :param vxyz: velocity array
    :type vxyz: (N1 x 3) floats
    :param xyz_princ: position arrays transformed into principal frame (varies from iteration to iteration)
    :type xyz_princ: (N1 x 3) floats, zeros
    :param masses: masses of the particles expressed in unit mass
    :type masses: (N1 x 1) floats
    :param ellipsoid: indices of points that fall into ellipsoid (varies from iteration to iteration)
    :type ellipsoid: (N1,) ints, zeros
    :param r_ell: semi-major axis a of the ellipsoid surface on which each particle lies (varies from iteration to iteration)
    :type r_ell: (N1,) floats, zeros
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
    :param delta_d: thickness of the shell in real space (constant across shells in logarithmic space)
    :type delta_d: float
    :param IT_TOL: convergence tolerance, eigenvalue fractions must differ by less than ``IT_TOL``
        for iteration to stop
    :type IT_TOL: float
    :param IT_WALL: maximum permissible number of iterations
    :type IT_WALL: float
    :param IT_MIN: minimum number of particles (DM or star particle) in any iteration; 
        if undercut, shape is unclassified
    :type IT_MIN: int
    :param SAFE: ellipsoidal radius will be maxdist(COM,point)+SAFE where point is any point in the point cloud. 
        The larger the better.
    :type SAFE: float
    :param reduced: whether or not reduced shape tensor (1/r^2 factor)
    :type reduced: boolean
    :return: ``morph_info`` containing d (= ``r200``), q, s, eigframe info
    :rtype: (12,) double array"""
    return