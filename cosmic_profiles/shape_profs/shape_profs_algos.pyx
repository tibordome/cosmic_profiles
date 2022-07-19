#cython: language_level=3
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

cimport openmp
import numpy as np
from cosmic_profiles.cython_helpers.helper_class cimport CythonHelpers
from libc.stdio cimport printf
from cython.parallel import prange
from cosmic_profiles.common.python_routines import respectPBCNoRef, calcCoM, calcMode
from cosmic_profiles.common.caching import np_cache_factory
cimport cython
from libc.math cimport sqrt

@cython.embedsignature(True)
cdef float[:] runShellAlgo(float[:] morph_info, float[:,:] xyz, float[:,:] xyz_princ, float[:] masses, int[:] shell, float[:] r_ell, float[:] center, complex[::1,:] shape_tensor, double[::1] eigval, complex[::1,:] eigvec, float d, float delta_d, float M_TOL, int N_WALL, int N_MIN, bint reduced) nogil:
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
    :param M_TOL: convergence tolerance, eigenvalue fractions must differ by less than ``M_TOL``
        for iteration to stop
    :type M_TOL: float
    :param N_WALL: maximum permissible number of iterations
    :type N_WALL: float
    :param N_MIN: minimum number of particles (DM or star particle) in any iteration; 
        if undercut, shape is unclassified
    :type N_MIN: int
    :param reduced: whether or not reduced shape tensor (1/r^2 factor)
    :type reduced: boolean
    :return: ``morph_info`` containing d, q, s, eigframe info
    :rtype: (12,) float array"""
    
    shell[:] = 0
    cdef int pts_in_shell = 0
    cdef int corr = 0
    cdef float err = 1.0
    cdef float q_new = 1.0
    cdef float s_new = 1.0
    cdef float q_old = 1.0
    cdef float s_old = 1.0
    cdef int iteration = 1
    cdef float vec2_norm = 1.0
    cdef float vec1_norm = 1.0
    cdef float vec0_norm = 1.0
    cdef int i
    # Start with spherical shell
    for i in range(xyz.shape[0]):
        if (center[0]-xyz[i,0])**2+(center[1]-xyz[i,1])**2+(center[2]-xyz[i,2])**2 < d**2 and (center[0]-xyz[i,0])**2+(center[1]-xyz[i,1])**2+(center[2]-xyz[i,2])**2 >= (d-delta_d)**2:
            shell[i-corr] = i
            pts_in_shell += 1
        else:
            corr += 1
        r_ell[i] = sqrt((center[0]-xyz[i,0])**2+(center[1]-xyz[i,1])**2/q_new**2+(center[2]-xyz[i,2])**2/s_new**2)
    while (err > M_TOL):
        if iteration > N_WALL:
            morph_info[:] = 0.0
            return morph_info
        if pts_in_shell < N_MIN:
            morph_info[:] = 0.0
            return morph_info
        # Get shape tensor
        shape_tensor = CythonHelpers.calcShapeTensor(xyz, shell, shape_tensor, masses, center, pts_in_shell, reduced, r_ell)
        # Diagonalize shape_tensor
        eigvec[:,:] = 0.0
        eigval[:] = 0.0
        CythonHelpers.ZHEEVR(shape_tensor[:,:], &eigval[0], eigvec, 3)
        q_old = q_new; s_old = s_new
        q_new = sqrt(eigval[1]/eigval[2])
        s_new = sqrt(eigval[0]/eigval[2]) # It is assumed that eigenvalues are approximately proportional to a^2 etc. (true for uniform ellipsoid or uniform shell), though I have never seen any proof..
        err = max(CythonHelpers.cython_abs(q_new - q_old)/q_old, CythonHelpers.cython_abs(s_new - s_old)/s_old) # Fractional differences
        vec2_norm = sqrt(eigvec[0,2].real**2+eigvec[1,2].real**2+eigvec[2,2].real**2)
        vec1_norm = sqrt(eigvec[0,1].real**2+eigvec[1,1].real**2+eigvec[2,1].real**2)
        vec0_norm = sqrt(eigvec[0,0].real**2+eigvec[1,0].real**2+eigvec[2,0].real**2)
        # Update morph_info
        morph_info[0] = d
        morph_info[1] = q_new
        morph_info[2] = s_new
        morph_info[3] = eigvec[0,2].real/vec2_norm
        morph_info[4] = eigvec[1,2].real/vec2_norm
        morph_info[5] = eigvec[2,2].real/vec2_norm
        morph_info[6] = eigvec[0,1].real/vec1_norm
        morph_info[7] = eigvec[1,1].real/vec1_norm
        morph_info[8] = eigvec[2,1].real/vec1_norm
        morph_info[9] = eigvec[0,0].real/vec0_norm
        morph_info[10] = eigvec[1,0].real/vec0_norm
        morph_info[11] = eigvec[2,0].real/vec0_norm
        # Transformation into the principal frame
        for i in range(xyz.shape[0]):
            xyz_princ[i,0] = eigvec[0,2].real/vec2_norm*(xyz[i,0]-center[0])+eigvec[1,2].real/vec2_norm*(xyz[i,1]-center[1])+eigvec[2,2].real/vec2_norm*(xyz[i,2]-center[2])
            xyz_princ[i,1] = eigvec[0,1].real/vec1_norm*(xyz[i,0]-center[0])+eigvec[1,1].real/vec1_norm*(xyz[i,1]-center[1])+eigvec[2,1].real/vec1_norm*(xyz[i,2]-center[2])
            xyz_princ[i,2] = eigvec[0,0].real/vec0_norm*(xyz[i,0]-center[0])+eigvec[1,0].real/vec0_norm*(xyz[i,1]-center[1])+eigvec[2,0].real/vec0_norm*(xyz[i,2]-center[2])
            r_ell[i] = sqrt(xyz_princ[i,0]**2+xyz_princ[i,1]**2/q_new**2+xyz_princ[i,2]**2/s_new**2)
        shell[:] = 0
        pts_in_shell = 0
        corr = 0
        if d == delta_d: # I.e. first shell
            for i in range(xyz_princ.shape[0]):
                if xyz_princ[i,0]**2+xyz_princ[i,1]**2/q_new**2+xyz_princ[i,2]**2/s_new**2 < d**2:
                    shell[i-corr] = i
                    pts_in_shell += 1
                else:
                    corr += 1
        else:
            for i in range(xyz_princ.shape[0]):
                if xyz_princ[i,0]**2+xyz_princ[i,1]**2/q_new**2+xyz_princ[i,2]**2/s_new**2 < d**2 and xyz_princ[i,0]**2/(d-delta_d)**2+xyz_princ[i,1]**2/(q_new*d-delta_d)**2+xyz_princ[i,2]**2/(s_new*d-delta_d)**2 >= 1:
                    shell[i-corr] = i
                    pts_in_shell += 1
                else:
                    corr += 1
        iteration += 1
    return morph_info

@cython.embedsignature(True)
cdef float[:] runEllAlgo(float[:] morph_info, float[:,:] xyz, float[:,:] xyz_princ, float[:] masses, int[:] ellipsoid, float[:] r_ell, float[:] center, complex[::1,:] shape_tensor, double[::1] eigval, complex[::1,:] eigvec, float d, float M_TOL, int N_WALL, int N_MIN, bint reduced) nogil:
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
    :param M_TOL: convergence tolerance, eigenvalue fractions must differ by less than ``M_TOL``
        for iteration to stop
    :type M_TOL: float
    :param N_WALL: maximum permissible number of iterations
    :type N_WALL: float
    :param N_MIN: minimum number of particles (DM or star particle) in any iteration; 
        if undercut, shape is unclassified
    :type N_MIN: int
    :param reduced: whether or not reduced shape tensor (1/r^2 factor)
    :type reduced: boolean
    :return: ``morph_info`` containing d, q, s, eigframe info
    :rtype: (12,) float array"""
    
    ellipsoid[:] = 0
    cdef int pts_in_ell = 0
    cdef int corr = 0
    cdef float err = 1.0
    cdef float q_new = 1.0
    cdef float s_new = 1.0
    cdef float q_old = 1.0
    cdef float s_old = 1.0
    cdef int iteration = 1
    cdef float vec2_norm = 1.0
    cdef float vec1_norm = 1.0
    cdef float vec0_norm = 1.0
    cdef int i
    # Start with sphere
    for i in range(xyz.shape[0]):
        if (center[0]-xyz[i,0])**2+(center[1]-xyz[i,1])**2+(center[2]-xyz[i,2])**2 < d**2:
            ellipsoid[i-corr] = i
            pts_in_ell += 1
        else:
            corr += 1
        r_ell[i] = sqrt((center[0]-xyz[i,0])**2+(center[1]-xyz[i,1])**2/q_new**2+(center[2]-xyz[i,2])**2/s_new**2)
    while (err > M_TOL):
        if iteration > N_WALL:
            morph_info[:] = 0.0
            return morph_info
        if pts_in_ell < N_MIN:
            morph_info[:] = 0.0
            return morph_info
        # Get shape tensor
        shape_tensor = CythonHelpers.calcShapeTensor(xyz, ellipsoid, shape_tensor, masses, center, pts_in_ell, reduced, r_ell)
        # Diagonalize shape_tensor
        eigvec[:,:] = 0.0
        eigval[:] = 0.0
        CythonHelpers.ZHEEVR(shape_tensor[:,:], &eigval[0], eigvec, 3)
        q_old = q_new; s_old = s_new
        q_new = sqrt(eigval[1]/eigval[2])
        s_new = sqrt(eigval[0]/eigval[2]) # It is assumed that eigenvalues are approximately proportional to a^2 etc. (true for uniform ellipsoid or uniform shell), though I have never seen any proof..
        err = max(CythonHelpers.cython_abs(q_new - q_old)/q_old, CythonHelpers.cython_abs(s_new - s_old)/s_old) # Fractional differences
        vec2_norm = sqrt(eigvec[0,2].real**2+eigvec[1,2].real**2+eigvec[2,2].real**2)
        vec1_norm = sqrt(eigvec[0,1].real**2+eigvec[1,1].real**2+eigvec[2,1].real**2)
        vec0_norm = sqrt(eigvec[0,0].real**2+eigvec[1,0].real**2+eigvec[2,0].real**2)
        # Update morph_info
        morph_info[0] = d
        morph_info[1] = q_new
        morph_info[2] = s_new
        morph_info[3] = eigvec[0,2].real/vec2_norm
        morph_info[4] = eigvec[1,2].real/vec2_norm
        morph_info[5] = eigvec[2,2].real/vec2_norm
        morph_info[6] = eigvec[0,1].real/vec1_norm
        morph_info[7] = eigvec[1,1].real/vec1_norm
        morph_info[8] = eigvec[2,1].real/vec1_norm
        morph_info[9] = eigvec[0,0].real/vec0_norm
        morph_info[10] = eigvec[1,0].real/vec0_norm
        morph_info[11] = eigvec[2,0].real/vec0_norm
        # Transformation into the principal frame
        for i in range(xyz.shape[0]):
            xyz_princ[i,0] = eigvec[0,2].real/vec2_norm*(xyz[i,0]-center[0])+eigvec[1,2].real/vec2_norm*(xyz[i,1]-center[1])+eigvec[2,2].real/vec2_norm*(xyz[i,2]-center[2])
            xyz_princ[i,1] = eigvec[0,1].real/vec1_norm*(xyz[i,0]-center[0])+eigvec[1,1].real/vec1_norm*(xyz[i,1]-center[1])+eigvec[2,1].real/vec1_norm*(xyz[i,2]-center[2])
            xyz_princ[i,2] = eigvec[0,0].real/vec0_norm*(xyz[i,0]-center[0])+eigvec[1,0].real/vec0_norm*(xyz[i,1]-center[1])+eigvec[2,0].real/vec0_norm*(xyz[i,2]-center[2])
            r_ell[i] = sqrt(xyz_princ[i,0]**2+xyz_princ[i,1]**2/q_new**2+xyz_princ[i,2]**2/s_new**2)
        ellipsoid[:] = 0
        pts_in_ell = 0
        corr = 0
        for i in range(xyz_princ.shape[0]):
            if xyz_princ[i,0]**2+xyz_princ[i,1]**2/q_new**2+xyz_princ[i,2]**2/s_new**2 < d**2:
                ellipsoid[i-corr] = i
                pts_in_ell += 1
            else:
                corr += 1
        iteration += 1
    return morph_info

@cython.embedsignature(True)
cdef float[:] runEllVDispAlgo(float[:] morph_info, float[:,:] xyz, float[:,:] vxyz, float[:,:] xyz_princ, float[:] masses, int[:] ellipsoid, float[:] r_ell, float[:] center, float[:] vcenter, complex[::1,:] shape_tensor, double[::1] eigval, complex[::1,:] eigvec, float d, float M_TOL, int N_WALL, int N_MIN, bint reduced) nogil:
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
    :param M_TOL: convergence tolerance, eigenvalue fractions must differ by less than ``M_TOL``
        for iteration to stop
    :type M_TOL: float
    :param N_WALL: maximum permissible number of iterations
    :type N_WALL: float
    :param N_MIN: minimum number of particles (DM or star particle) in any iteration; 
        if undercut, shape is unclassified
    :type N_MIN: int
    :param reduced: whether or not reduced shape tensor (1/r^2 factor)
    :type reduced: boolean
    :return: ``morph_info`` containing d, q, s, eigframe info
    :rtype: (12,) float array"""
    
    ellipsoid[:] = 0
    cdef int pts_in_ell = 0
    cdef int corr = 0
    cdef float err = 1.0
    cdef float q_new = 1.0
    cdef float s_new = 1.0
    cdef float q_old = 1.0
    cdef float s_old = 1.0
    cdef int iteration = 1
    cdef float vec2_norm = 1.0
    cdef float vec1_norm = 1.0
    cdef float vec0_norm = 1.0
    cdef int i
    # Start with sphere
    for i in range(xyz.shape[0]):
        if (center[0]-xyz[i,0])**2+(center[1]-xyz[i,1])**2+(center[2]-xyz[i,2])**2 < d**2:
            ellipsoid[i-corr] = i
            pts_in_ell += 1
        else:
            corr += 1
        r_ell[i] = sqrt((center[0]-xyz[i,0])**2+(center[1]-xyz[i,1])**2/q_new**2+(center[2]-xyz[i,2])**2/s_new**2)
    while (err > M_TOL):
        if iteration > N_WALL:
            morph_info[:] = 0.0
            return morph_info
        if pts_in_ell < N_MIN:
            morph_info[:] = 0.0
            return morph_info
        # Get shape tensor
        shape_tensor = CythonHelpers.calcShapeTensor(vxyz, ellipsoid, shape_tensor, masses, vcenter, pts_in_ell, reduced, r_ell)
        # Diagonalize shape_tensor
        eigvec[:,:] = 0.0
        eigval[:] = 0.0
        CythonHelpers.ZHEEVR(shape_tensor[:,:], &eigval[0], eigvec, 3)
        q_old = q_new; s_old = s_new
        q_new = sqrt(eigval[1]/eigval[2])
        s_new = sqrt(eigval[0]/eigval[2]) # It is assumed that eigenvalues are approximately proportional to a^2 etc. (true for uniform ellipsoid or uniform shell), though I have never seen any proof..
        err = max(CythonHelpers.cython_abs(q_new - q_old)/q_old, CythonHelpers.cython_abs(s_new - s_old)/s_old) # Fractional differences
        vec2_norm = sqrt(eigvec[0,2].real**2+eigvec[1,2].real**2+eigvec[2,2].real**2)
        vec1_norm = sqrt(eigvec[0,1].real**2+eigvec[1,1].real**2+eigvec[2,1].real**2)
        vec0_norm = sqrt(eigvec[0,0].real**2+eigvec[1,0].real**2+eigvec[2,0].real**2)
        # Update morph_info
        morph_info[0] = d
        morph_info[1] = q_new
        morph_info[2] = s_new
        morph_info[3] = eigvec[0,2].real/vec2_norm
        morph_info[4] = eigvec[1,2].real/vec2_norm
        morph_info[5] = eigvec[2,2].real/vec2_norm
        morph_info[6] = eigvec[0,1].real/vec1_norm
        morph_info[7] = eigvec[1,1].real/vec1_norm
        morph_info[8] = eigvec[2,1].real/vec1_norm
        morph_info[9] = eigvec[0,0].real/vec0_norm
        morph_info[10] = eigvec[1,0].real/vec0_norm
        morph_info[11] = eigvec[2,0].real/vec0_norm
        # Transformation into the principal frame
        for i in range(xyz.shape[0]):
            xyz_princ[i,0] = eigvec[0,2].real/vec2_norm*(xyz[i,0]-center[0])+eigvec[1,2].real/vec2_norm*(xyz[i,1]-center[1])+eigvec[2,2].real/vec2_norm*(xyz[i,2]-center[2])
            xyz_princ[i,1] = eigvec[0,1].real/vec1_norm*(xyz[i,0]-center[0])+eigvec[1,1].real/vec1_norm*(xyz[i,1]-center[1])+eigvec[2,1].real/vec1_norm*(xyz[i,2]-center[2])
            xyz_princ[i,2] = eigvec[0,0].real/vec0_norm*(xyz[i,0]-center[0])+eigvec[1,0].real/vec0_norm*(xyz[i,1]-center[1])+eigvec[2,0].real/vec0_norm*(xyz[i,2]-center[2])
            r_ell[i] = sqrt(xyz_princ[i,0]**2+xyz_princ[i,1]**2/q_new**2+xyz_princ[i,2]**2/s_new**2)
        ellipsoid[:] = 0
        pts_in_ell = 0
        corr = 0
        for i in range(xyz_princ.shape[0]):
            if xyz_princ[i,0]**2+xyz_princ[i,1]**2/q_new**2+xyz_princ[i,2]**2/s_new**2 < d**2:
                ellipsoid[i-corr] = i
                pts_in_ell += 1
            else:
                corr += 1
        iteration += 1
    return morph_info

@cython.embedsignature(True)
cdef float[:] runShellVDispAlgo(float[:] morph_info, float[:,:] xyz, float[:,:] vxyz, float[:,:] xyz_princ, float[:] masses, int[:] shell, float[:] r_ell, float[:] center, float[:] vcenter, complex[::1,:] shape_tensor, double[::1] eigval, complex[::1,:] eigvec, float d, float delta_d, float M_TOL, int N_WALL, int N_MIN, bint reduced) nogil:
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
    :param M_TOL: convergence tolerance, eigenvalue fractions must differ by less than ``M_TOL``
        for iteration to stop
    :type M_TOL: float
    :param N_WALL: maximum permissible number of iterations
    :type N_WALL: float
    :param N_MIN: minimum number of particles (DM or star particle) in any iteration; 
        if undercut, shape is unclassified
    :type N_MIN: int
    :param reduced: whether or not reduced shape tensor (1/r^2 factor)
    :type reduced: boolean
    :return: ``morph_info`` containing d, q, s, eigframe info
    :rtype: (12,) float array"""
    
    shell[:] = 0
    cdef int pts_in_shell = 0
    cdef int corr = 0
    cdef float err = 1.0
    cdef float q_new = 1.0
    cdef float s_new = 1.0
    cdef float q_old = 1.0
    cdef float s_old = 1.0
    cdef int iteration = 1
    cdef float vec2_norm = 1.0
    cdef float vec1_norm = 1.0
    cdef float vec0_norm = 1.0
    cdef int i
    # Start with spherical shell
    for i in range(xyz.shape[0]):
        if (center[0]-xyz[i,0])**2+(center[1]-xyz[i,1])**2+(center[2]-xyz[i,2])**2 < d**2 and (center[0]-xyz[i,0])**2+(center[1]-xyz[i,1])**2+(center[2]-xyz[i,2])**2 >= (d-delta_d)**2:
            shell[i-corr] = i
            pts_in_shell += 1
        else:
            corr += 1
        r_ell[i] = sqrt((center[0]-xyz[i,0])**2+(center[1]-xyz[i,1])**2/q_new**2+(center[2]-xyz[i,2])**2/s_new**2)
    while (err > M_TOL):
        if iteration > N_WALL:
            morph_info[:] = 0.0
            return morph_info
        if pts_in_shell < N_MIN:
            morph_info[:] = 0.0
            return morph_info
        # Get shape tensor
        shape_tensor = CythonHelpers.calcShapeTensor(vxyz, shell, shape_tensor, masses, vcenter, pts_in_shell, reduced, r_ell)
        # Diagonalize shape_tensor
        eigvec[:,:] = 0.0
        eigval[:] = 0.0
        CythonHelpers.ZHEEVR(shape_tensor[:,:], &eigval[0], eigvec, 3)
        q_old = q_new; s_old = s_new
        q_new = sqrt(eigval[1]/eigval[2])
        s_new = sqrt(eigval[0]/eigval[2]) # It is assumed that eigenvalues are approximately proportional to a^2 etc. (true for uniform ellipsoid or uniform shell), though I have never seen any proof..
        err = max(CythonHelpers.cython_abs(q_new - q_old)/q_old, CythonHelpers.cython_abs(s_new - s_old)/s_old) # Fractional differences
        vec2_norm = sqrt(eigvec[0,2].real**2+eigvec[1,2].real**2+eigvec[2,2].real**2)
        vec1_norm = sqrt(eigvec[0,1].real**2+eigvec[1,1].real**2+eigvec[2,1].real**2)
        vec0_norm = sqrt(eigvec[0,0].real**2+eigvec[1,0].real**2+eigvec[2,0].real**2)
        # Update morph_info
        morph_info[0] = d
        morph_info[1] = q_new
        morph_info[2] = s_new
        morph_info[3] = eigvec[0,2].real/vec2_norm
        morph_info[4] = eigvec[1,2].real/vec2_norm
        morph_info[5] = eigvec[2,2].real/vec2_norm
        morph_info[6] = eigvec[0,1].real/vec1_norm
        morph_info[7] = eigvec[1,1].real/vec1_norm
        morph_info[8] = eigvec[2,1].real/vec1_norm
        morph_info[9] = eigvec[0,0].real/vec0_norm
        morph_info[10] = eigvec[1,0].real/vec0_norm
        morph_info[11] = eigvec[2,0].real/vec0_norm
        # Transformation into the principal frame
        for i in range(xyz.shape[0]):
            xyz_princ[i,0] = eigvec[0,2].real/vec2_norm*(xyz[i,0]-center[0])+eigvec[1,2].real/vec2_norm*(xyz[i,1]-center[1])+eigvec[2,2].real/vec2_norm*(xyz[i,2]-center[2])
            xyz_princ[i,1] = eigvec[0,1].real/vec1_norm*(xyz[i,0]-center[0])+eigvec[1,1].real/vec1_norm*(xyz[i,1]-center[1])+eigvec[2,1].real/vec1_norm*(xyz[i,2]-center[2])
            xyz_princ[i,2] = eigvec[0,0].real/vec0_norm*(xyz[i,0]-center[0])+eigvec[1,0].real/vec0_norm*(xyz[i,1]-center[1])+eigvec[2,0].real/vec0_norm*(xyz[i,2]-center[2])
            r_ell[i] = sqrt(xyz_princ[i,0]**2+xyz_princ[i,1]**2/q_new**2+xyz_princ[i,2]**2/s_new**2)
        shell[:] = 0
        pts_in_shell = 0
        corr = 0
        if d == delta_d: # I.e. first shell
            for i in range(xyz_princ.shape[0]):
                if xyz_princ[i,0]**2+xyz_princ[i,1]**2/q_new**2+xyz_princ[i,2]**2/s_new**2 < d**2:
                    shell[i-corr] = i
                    pts_in_shell += 1
                else:
                    corr += 1
        else:
            for i in range(xyz_princ.shape[0]):
                if xyz_princ[i,0]**2+xyz_princ[i,1]**2/q_new**2+xyz_princ[i,2]**2/s_new**2 < d**2 and xyz_princ[i,0]**2/(d-delta_d)**2+xyz_princ[i,1]**2/(q_new*d-delta_d)**2+xyz_princ[i,2]**2/(s_new*d-delta_d)**2 >= 1:
                    shell[i-corr] = i
                    pts_in_shell += 1
                else:
                    corr += 1
        iteration += 1
    return morph_info

@cython.embedsignature(True)
@cython.binding(True)
def calcMorphLocal(float[:,:] xyz, float[:] masses, float[:] r200, cat, float L_BOX, int MIN_NUMBER_PTCS, int D_LOGSTART, int D_LOGEND, int D_BINS, int M_TOL, int N_WALL, int N_MIN, str CENTER, bint reduced, bint shell_based):
    """ Calculates the local shape catalogue
    
    Calls ``calcObjMorphLocal()`` in a parallelized manner.\n
    Calculates the axis ratios for the range [ ``r200`` x 10**(``D_LOGSTART``), ``r200`` x 10**(``D_LOGEND``)] from the centers, for each object.
    
    :param xyz: positions of all (DM or star) particles in simulation box
    :type xyz: (N2 x 3) floats
    :param cat: each entry of the list is a list containing indices of particles belonging to an object
    :type cat: list of length N1
    :param masses: masses of the particles expressed in unit mass
    :type masses: (N2 x 1) floats
    :param r200: each entry of the list gives the R_200 radius of the parent halo
    :type r200: list of length N1
    :param L_BOX: simulation box side length
    :type L_BOX: float, units: Mpc/h
    :param MIN_NUMBER_PTCS: minimum number of particles for object to qualify for morphology calculation
    :type MIN_NUMBER_PTCS: int
    :param D_LOGSTART: logarithm of minimum ellipsoidal radius of interest, in units of R200 of parent halo
    :type D_LOGSTART: int
    :param D_LOGEND: logarithm of maximum ellipsoidal radius of interest, in units of R200 of parent halo
    :type D_LOGEND: int
    :param D_BINS: number of ellipsoidal radii of interest minus 1 (i.e. number of bins)
    :type D_BINS: int
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
    :param reduced: whether or not reduced shape tensor (1/r^2 factor)
    :type reduced: boolean
    :param shell_based: whether shell-based or ellipsoid-based algorithm should be run
    :type shell_based: boolean
    :return: d, q, s, eigframe, centers, masses, l_succeed: list of object indices for which morphology could be determined at R200 (length: N3)
    :rtype: (N3, ``D_BINS`` + 1) floats (for d, q, s, eigframe (x3)), (N3, 3) floats (for centers), (N3,) floats (for masses), N3-list of ints for l_succeed
    """
    def inner(float[:,:] xyz, float[:] masses, float[:] r200, cat, float L_BOX, int MIN_NUMBER_PTCS, int D_LOGSTART, int D_LOGEND, int D_BINS, int M_TOL, int N_WALL, int N_MIN, str CENTER, bint reduced, bint shell_based):
        # Transform cat to int[:,:]
        cdef int nb_objs = len(cat)
        cdef int p
        cdef int[:] obj_pass = np.zeros((nb_objs,), dtype = np.int32)
        cdef int[:] obj_size = np.zeros((nb_objs,), dtype = np.int32)
        for p in range(nb_objs):
            if len(cat[p]) >= MIN_NUMBER_PTCS: # Only add objects that have sufficient resolution
                obj_pass[p] = 1      
                obj_size[p] = len(cat[p]) 
        cdef int nb_pass = np.sum(obj_pass.base)
        cdef int[:,:] cat_arr = np.zeros((nb_pass,np.max([len(cat[p]) for p in range(nb_objs)])), dtype = np.int32)
        cdef int[:] idxs_compr = np.zeros((nb_objs,), dtype = np.int32)
        idxs_compr.base[obj_pass.base.nonzero()[0]] = np.arange(np.sum(obj_pass.base))
        for p in range(nb_objs):
            if obj_pass[p] == 1:
                cat_arr.base[idxs_compr[p],:obj_size[p]] = np.array(cat[p])
    
        cdef float[:] m = np.zeros((nb_pass,), dtype = np.float32)
        cdef float[:,:] d = np.zeros((nb_pass, D_BINS+1), dtype = np.float32)
        cdef float[:,:] q = np.zeros((nb_pass, D_BINS+1), dtype = np.float32)
        cdef float[:,:] s = np.zeros((nb_pass, D_BINS+1), dtype = np.float32)
        cdef float[:,:] major_x = np.zeros((nb_pass, D_BINS+1), dtype = np.float32)
        cdef float[:,:] major_y = np.zeros((nb_pass, D_BINS+1), dtype = np.float32)
        cdef float[:,:] major_z = np.zeros((nb_pass, D_BINS+1), dtype = np.float32)
        cdef float[:,:] inter_x = np.zeros((nb_pass, D_BINS+1), dtype = np.float32)
        cdef float[:,:] inter_y = np.zeros((nb_pass, D_BINS+1), dtype = np.float32)
        cdef float[:,:] inter_z = np.zeros((nb_pass, D_BINS+1), dtype = np.float32)
        cdef float[:,:] minor_x = np.zeros((nb_pass, D_BINS+1), dtype = np.float32)
        cdef float[:,:] minor_y = np.zeros((nb_pass, D_BINS+1), dtype = np.float32)
        cdef float[:,:] minor_z = np.zeros((nb_pass, D_BINS+1), dtype = np.float32)
        cdef float[:,:,:] morph_info = np.zeros((openmp.omp_get_max_threads(), 12, D_BINS+1), dtype = np.float32)
        cdef float[:,:,:] xyz_obj = np.zeros((openmp.omp_get_max_threads(), cat_arr.shape[1],3), dtype = np.float32)
        cdef float[:,:,:] xyz_princ = np.zeros((openmp.omp_get_max_threads(), cat_arr.shape[1],3), dtype = np.float32)
        cdef int[:,:] shell = np.zeros((openmp.omp_get_max_threads(), cat_arr.shape[1]), dtype = np.int32)
        cdef float[:,:] r_ell = np.zeros((openmp.omp_get_max_threads(), cat_arr.shape[1]), dtype = np.float32)
        cdef complex[::1,:,:] shape_tensor = np.zeros((3, 3, openmp.omp_get_max_threads()), dtype = np.complex128, order='F')
        cdef double[::1,:] eigval = np.zeros((3, openmp.omp_get_max_threads()), dtype=np.float64, order='F')
        cdef complex[::1,:,:] eigvec = np.zeros((3,3, openmp.omp_get_max_threads()), dtype=np.complex128, order='F')
        cdef float[:,:] m_obj = np.zeros((openmp.omp_get_max_threads(), cat_arr.shape[1]), dtype = np.float32)
        cdef float[:,:] log_d_tiled = np.reshape(np.tile(np.logspace(D_LOGSTART,D_LOGEND,D_BINS+1, dtype = np.float32), reps = openmp.omp_get_max_threads()), (openmp.omp_get_max_threads(), D_BINS+1))
        cdef int n
        cdef int r
        cdef float[:,:] centers = np.zeros((nb_pass,3), dtype = np.float32)
        for p in range(nb_objs): # Calculate centers of objects
            if obj_pass[p] == 1:
                xyz_ = respectPBCNoRef(xyz.base[cat_arr.base[idxs_compr[p],:obj_size[p]]], L_BOX)
                if CENTER == 'mode':
                    centers.base[idxs_compr[p]] = calcMode(xyz_, masses.base[cat_arr.base[idxs_compr[p],:obj_size[p]]], max((max(xyz_[:,0])-min(xyz_[:,0]), max(xyz_[:,1])-min(xyz_[:,1]), max(xyz_[:,2])-min(xyz_[:,2]))))
                else:
                    centers.base[idxs_compr[p]] = calcCoM(xyz_, masses.base[cat_arr.base[idxs_compr[p],:obj_size[p]]])
        for p in prange(nb_objs, schedule = 'dynamic', nogil = True):
            if obj_pass[p] == 1:
                for n in range(obj_size[p]):
                    xyz_obj[openmp.omp_get_thread_num(),n,0] = xyz[cat_arr[idxs_compr[p],n],0]
                    xyz_obj[openmp.omp_get_thread_num(),n,1] = xyz[cat_arr[idxs_compr[p],n],1]
                    xyz_obj[openmp.omp_get_thread_num(),n,2] = xyz[cat_arr[idxs_compr[p],n],2]
                    m_obj[openmp.omp_get_thread_num(),n] = masses[cat_arr[idxs_compr[p],n]]
                    m[idxs_compr[p]] = m[idxs_compr[p]] + masses[cat_arr[idxs_compr[p],n]]
                xyz_obj[openmp.omp_get_thread_num(),:obj_size[p]] = CythonHelpers.respectPBCNoRef(xyz_obj[openmp.omp_get_thread_num(),:obj_size[p]], L_BOX)
                morph_info[openmp.omp_get_thread_num(),:,:] = calcObjMorphLocal(morph_info[openmp.omp_get_thread_num(),:,:], r200[p], log_d_tiled[openmp.omp_get_thread_num()], xyz_obj[openmp.omp_get_thread_num(),:obj_size[p]], xyz_princ[openmp.omp_get_thread_num(),:obj_size[p]], m_obj[openmp.omp_get_thread_num(),:obj_size[p]], shell[openmp.omp_get_thread_num()], r_ell[openmp.omp_get_thread_num()], centers[idxs_compr[p]], shape_tensor[:,:,openmp.omp_get_thread_num()], eigval[:,openmp.omp_get_thread_num()], eigvec[:,:,openmp.omp_get_thread_num()], M_TOL, N_WALL, N_MIN, reduced, shell_based)
                d[idxs_compr[p]] = morph_info[openmp.omp_get_thread_num(),0]
                q[idxs_compr[p]] = morph_info[openmp.omp_get_thread_num(),1]
                s[idxs_compr[p]] = morph_info[openmp.omp_get_thread_num(),2]
                major_x[idxs_compr[p]] = morph_info[openmp.omp_get_thread_num(),3]
                major_y[idxs_compr[p]] = morph_info[openmp.omp_get_thread_num(),4]
                major_z[idxs_compr[p]] = morph_info[openmp.omp_get_thread_num(),5]
                inter_x[idxs_compr[p]] = morph_info[openmp.omp_get_thread_num(),6]
                inter_y[idxs_compr[p]] = morph_info[openmp.omp_get_thread_num(),7]
                inter_z[idxs_compr[p]] = morph_info[openmp.omp_get_thread_num(),8]
                minor_x[idxs_compr[p]] = morph_info[openmp.omp_get_thread_num(),9]
                minor_y[idxs_compr[p]] = morph_info[openmp.omp_get_thread_num(),10]
                minor_z[idxs_compr[p]] = morph_info[openmp.omp_get_thread_num(),11]
                printf("Calculating shape profile. Dealing with object number %d. The number of ptcs is %d.\n", p, obj_size[p])
            
        minor = np.transpose(np.stack((minor_x.base,minor_y.base,minor_z.base)),(1,2,0))
        inter = np.transpose(np.stack((inter_x.base,inter_y.base,inter_z.base)),(1,2,0))
        major = np.transpose(np.stack((major_x.base,major_y.base,major_z.base)),(1,2,0))
        d.base[d.base==0.0] = np.nan
        s.base[s.base==0.0] = np.nan
        q.base[q.base==0.0] = np.nan
        minor[minor==0.0] = np.nan
        inter[inter==0.0] = np.nan
        major[major==0.0] = np.nan
        del xyz; del masses; del r200; del cat
        del cat_arr; del major_x; del major_y; del major_z; del inter_x; del inter_y; del inter_z; del minor_x; del minor_y; del minor_z
        del morph_info; del xyz_obj; del xyz_princ; del shell; del r_ell; del shape_tensor; del eigval; del eigvec; del m_obj; del log_d_tiled
        return d.base, q.base, s.base, minor, inter, major, centers.base, m.base # Only rank = 0 content matters
    if(not hasattr(calcMorphLocal, "inner")):
        calcMorphLocal.inner = np_cache_factory(3,1)(inner)
    calcMorphLocal.inner(xyz.base, masses.base, r200.base, cat, L_BOX, MIN_NUMBER_PTCS, D_LOGSTART, D_LOGEND, D_BINS, M_TOL, N_WALL, N_MIN, CENTER, reduced, shell_based)
    return calcMorphLocal.inner(xyz.base, masses.base, r200.base, cat, L_BOX, MIN_NUMBER_PTCS, D_LOGSTART, D_LOGEND, D_BINS, M_TOL, N_WALL, N_MIN, CENTER, reduced, shell_based)

@cython.embedsignature(True)
@cython.binding(True)
def calcMorphGlobal(float[:,:] xyz, float[:] masses, float[:] r200, cat, float L_BOX, int MIN_NUMBER_PTCS, int M_TOL, int N_WALL, int N_MIN, str CENTER, float SAFE, bint reduced):
    """ Calculates the overall shape catalogue
    
    Calls ``calcObjMorphGlobal()`` in a parallelized manner.\n
    Calculates the overall axis ratios and eigenframe for each object.
    
    :param xyz: positions of all (DM or star) particles in simulation box
    :type xyz: (N2 x 3) floats
    :param cat: each entry of the list is a list containing indices of particles belonging to an object
    :type cat: list of length N1
    :param masses: masses of the particles expressed in unit mass
    :type masses: (N2 x 1) floats
    :param r200: each entry of the list gives the R_200 radius of the parent halo
    :type r200: list of length N1
    :param L_BOX: simulation box side length
    :type L_BOX: float, units: Mpc/h
    :param MIN_NUMBER_PTCS: minimum number of particles for object to qualify for morphology calculation
    :type MIN_NUMBER_PTCS: int
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
    :param SAFE: ellipsoidal radius will be maxdist(COM,point)+SAFE where point is any point in the point cloud. 
        The larger the better.
    :type SAFE: float
    :param reduced: whether or not reduced shape tensor (1/r^2 factor)
    :type reduced: boolean
    :return: d, q, s, eigframe, centers, masses
    :rtype: (N3,) floats (for d, q, s, eigframe (x3)), (N3, 3) floats (for centers), (N3,) floats (for masses)
    """
    def inner(float[:,:] xyz, float[:] masses, float[:] r200, cat, float L_BOX, int MIN_NUMBER_PTCS, int M_TOL, int N_WALL, int N_MIN, str CENTER, float SAFE, bint reduced):
        # Transform cat to int[:,:]
        cdef int nb_objs = len(cat)
        cdef int p
        cdef int[:] obj_pass = np.zeros((nb_objs,), dtype = np.int32)
        cdef int[:] obj_size = np.zeros((nb_objs,), dtype = np.int32)
        for p in range(nb_objs):
            if len(cat[p]) >= MIN_NUMBER_PTCS: # Only add objects that have sufficient resolution
                obj_pass[p] = 1      
                obj_size[p] = len(cat[p]) 
        cdef int nb_pass = np.sum(obj_pass.base)
        cdef int[:,:] cat_arr = np.zeros((nb_pass,np.max([len(cat[p]) for p in range(nb_objs)])), dtype = np.int32)
        cdef int[:] idxs_compr = np.zeros((nb_objs,), dtype = np.int32)
        idxs_compr.base[obj_pass.base.nonzero()[0]] = np.arange(np.sum(obj_pass.base))
        for p in range(nb_objs):
            if obj_pass[p] == 1:
                cat_arr.base[idxs_compr[p],:obj_size[p]] = np.array(cat[p])
    
        cdef float[:] m = np.zeros((nb_pass,), dtype = np.float32)
        cdef float[:] d = np.zeros((nb_pass,), dtype = np.float32)
        cdef float[:] q = np.zeros((nb_pass,), dtype = np.float32)
        cdef float[:] s = np.zeros((nb_pass,), dtype = np.float32)
        cdef float[:] major_x = np.zeros((nb_pass,), dtype = np.float32)
        cdef float[:] major_y = np.zeros((nb_pass,), dtype = np.float32)
        cdef float[:] major_z = np.zeros((nb_pass,), dtype = np.float32)
        cdef float[:] inter_x = np.zeros((nb_pass,), dtype = np.float32)
        cdef float[:] inter_y = np.zeros((nb_pass,), dtype = np.float32)
        cdef float[:] inter_z = np.zeros((nb_pass,), dtype = np.float32)
        cdef float[:] minor_x = np.zeros((nb_pass,), dtype = np.float32)
        cdef float[:] minor_y = np.zeros((nb_pass,), dtype = np.float32)
        cdef float[:] minor_z = np.zeros((nb_pass,), dtype = np.float32)
        cdef float[:,:] centers = np.zeros((nb_pass,3), dtype = np.float32)
        cdef float[:,:] morph_info = np.zeros((openmp.omp_get_max_threads(), 12), dtype = np.float32)
        cdef float[:,:,:] xyz_obj = np.zeros((openmp.omp_get_max_threads(), cat_arr.shape[1],3), dtype = np.float32)
        cdef float[:,:,:] xyz_princ = np.zeros((openmp.omp_get_max_threads(), cat_arr.shape[1],3), dtype = np.float32)
        cdef int[:,:] ellipsoid = np.zeros((openmp.omp_get_max_threads(), cat_arr.shape[1]), dtype = np.int32)
        cdef float[:,:] r_ell = np.zeros((openmp.omp_get_max_threads(), cat_arr.shape[1]), dtype = np.float32)
        cdef complex[::1,:,:] shape_tensor = np.zeros((3, 3, openmp.omp_get_max_threads()), dtype = np.complex128, order='F')
        cdef double[::1,:] eigval = np.zeros((3, openmp.omp_get_max_threads()), dtype=np.float64, order='F')
        cdef complex[::1,:,:] eigvec = np.zeros((3,3, openmp.omp_get_max_threads()), dtype=np.complex128, order='F')
        cdef float[:,:] m_obj = np.zeros((openmp.omp_get_max_threads(), cat_arr.shape[1]), dtype = np.float32)
        cdef int n
        for p in range(nb_objs): # Calculate centers of objects
            if obj_pass[p] == 1:
                xyz_ = respectPBCNoRef(xyz.base[cat_arr.base[idxs_compr[p],:obj_size[p]]], L_BOX)
                if CENTER == 'mode':
                    centers.base[idxs_compr[p]] = calcMode(xyz_, masses.base[cat_arr.base[idxs_compr[p],:obj_size[p]]], max((max(xyz_[:,0])-min(xyz_[:,0]), max(xyz_[:,1])-min(xyz_[:,1]), max(xyz_[:,2])-min(xyz_[:,2]))))
                else:
                    centers.base[idxs_compr[p]] = calcCoM(xyz_, masses.base[cat_arr.base[idxs_compr[p],:obj_size[p]]])
        for p in prange(nb_objs, schedule = 'dynamic', nogil = True):
            if obj_pass[p] == 1:
                for n in range(obj_size[p]):
                    xyz_obj[openmp.omp_get_thread_num(),n] = xyz[cat_arr[idxs_compr[p],n]]
                    m_obj[openmp.omp_get_thread_num(),n] = masses[cat_arr[idxs_compr[p],n]]
                    m[idxs_compr[p]] = m[idxs_compr[p]] + masses[cat_arr[idxs_compr[p],n]]
                xyz_obj[openmp.omp_get_thread_num(),:obj_size[p]] = CythonHelpers.respectPBCNoRef(xyz_obj[openmp.omp_get_thread_num(),:obj_size[p]], L_BOX)
                morph_info[openmp.omp_get_thread_num(),:] = calcObjMorphGlobal(morph_info[openmp.omp_get_thread_num(),:], r200[p], xyz_obj[openmp.omp_get_thread_num(),:obj_size[p]], xyz_princ[openmp.omp_get_thread_num(),:obj_size[p]], m_obj[openmp.omp_get_thread_num(),:obj_size[p]], ellipsoid[openmp.omp_get_thread_num()], r_ell[openmp.omp_get_thread_num()], centers[idxs_compr[p]], shape_tensor[:,:,openmp.omp_get_thread_num()], eigval[:,openmp.omp_get_thread_num()], eigvec[:,:,openmp.omp_get_thread_num()], M_TOL, N_WALL, N_MIN, SAFE, reduced)
                d[idxs_compr[p]] = morph_info[openmp.omp_get_thread_num(),0]
                q[idxs_compr[p]] = morph_info[openmp.omp_get_thread_num(),1]
                s[idxs_compr[p]] = morph_info[openmp.omp_get_thread_num(),2]
                major_x[idxs_compr[p]] = morph_info[openmp.omp_get_thread_num(),3]
                major_y[idxs_compr[p]] = morph_info[openmp.omp_get_thread_num(),4]
                major_z[idxs_compr[p]] = morph_info[openmp.omp_get_thread_num(),5]
                inter_x[idxs_compr[p]] = morph_info[openmp.omp_get_thread_num(),6]
                inter_y[idxs_compr[p]] = morph_info[openmp.omp_get_thread_num(),7]
                inter_z[idxs_compr[p]] = morph_info[openmp.omp_get_thread_num(),8]
                minor_x[idxs_compr[p]] = morph_info[openmp.omp_get_thread_num(),9]
                minor_y[idxs_compr[p]] = morph_info[openmp.omp_get_thread_num(),10]
                minor_z[idxs_compr[p]] = morph_info[openmp.omp_get_thread_num(),11]
                printf("Calculating overall shapes. Dealing with object number %d. The number of ptcs is %d.\n", p, obj_size[p])
        
        minor = np.hstack((np.reshape(minor_x.base, (minor_x.base.shape[0],1)), np.reshape(minor_y.base, (minor_y.base.shape[0],1)), np.reshape(minor_z.base, (minor_z.base.shape[0],1))))
        inter = np.hstack((np.reshape(inter_x.base, (inter_x.base.shape[0],1)), np.reshape(inter_y.base, (inter_y.base.shape[0],1)), np.reshape(inter_z.base, (inter_z.base.shape[0],1))))
        major = np.hstack((np.reshape(major_x.base, (major_x.base.shape[0],1)), np.reshape(major_y.base, (major_y.base.shape[0],1)), np.reshape(major_z.base, (major_z.base.shape[0],1))))
        d.base[d.base==0.0] = np.nan
        s.base[s.base==0.0] = np.nan
        q.base[q.base==0.0] = np.nan
        minor[minor==0.0] = np.nan
        inter[inter==0.0] = np.nan
        major[major==0.0] = np.nan
        del xyz; del masses; del r200; del cat
        del major_x; del major_y; del major_z; del inter_x; del inter_y; del inter_z; del minor_x; del minor_y; del minor_z
        del morph_info; del xyz_obj; del xyz_princ; del ellipsoid; del r_ell; del shape_tensor; del eigval; del eigvec; del m_obj
        return d.base, q.base, s.base, minor, inter, major, centers.base, m.base # Only rank = 0 content matters
    if(not hasattr(calcMorphGlobal, "inner")):
        calcMorphGlobal.inner = np_cache_factory(3,1)(inner)
    calcMorphGlobal.inner(xyz.base, masses.base, r200.base, cat, L_BOX, MIN_NUMBER_PTCS, M_TOL, N_WALL, N_MIN, CENTER, SAFE, reduced)
    return calcMorphGlobal.inner(xyz.base, masses.base, r200.base, cat, L_BOX, MIN_NUMBER_PTCS, M_TOL, N_WALL, N_MIN, CENTER, SAFE, reduced)

@cython.embedsignature(True)
@cython.binding(True)
def calcMorphLocalVelDisp(float[:,:] xyz, float[:,:] vxyz, float[:] masses, float[:] r200, cat, float L_BOX, int MIN_NUMBER_PTCS, int D_LOGSTART, int D_LOGEND, int D_BINS, int M_TOL, int N_WALL, int N_MIN, str CENTER, bint reduced, bint shell_based):
    """ Calculates the local velocity dispersion shape catalogue
    
    Calls ``calcObjMorphLocalVelDisp()`` in a parallelized manner.\n
    Calculates the overall axis ratios and eigenframe for each object.
    
    :param xyz: positions of all (DM or star) particles in simulation box
    :type xyz: (N2 x 3) floats
    :param vxyz: velocities of all (DM or star) particles in simulation box
    :type vxyz: (N2 x 3) floats
    :param cat: each entry of the list is a list containing indices of particles belonging to an object
    :type cat: list of length N2
    :param masses: masses of the particles expressed in unit mass
    :type masses: (N2 x 1) floats
    :param r200: each entry of the list gives the R_200 radius of the parent halo
    :type r200: list of length N1
    :param L_BOX: simulation box side length
    :type L_BOX: float, units: Mpc/h
    :param MIN_NUMBER_PTCS: minimum number of particles for object to qualify for morphology calculation
    :type MIN_NUMBER_PTCS: int
    :param D_LOGSTART: logarithm of minimum ellipsoidal radius of interest, in units of R200 of parent halo
    :type D_LOGSTART: int
    :param D_LOGEND: logarithm of maximum ellipsoidal radius of interest, in units of R200 of parent halo
    :type D_LOGEND: int
    :param D_BINS: number of ellipsoidal radii of interest mi
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
    :param reduced: whether or not reduced shape tensor (1/r^2 factor)
    :type reduced: boolean
    :param shell_based: whether shell-based or ellipsoid-based algorithm should be run
    :type shell_based: boolean
    :return: d, q, s, eigframe, centers, masses, l_succeed: list of object indices for which morphology could be determined at R200 (length: N3)
    :rtype: (N3, ``D_BINS`` + 1) floats (for d, q, s, eigframe (x3)), (N3, 3) floats (for centers), (N3,) floats (for masses), N3-list of ints for l_succeed
    """
    def inner(float[:,:] xyz, float[:,:] vxyz, float[:] masses, float[:] r200, cat, float L_BOX, int MIN_NUMBER_PTCS, int D_LOGSTART, int D_LOGEND, int D_BINS, int M_TOL, int N_WALL, int N_MIN, str CENTER, bint reduced, bint shell_based):
        # Transform cat to int[:,:]
        cdef int nb_objs = len(cat)
        cdef int p
        cdef int[:] obj_pass = np.zeros((nb_objs,), dtype = np.int32)
        cdef int[:] obj_size = np.zeros((nb_objs,), dtype = np.int32)
        for p in range(nb_objs):
            if len(cat[p]) >= MIN_NUMBER_PTCS: # Only add objects that have sufficient resolution
                obj_pass[p] = 1      
                obj_size[p] = len(cat[p]) 
        cdef int nb_pass = np.sum(obj_pass.base)
        cdef int[:,:] cat_arr = np.zeros((nb_pass,np.max([len(cat[p]) for p in range(nb_objs)])), dtype = np.int32)
        cdef int[:] idxs_compr = np.zeros((nb_objs,), dtype = np.int32)
        idxs_compr.base[obj_pass.base.nonzero()[0]] = np.arange(np.sum(obj_pass.base))
        for p in range(nb_objs):
            if obj_pass[p] == 1:
                cat_arr.base[idxs_compr[p],:obj_size[p]] = np.array(cat[p])
    
        cdef float[:] m = np.zeros((nb_pass,), dtype = np.float32)
        cdef float[:,:] d = np.zeros((nb_pass, D_BINS+1), dtype = np.float32)
        cdef float[:,:] q = np.zeros((nb_pass, D_BINS+1), dtype = np.float32)
        cdef float[:,:] s = np.zeros((nb_pass, D_BINS+1), dtype = np.float32)
        cdef float[:,:] major_x = np.zeros((nb_pass, D_BINS+1), dtype = np.float32)
        cdef float[:,:] major_y = np.zeros((nb_pass, D_BINS+1), dtype = np.float32)
        cdef float[:,:] major_z = np.zeros((nb_pass, D_BINS+1), dtype = np.float32)
        cdef float[:,:] inter_x = np.zeros((nb_pass, D_BINS+1), dtype = np.float32)
        cdef float[:,:] inter_y = np.zeros((nb_pass, D_BINS+1), dtype = np.float32)
        cdef float[:,:] inter_z = np.zeros((nb_pass, D_BINS+1), dtype = np.float32)
        cdef float[:,:] minor_x = np.zeros((nb_pass, D_BINS+1), dtype = np.float32)
        cdef float[:,:] minor_y = np.zeros((nb_pass, D_BINS+1), dtype = np.float32)
        cdef float[:,:] minor_z = np.zeros((nb_pass, D_BINS+1), dtype = np.float32)
        cdef float[:,:] centers = np.zeros((nb_pass,3), dtype = np.float32)
        cdef float[:,:] vcenters = np.zeros((nb_pass,3), dtype = np.float32)
        cdef float[:,:,:] morph_info = np.zeros((openmp.omp_get_max_threads(), 12, D_BINS+1), dtype = np.float32)
        cdef float[:,:,:] xyz_obj = np.zeros((openmp.omp_get_max_threads(), cat_arr.shape[1],3), dtype = np.float32)
        cdef float[:,:,:] vxyz_obj = np.zeros((openmp.omp_get_max_threads(), cat_arr.shape[1],3), dtype = np.float32)
        cdef float[:,:,:] xyz_princ = np.zeros((openmp.omp_get_max_threads(), cat_arr.shape[1],3), dtype = np.float32)
        cdef int[:,:] shell = np.zeros((openmp.omp_get_max_threads(), cat_arr.shape[1]), dtype = np.int32)
        cdef float[:,:] r_ell = np.zeros((openmp.omp_get_max_threads(), cat_arr.shape[1]), dtype = np.float32)
        cdef complex[::1,:,:] shape_tensor = np.zeros((3, 3, openmp.omp_get_max_threads()), dtype = np.complex128, order='F')
        cdef double[::1,:] eigval = np.zeros((3, openmp.omp_get_max_threads()), dtype=np.float64, order='F')
        cdef complex[::1,:,:] eigvec = np.zeros((3,3, openmp.omp_get_max_threads()), dtype=np.complex128, order='F')
        cdef float[:,:] m_obj = np.zeros((openmp.omp_get_max_threads(), cat_arr.shape[1]), dtype = np.float32)
        cdef float[:,:] log_d_tiled = np.reshape(np.tile(np.logspace(D_LOGSTART,D_LOGEND,D_BINS+1, dtype = np.float32), reps = openmp.omp_get_max_threads()), (openmp.omp_get_max_threads(), D_BINS+1))
        cdef int n
        cdef int r
        for p in range(nb_objs): # Calculate centers of objects
            if obj_pass[p] == 1:
                xyz_ = respectPBCNoRef(xyz.base[cat_arr.base[idxs_compr[p],:obj_size[p]]], L_BOX)
                if CENTER == 'mode':
                    centers.base[idxs_compr[p]] = calcMode(xyz_, masses.base[cat_arr.base[idxs_compr[p],:obj_size[p]]], max((max(xyz_[:,0])-min(xyz_[:,0]), max(xyz_[:,1])-min(xyz_[:,1]), max(xyz_[:,2])-min(xyz_[:,2]))))
                else:
                    centers.base[idxs_compr[p]] = calcCoM(xyz_, masses.base[cat_arr.base[idxs_compr[p],:obj_size[p]]])
        for p in prange(nb_objs, schedule = 'dynamic', nogil = True):
            if obj_pass[p] == 1:
                for n in range(obj_size[p]):
                    xyz_obj[openmp.omp_get_thread_num(),n,0] = xyz[cat_arr[idxs_compr[p],n],0]
                    xyz_obj[openmp.omp_get_thread_num(),n,1] = xyz[cat_arr[idxs_compr[p],n],1]
                    xyz_obj[openmp.omp_get_thread_num(),n,2] = xyz[cat_arr[idxs_compr[p],n],2]
                    m_obj[openmp.omp_get_thread_num(),n] = masses[cat_arr[idxs_compr[p],n]]
                    m[idxs_compr[p]] = m[idxs_compr[p]] + masses[cat_arr[idxs_compr[p],n]]
                xyz_obj[openmp.omp_get_thread_num(),:obj_size[p]] = CythonHelpers.respectPBCNoRef(xyz_obj[openmp.omp_get_thread_num(),:obj_size[p]], L_BOX)
                vcenters[idxs_compr[p]] = CythonHelpers.calcCoM(vxyz_obj[openmp.omp_get_thread_num(),:obj_size[p]], m_obj[openmp.omp_get_thread_num(),:obj_size[p]], vcenters[p])
                morph_info[openmp.omp_get_thread_num(),:,:] = calcObjMorphLocalVelDisp(morph_info[openmp.omp_get_thread_num(),:,:], r200[p], log_d_tiled[openmp.omp_get_thread_num()], xyz_obj[openmp.omp_get_thread_num(),:obj_size[p]], vxyz_obj[openmp.omp_get_thread_num(),:obj_size[p]], xyz_princ[openmp.omp_get_thread_num(),:obj_size[p]], m_obj[openmp.omp_get_thread_num(),:obj_size[p]], shell[openmp.omp_get_thread_num()], r_ell[openmp.omp_get_thread_num()], centers[idxs_compr[p]], vcenters[idxs_compr[p]], shape_tensor[:,:,openmp.omp_get_thread_num()], eigval[:,openmp.omp_get_thread_num()], eigvec[:,:,openmp.omp_get_thread_num()], M_TOL, N_WALL, N_MIN, reduced, shell_based)
                d[idxs_compr[p]] = morph_info[openmp.omp_get_thread_num(),0]
                q[idxs_compr[p]] = morph_info[openmp.omp_get_thread_num(),1]
                s[idxs_compr[p]] = morph_info[openmp.omp_get_thread_num(),2]
                major_x[idxs_compr[p]] = morph_info[openmp.omp_get_thread_num(),3]
                major_y[idxs_compr[p]] = morph_info[openmp.omp_get_thread_num(),4]
                major_z[idxs_compr[p]] = morph_info[openmp.omp_get_thread_num(),5]
                inter_x[idxs_compr[p]] = morph_info[openmp.omp_get_thread_num(),6]
                inter_y[idxs_compr[p]] = morph_info[openmp.omp_get_thread_num(),7]
                inter_z[idxs_compr[p]] = morph_info[openmp.omp_get_thread_num(),8]
                minor_x[idxs_compr[p]] = morph_info[openmp.omp_get_thread_num(),9]
                minor_y[idxs_compr[p]] = morph_info[openmp.omp_get_thread_num(),10]
                minor_z[idxs_compr[p]] = morph_info[openmp.omp_get_thread_num(),11]
                printf("Calculating velocity dispersion shape profiles. Dealing with object number %d. The number of ptcs is %d.\n", p, obj_size[p])
        
        minor = np.transpose(np.stack((minor_x.base,minor_y.base,minor_z.base)),(1,2,0))
        inter = np.transpose(np.stack((inter_x.base,inter_y.base,inter_z.base)),(1,2,0))
        major = np.transpose(np.stack((major_x.base,major_y.base,major_z.base)),(1,2,0))
        d.base[d.base==0.0] = np.nan
        s.base[s.base==0.0] = np.nan
        q.base[q.base==0.0] = np.nan
        minor[minor==0.0] = np.nan
        inter[inter==0.0] = np.nan
        major[major==0.0] = np.nan
        del xyz; del vxyz; del masses; del r200; del cat
        del major_x; del major_y; del major_z; del inter_x; del inter_y; del inter_z; del minor_x; del minor_y; del minor_z
        del morph_info; del xyz_obj; del vxyz_obj; del xyz_princ; del shell; del r_ell; del shape_tensor; del eigval; del eigvec; del m_obj; del log_d_tiled
        return d.base, q.base, s.base, minor, inter, major, centers.base, m.base # Only rank = 0 content matters
    if(not hasattr(calcMorphLocalVelDisp, "inner")):
        calcMorphLocalVelDisp.inner = np_cache_factory(4,1)(inner)
    calcMorphLocalVelDisp.inner(xyz.base, vxyz.base, masses.base, r200.base, cat, L_BOX, MIN_NUMBER_PTCS, D_LOGSTART, D_LOGEND, D_BINS, M_TOL, N_WALL, N_MIN, CENTER, reduced, shell_based)
    return calcMorphLocalVelDisp.inner(xyz.base, vxyz.base, masses.base, r200.base, cat, L_BOX, MIN_NUMBER_PTCS, D_LOGSTART, D_LOGEND, D_BINS, M_TOL, N_WALL, N_MIN, CENTER, reduced, shell_based)

@cython.embedsignature(True)
@cython.binding(True)
def calcMorphGlobalVelDisp(float[:,:] xyz, float[:,:] vxyz, float[:] masses, float[:] r200, cat, float L_BOX, int MIN_NUMBER_PTCS, int M_TOL, int N_WALL, int N_MIN, str CENTER, float SAFE, bint reduced):
    """ Calculates the global velocity dipsersion shape catalogue
    
    Calls ``calcObjMorphGlobalVelDisp()`` in a parallelized manner.\n
    Calculates the overall axis ratios and eigenframe for each object.
    
    :param xyz: positions of all (DM or star) particles in simulation box
    :type xyz: (N2 x 3) floats
    :param vxyz: velocities of all (DM or star) particles in simulation box
    :type vxyz: (N2 x 3) floats
    :param cat: each entry of the list is a list containing indices of particles belonging to an object
    :type cat: list of length N2
    :param masses: masses of the particles expressed in unit mass
    :type masses: (N2 x 1) floats
    :param r200: R_200 radii of the parent halos
    :type r200: (N1,) floats
    :param L_BOX: simulation box side length
    :type L_BOX: float, units: Mpc/h
    :param MIN_NUMBER_PTCS: minimum number of particles for object to qualify for morphology calculation
    :type MIN_NUMBER_PTCS: int
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
    :param SAFE: ellipsoidal radius will be maxdist(COM,point)+SAFE where point is any point in the point cloud. 
        The larger the better.
    :type SAFE: float
    :param reduced: whether or not reduced shape tensor (1/r^2 factor)
    :type reduced: boolean
    :return: d, q, s, eigframe, centers, masses
    :rtype: (N3, ``D_BINS`` + 1) floats (for d, q, s, eigframe (x3)), (N3, 3) floats (for centers), (N3,) floats (for masses)
    """
    def inner(float[:,:] xyz, float[:,:] vxyz, float[:] masses, float[:] r200, cat, float L_BOX, int MIN_NUMBER_PTCS, int M_TOL, int N_WALL, int N_MIN, str CENTER, float SAFE, bint reduced):
        # Transform cat to int[:,:]
        cdef int nb_objs = len(cat)
        cdef int p
        cdef int[:] obj_pass = np.zeros((nb_objs,), dtype = np.int32)
        cdef int[:] obj_size = np.zeros((nb_objs,), dtype = np.int32)
        for p in range(nb_objs):
            if len(cat[p]) >= MIN_NUMBER_PTCS: # Only add objects that have sufficient resolution
                obj_pass[p] = 1      
                obj_size[p] = len(cat[p]) 
        cdef int nb_pass = np.sum(obj_pass.base)
        cdef int[:,:] cat_arr = np.zeros((nb_pass,np.max([len(cat[p]) for p in range(nb_objs)])), dtype = np.int32)
        cdef int[:] idxs_compr = np.zeros((nb_objs,), dtype = np.int32)
        idxs_compr.base[obj_pass.base.nonzero()[0]] = np.arange(np.sum(obj_pass.base))
        for p in range(nb_objs):
            if obj_pass[p] == 1:
                cat_arr.base[idxs_compr[p],:obj_size[p]] = np.array(cat[p])
    
        cdef float[:] m = np.zeros((nb_pass,), dtype = np.float32)
        cdef float[:] d = np.zeros((nb_pass,), dtype = np.float32)
        cdef float[:] q = np.zeros((nb_pass,), dtype = np.float32)
        cdef float[:] s = np.zeros((nb_pass,), dtype = np.float32)
        cdef float[:] major_x = np.zeros((nb_pass,), dtype = np.float32)
        cdef float[:] major_y = np.zeros((nb_pass,), dtype = np.float32)
        cdef float[:] major_z = np.zeros((nb_pass,), dtype = np.float32)
        cdef float[:] inter_x = np.zeros((nb_pass,), dtype = np.float32)
        cdef float[:] inter_y = np.zeros((nb_pass,), dtype = np.float32)
        cdef float[:] inter_z = np.zeros((nb_pass,), dtype = np.float32)
        cdef float[:] minor_x = np.zeros((nb_pass,), dtype = np.float32)
        cdef float[:] minor_y = np.zeros((nb_pass,), dtype = np.float32)
        cdef float[:] minor_z = np.zeros((nb_pass,), dtype = np.float32)
        cdef float[:,:] centers = np.zeros((nb_pass,3), dtype = np.float32)
        cdef float[:,:] vcenters = np.zeros((nb_pass,3), dtype = np.float32)
        cdef float[:,:] morph_info = np.zeros((openmp.omp_get_max_threads(), 12), dtype = np.float32)
        cdef float[:,:,:] xyz_obj = np.zeros((openmp.omp_get_max_threads(), cat_arr.shape[1],3), dtype = np.float32)
        cdef float[:,:,:] vxyz_obj = np.zeros((openmp.omp_get_max_threads(), cat_arr.shape[1],3), dtype = np.float32)
        cdef float[:,:,:] xyz_princ = np.zeros((openmp.omp_get_max_threads(), cat_arr.shape[1],3), dtype = np.float32)
        cdef int[:,:] ellipsoid = np.zeros((openmp.omp_get_max_threads(), cat_arr.shape[1]), dtype = np.int32)
        cdef float[:,:] r_ell = np.zeros((openmp.omp_get_max_threads(), cat_arr.shape[1]), dtype = np.float32)
        cdef complex[::1,:,:] shape_tensor = np.zeros((3, 3, openmp.omp_get_max_threads()), dtype = np.complex128, order='F')
        cdef double[::1,:] eigval = np.zeros((3, openmp.omp_get_max_threads()), dtype=np.float64, order='F')
        cdef complex[::1,:,:] eigvec = np.zeros((3,3, openmp.omp_get_max_threads()), dtype=np.complex128, order='F')
        cdef float[:,:] m_obj = np.zeros((openmp.omp_get_max_threads(), cat_arr.shape[1]), dtype = np.float32)
        cdef int n
        for p in range(nb_objs): # Calculate centers of objects
            if obj_pass[p] == 1:
                xyz_ = respectPBCNoRef(xyz.base[cat_arr.base[idxs_compr[p],:obj_size[p]]], L_BOX)
                if CENTER == 'mode':
                    centers.base[idxs_compr[p]] = calcMode(xyz_, masses.base[cat_arr.base[idxs_compr[p],:obj_size[p]]], max((max(xyz_[:,0])-min(xyz_[:,0]), max(xyz_[:,1])-min(xyz_[:,1]), max(xyz_[:,2])-min(xyz_[:,2]))))
                else:
                    centers.base[idxs_compr[p]] = calcCoM(xyz_, masses.base[cat_arr.base[idxs_compr[p],:obj_size[p]]])
        for p in prange(nb_objs, schedule = 'dynamic', nogil = True):
            if obj_pass[p] == 1:
                for n in range(obj_size[p]):
                    xyz_obj[openmp.omp_get_thread_num(),n] = xyz[cat_arr[idxs_compr[p],n]]
                    vxyz_obj[openmp.omp_get_thread_num(),n] = vxyz[cat_arr[idxs_compr[p],n]]
                    m_obj[openmp.omp_get_thread_num(),n] = masses[cat_arr[idxs_compr[p],n]]
                    m[idxs_compr[p]] = m[idxs_compr[p]] + masses[cat_arr[idxs_compr[p],n]]
                xyz_obj[openmp.omp_get_thread_num(),:obj_size[p]] = CythonHelpers.respectPBCNoRef(xyz_obj[openmp.omp_get_thread_num(),:obj_size[p]], L_BOX)
                vcenters[idxs_compr[p]] = CythonHelpers.calcCoM(vxyz_obj[openmp.omp_get_thread_num(),:obj_size[p]], m_obj[openmp.omp_get_thread_num(),:obj_size[p]], vcenters[p])
                morph_info[openmp.omp_get_thread_num(),:] = calcObjMorphGlobalVelDisp(morph_info[openmp.omp_get_thread_num(),:], r200[p], xyz_obj[openmp.omp_get_thread_num(),:obj_size[p]], vxyz_obj[openmp.omp_get_thread_num(),:obj_size[p]], xyz_princ[openmp.omp_get_thread_num(),:obj_size[p]], m_obj[openmp.omp_get_thread_num(),:obj_size[p]], ellipsoid[openmp.omp_get_thread_num()], r_ell[openmp.omp_get_thread_num()], centers[idxs_compr[p]], vcenters[idxs_compr[p]], shape_tensor[:,:,openmp.omp_get_thread_num()], eigval[:,openmp.omp_get_thread_num()], eigvec[:,:,openmp.omp_get_thread_num()], M_TOL, N_WALL, N_MIN, SAFE, reduced)
                d[idxs_compr[p]] = morph_info[openmp.omp_get_thread_num(),0]
                q[idxs_compr[p]] = morph_info[openmp.omp_get_thread_num(),1]
                s[idxs_compr[p]] = morph_info[openmp.omp_get_thread_num(),2]
                major_x[idxs_compr[p]] = morph_info[openmp.omp_get_thread_num(),3]
                major_y[idxs_compr[p]] = morph_info[openmp.omp_get_thread_num(),4]
                major_z[idxs_compr[p]] = morph_info[openmp.omp_get_thread_num(),5]
                inter_x[idxs_compr[p]] = morph_info[openmp.omp_get_thread_num(),6]
                inter_y[idxs_compr[p]] = morph_info[openmp.omp_get_thread_num(),7]
                inter_z[idxs_compr[p]] = morph_info[openmp.omp_get_thread_num(),8]
                minor_x[idxs_compr[p]] = morph_info[openmp.omp_get_thread_num(),9]
                minor_y[idxs_compr[p]] = morph_info[openmp.omp_get_thread_num(),10]
                minor_z[idxs_compr[p]] = morph_info[openmp.omp_get_thread_num(),11]
                printf("Calculating overall velocity dispersion shapes. Dealing with object number %d. The number of ptcs is %d.\n", p, obj_size[p])
        
        minor = np.hstack((np.reshape(minor_x.base, (minor_x.base.shape[0],1)), np.reshape(minor_y.base, (minor_y.base.shape[0],1)), np.reshape(minor_z.base, (minor_z.base.shape[0],1))))
        inter = np.hstack((np.reshape(inter_x.base, (inter_x.base.shape[0],1)), np.reshape(inter_y.base, (inter_y.base.shape[0],1)), np.reshape(inter_z.base, (inter_z.base.shape[0],1))))
        major = np.hstack((np.reshape(major_x.base, (major_x.base.shape[0],1)), np.reshape(major_y.base, (major_y.base.shape[0],1)), np.reshape(major_z.base, (major_z.base.shape[0],1))))
        d.base[d.base==0.0] = np.nan
        s.base[s.base==0.0] = np.nan
        q.base[q.base==0.0] = np.nan
        minor[minor==0.0] = np.nan
        inter[inter==0.0] = np.nan
        major[major==0.0] = np.nan
        del xyz; del vxyz; del masses; del r200; del cat
        del major_x; del major_y; del major_z; del inter_x; del inter_y; del inter_z; del minor_x; del minor_y; del minor_z
        del morph_info; del xyz_obj; del vxyz_obj; del xyz_princ; del ellipsoid; del r_ell; del shape_tensor; del eigval; del eigvec; del m_obj
        return d.base, q.base, s.base, minor, inter, major, centers.base, m.base # Only rank = 0 content matters
    if(not hasattr(calcMorphGlobalVelDisp, "inner")):
        calcMorphGlobalVelDisp.inner = np_cache_factory(4,1)(inner)
    calcMorphGlobalVelDisp.inner(xyz.base, vxyz.base, masses.base, r200.base, cat, L_BOX, MIN_NUMBER_PTCS, M_TOL, N_WALL, N_MIN, CENTER, SAFE, reduced)
    return calcMorphGlobalVelDisp.inner(xyz.base, vxyz.base, masses.base, r200.base, cat, L_BOX, MIN_NUMBER_PTCS, M_TOL, N_WALL, N_MIN, CENTER, SAFE, reduced)

@cython.embedsignature(True)
cdef float[:,:] calcObjMorphLocal(float[:,:] morph_info, float r200, float[:] log_d, float[:,:] xyz, float[:,:] xyz_princ, float[:] masses, int[:] shell, float[:] r_ell, float[:] center, complex[::1,:] shape_tensor, double[::1] eigval, complex[::1,:] eigvec, float M_TOL, int N_WALL, int N_MIN, bint reduced, bint shell_based) nogil:
    """ Calculates the local axis ratios
    
    The local morphology is calculated for the ellipsoidal radius range [ ``r200`` x ``log_d`` [0], ``r200`` x ``log_d`` [-1]] 
    from the center of the point cloud
    
    :param morph_info: Array to be filled with morphological info. For each column, 1st entry: d,
        2nd entry: q, 3rd entry: s, 4th to 6th: normalized major axis, 7th to 9th: normalized intermediate axis,
        10th to 12th: normalized minor axis
    :type morph_info: (12,N) floats
    :param r200: R_200 radius of the parent halo
    :type r200: (N2,) float array
    :param log_d: logarithmically equally spaced ellipsoidal radius array of interest, in units of R_200 
        radius of the parent halo, e.g. np.logspace(-2,1,100)
    :type log_d: (N3,) floats
    :param xyz: positions of particles in point cloud
    :type xyz: (N1 x 3) floats
    :param xyz_princ: position arrays transformed into principal frame (varies from iteration to iteration)
    :type xyz_princ: (N1 x 3) floats, zeros
    :param masses: masses of the particles expressed in unit mass
    :type masses: (N1 x 1) floats
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
    :param M_TOL: convergence tolerance, eigenvalue fractions must differ by less than ``M_TOL``
        for iteration to stop
    :type M_TOL: float
    :param N_WALL: maximum permissible number of iterations
    :type N_WALL: float
    :param N_MIN: minimum number of particles (DM or star particle) in any iteration; 
        if undercut, shape is unclassified
    :type N_MIN: int
    :param reduced: whether or not reduced shape tensor (1/r^2 factor)
    :type reduced: boolean
    :param shell_based: whether shell-based or ellipsoid-based algorithm should be run
    :type shell_based: boolean
    :return: ``morph_info`` containing d, q, s, eigframe info in each column, for each ellipsoidal radius
    :rtype: (12,N) float array"""
    # Return if problematic
    morph_info[:,:] = 0.0
    if CythonHelpers.calcLocalSpread(xyz) == 0.0: # Too low resolution = no points in this object
        morph_info[:,:] = 0.0
        return morph_info
    if r200 == 0.0: # We are dealing with a halo which does not have any SHs, so R_200 = 0.0 according to AREPO
        morph_info[:,:] = 0.0
        return morph_info
    
    # Retrieve morphologies for all shells
    cdef int nb_shells = 0
    cdef int shell_nb
    for shell_nb in range(log_d.shape[0]):
        morph_info[0,shell_nb] = r200*log_d[shell_nb]
    nb_shells = log_d.shape[0]
    for shell_nb in range(nb_shells):
        if shell_based:
            if shell_nb == 0:
                morph_info[:,shell_nb] = runShellAlgo(morph_info[:,shell_nb], xyz, xyz_princ, masses, shell, r_ell, center, shape_tensor, eigval, eigvec, r200*log_d[shell_nb], r200*log_d[shell_nb], M_TOL, N_WALL, N_MIN, reduced)
            else:
                morph_info[:,shell_nb] = runShellAlgo(morph_info[:,shell_nb], xyz, xyz_princ, masses, shell, r_ell, center, shape_tensor, eigval, eigvec, r200*log_d[shell_nb], r200*log_d[shell_nb]-r200*log_d[shell_nb-1], M_TOL, N_WALL, N_MIN, reduced)
        else:
            morph_info[:,shell_nb] = runEllAlgo(morph_info[:,shell_nb], xyz, xyz_princ, masses, shell, r_ell, center, shape_tensor, eigval, eigvec, r200*log_d[shell_nb], M_TOL, N_WALL, N_MIN, reduced)
    
    return morph_info

@cython.embedsignature(True)
cdef float[:] calcObjMorphGlobal(float[:] morph_info, float r200, float[:,:] xyz, float[:,:] xyz_princ, float[:] masses, int[:] ellipsoid, float[:] r_ell, float[:] center, complex[::1,:] shape_tensor, double[::1] eigval, complex[::1,:] eigvec, float M_TOL, int N_WALL, int N_MIN, float SAFE, bint reduced) nogil:
    """ Calculates the global axis ratios and eigenframe of the point cloud
    
    :param morph_info: Array to be filled with morphological info. 1st entry: d,
        2nd entry: q, 3rd entry: s, 4th to 6th: normalized major axis, 7th to 9th: normalized intermediate axis,
        10th to 12th: normalized minor axis
    :type morph_info: (12,) floats
    :param r200: R_200 radius of the parent halo
    :type r200: (N2,) float array
    :param xyz: positions of particles in point cloud
    :type xyz: (N1 x 3) floats
    :param xyz_princ: position arrays transformed into principal frame (varies from iteration to iteration)
    :type xyz_princ: (N1 x 3) floats, zeros
    :param masses: masses of the particles expressed in unit mass
    :type masses: (N1 x 1) floats
    :param ellipsoid: indices of points that fall into ellipsoid (varies from iteration to iteration)
    :type ellipsoid: (N1,) ints, zeros
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
    :param M_TOL: convergence tolerance, eigenvalue fractions must differ by less than ``M_TOL``
        for iteration to stop
    :type M_TOL: float
    :param N_WALL: maximum permissible number of iterations
    :type N_WALL: float
    :param N_MIN: minimum number of particles (DM or star particle) in any iteration; 
        if undercut, shape is unclassified
    :type N_MIN: int
    :param SAFE: ellipsoidal radius will be maxdist(COM,point)+SAFE where point is any point in the point cloud. 
        The larger the better.
    :type SAFE: float
    :param reduced: whether or not reduced shape tensor (1/r^2 factor)
    :type reduced: boolean
    :return: ``morph_info`` containing d, q, s, eigframe info
    :rtype: (12,) float array"""
    # Return if problematic
    morph_info[:] = 0.0
    if CythonHelpers.calcLocalSpread(xyz) == 0.0: # Too low resolution = no points in this object
        morph_info[:] = 0.0
        return morph_info
    morph_info[0] = r200+SAFE
    
    # Retrieve morphology
    morph_info[:] = runEllAlgo(morph_info[:], xyz, xyz_princ, masses, ellipsoid, r_ell, center, shape_tensor, eigval, eigvec, morph_info[0], M_TOL, N_WALL, N_MIN, reduced)
    return morph_info

@cython.embedsignature(True)
cdef float[:,:] calcObjMorphLocalVelDisp(float[:,:] morph_info, float r200, float[:] log_d, float[:,:] xyz, float[:,:] vxyz, float[:,:] xyz_princ, float[:] masses, int[:] shell, float[:] r_ell, float[:] center, float[:] vcenter, complex[::1,:] shape_tensor, double[::1] eigval, complex[::1,:] eigvec, float M_TOL, int N_WALL, int N_MIN, bint reduced, bint shell_based) nogil:
    """ Calculates the local axis ratios of the velocity dispersion tensor 
    
    The local morphology is calculated for the ellipsoidal radius range [ ``r200`` x ``log_d`` [0], ``r200`` x ``log_d`` [-1]] 
    from the center of the point cloud
    
    :param morph_info: Array to be filled with morphological info. For each column, 1st entry: d,
        2nd entry: q, 3rd entry: s, 4th to 6th: normalized major axis, 7th to 9th: normalized intermediate axis,
        10th to 12th: normalized minor axis
    :type morph_info: (12,N) floats
    :param r200: R_200 radius of the parent halo
    :type r200: (N2,) float array
    :param log_d: logarithmically equally spaced ellipsoidal radius array of interest, in units of R_200 
        radius of the parent halo, e.g. np.logspace(-2,1,100)
    :type log_d: (N3,) floats
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
    :param M_TOL: convergence tolerance, eigenvalue fractions must differ by less than ``M_TOL``
        for iteration to stop
    :type M_TOL: float
    :param N_WALL: maximum permissible number of iterations
    :type N_WALL: float
    :param N_MIN: minimum number of particles (DM or star particle) in any iteration; 
        if undercut, shape is unclassified
    :type N_MIN: int
    :param reduced: whether or not reduced shape tensor (1/r^2 factor)
    :type reduced: boolean
    :param shell_based: whether shell-based or ellipsoid-based algorithm should be run
    :type shell_based: boolean
    :return: ``morph_info`` containing d (= ``r200``), q, s, eigframe info
    :rtype: (12,) float array"""
    # Return if problematic
    morph_info[:,:] = 0.0
    if CythonHelpers.calcLocalSpread(xyz) == 0.0: # Too low resolution = no points in this object
        morph_info[:,:] = 0.0
        return morph_info
    if r200 == 0.0: # We are dealing with a halo which does not have any SHs, so R_200 = 0.0 according to AREPO
        morph_info[:,:] = 0.0
        return morph_info
    
    # Retrieve morphologies for all shells
    cdef int nb_shells = 0
    cdef int shell_nb
    for shell_nb in range(log_d.shape[0]):
        morph_info[0,shell_nb] = r200*log_d[shell_nb]
    nb_shells = log_d.shape[0]
    for shell_nb in range(nb_shells):
        if shell_based:
            if shell_nb == 0:
                morph_info[:,shell_nb] = runShellVDispAlgo(morph_info[:,shell_nb], xyz, vxyz, xyz_princ, masses, shell, r_ell, center, vcenter, shape_tensor, eigval, eigvec, r200*log_d[shell_nb], r200*log_d[shell_nb], M_TOL, N_WALL, N_MIN, reduced)
            else:
                morph_info[:,shell_nb] = runShellVDispAlgo(morph_info[:,shell_nb], xyz, vxyz, xyz_princ, masses, shell, r_ell, center, vcenter, shape_tensor, eigval, eigvec, r200*log_d[shell_nb], r200*log_d[shell_nb]-r200*log_d[shell_nb-1], M_TOL, N_WALL, N_MIN, reduced)
        else:
            morph_info[:,shell_nb] = runEllVDispAlgo(morph_info[:,shell_nb], xyz, vxyz, xyz_princ, masses, shell, r_ell, center, vcenter, shape_tensor, eigval, eigvec, r200*log_d[shell_nb], M_TOL, N_WALL, N_MIN, reduced)
    
    return morph_info

@cython.embedsignature(True)
cdef float[:] calcObjMorphGlobalVelDisp(float[:] morph_info, float r200, float[:,:] xyz, float[:,:] vxyz, float[:,:] xyz_princ, float[:] masses, int[:] ellipsoid, float[:] r_ell, float[:] center, float[:] vcenter, complex[::1,:] shape_tensor, double[::1] eigval, complex[::1,:] eigvec, float M_TOL, int N_WALL, int N_MIN, float SAFE, bint reduced) nogil:
    """ Calculates the global axis ratios and eigenframe of the velocity dispersion tensor
    
    :param morph_info: Array to be filled with morphological info. 1st entry: d,
        2nd entry: q, 3rd entry: s, 4th to 6th: normalized major axis, 7th to 9th: normalized intermediate axis,
        10th to 12th: normalized minor axis
    :type morph_info: (12,) floats
    :param r200: R_200 radius of the parent halo
    :type r200: (N2,) float array
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
    :param M_TOL: convergence tolerance, eigenvalue fractions must differ by less than ``M_TOL``
        for iteration to stop
    :type M_TOL: float
    :param N_WALL: maximum permissible number of iterations
    :type N_WALL: float
    :param N_MIN: minimum number of particles (DM or star particle) in any iteration; 
        if undercut, shape is unclassified
    :type N_MIN: int
    :param SAFE: ellipsoidal radius will be maxdist(COM,point)+SAFE where point is any point in the point cloud. 
        The larger the better.
    :type SAFE: float
    :param reduced: whether or not reduced shape tensor (1/r^2 factor)
    :type reduced: boolean
    :return: ``morph_info`` containing d (= ``r200``), q, s, eigframe info
    :rtype: (12,) float array"""
    # Return if problematic
    morph_info[:] = 0.0
    if CythonHelpers.calcLocalSpread(xyz) == 0.0: # Too low resolution = no points in this object
        morph_info[:] = 0.0
        return morph_info
    morph_info[0] = r200+SAFE
    
    # Retrieve morphology
    morph_info[:] = runEllVDispAlgo(morph_info[:], xyz, vxyz, xyz_princ, masses, ellipsoid, r_ell, center, vcenter, shape_tensor, eigval, eigvec, morph_info[0], M_TOL, N_WALL, N_MIN, reduced)
    return morph_info
