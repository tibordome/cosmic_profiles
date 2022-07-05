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
cdef float[:] runEllShellAlgo(float[:] morph_info, float[:,:] xyz, float[:,:] xyz_princ, float[:] masses, int[:] shell, float[:] center, complex[::1,:] shape_tensor, double[::1] eigval, complex[::1,:] eigvec, float d, float delta_d, float M_TOL, int N_WALL, int N_MIN) nogil:
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
    while (err > M_TOL):
        if iteration > N_WALL:
            morph_info[:] = 0.0
            return morph_info
        if pts_in_shell < N_MIN:
            morph_info[:] = 0.0
            return morph_info
        # Get shape tensor
        shape_tensor = CythonHelpers.calcShapeTensor(xyz, shell, shape_tensor, masses, center, pts_in_shell)
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
        shell[:] = 0
        pts_in_shell = 0
        corr = 0
        if q_new*d <= delta_d or s_new*d <= delta_d:
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
cdef float[:] runEllAlgo(float[:] morph_info, float[:,:] xyz, float[:,:] xyz_princ, float[:] masses, int[:] ellipsoid, float[:] center, complex[::1,:] shape_tensor, double[::1] eigval, complex[::1,:] eigvec, float d, float M_TOL, int N_WALL, int N_MIN) nogil:
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
    while (err > M_TOL):
        if iteration > N_WALL:
            morph_info[:] = 0.0
            return morph_info
        if pts_in_ell < N_MIN:
            morph_info[:] = 0.0
            return morph_info
        # Get shape tensor
        shape_tensor = CythonHelpers.calcShapeTensor(xyz, ellipsoid, shape_tensor, masses, center, pts_in_ell)
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
cdef float[:] runEllVDispAlgo(float[:] morph_info, float[:,:] xyz, float[:,:] vxyz, float[:,:] xyz_princ, float[:] masses, int[:] ellipsoid, float[:] center, float[:] vcenter, complex[::1,:] shape_tensor, double[::1] eigval, complex[::1,:] eigvec, float d, float M_TOL, int N_WALL, int N_MIN) nogil:
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
    while (err > M_TOL):
        if iteration > N_WALL:
            morph_info[:] = 0.0
            return morph_info
        if pts_in_ell < N_MIN:
            morph_info[:] = 0.0
            return morph_info
        # Get shape tensor
        shape_tensor = CythonHelpers.calcShapeTensor(vxyz, ellipsoid, shape_tensor, masses, vcenter, pts_in_ell)
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
@np_cache_factory(3,1)
def calcMorphLocal(float[:,:] xyz, float[:] masses, float[:] r200, cat, float L_BOX, int MIN_NUMBER_PTCS, int D_LOGSTART, int D_LOGEND, int D_BINS, int M_TOL, int N_WALL, int N_MIN, str CENTER):
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
    :return: d, q, s, eigframe, centers, masses, l_succeed: list of object indices for which morphology could be determined at R200 (length: N3)
    :rtype: (N3, ``D_BINS`` + 1) floats (for d, q, s, eigframe (x3)), (N3, 3) floats (for centers), (N3,) floats (for masses), N3-list of ints for l_succeed
    """
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

    cdef float[:] m = np.zeros((nb_objs,), dtype = np.float32)
    cdef float[:,:] d = np.zeros((nb_objs, D_BINS+1), dtype = np.float32)
    cdef float[:,:] q = np.zeros((nb_objs, D_BINS+1), dtype = np.float32)
    cdef float[:,:] s = np.zeros((nb_objs, D_BINS+1), dtype = np.float32)
    cdef float[:,:] major_x = np.zeros((nb_objs, D_BINS+1), dtype = np.float32)
    cdef float[:,:] major_y = np.zeros((nb_objs, D_BINS+1), dtype = np.float32)
    cdef float[:,:] major_z = np.zeros((nb_objs, D_BINS+1), dtype = np.float32)
    cdef float[:,:] inter_x = np.zeros((nb_objs, D_BINS+1), dtype = np.float32)
    cdef float[:,:] inter_y = np.zeros((nb_objs, D_BINS+1), dtype = np.float32)
    cdef float[:,:] inter_z = np.zeros((nb_objs, D_BINS+1), dtype = np.float32)
    cdef float[:,:] minor_x = np.zeros((nb_objs, D_BINS+1), dtype = np.float32)
    cdef float[:,:] minor_y = np.zeros((nb_objs, D_BINS+1), dtype = np.float32)
    cdef float[:,:] minor_z = np.zeros((nb_objs, D_BINS+1), dtype = np.float32)
    cdef float[:,:,:] morph_info = np.zeros((openmp.omp_get_max_threads(), 12, D_BINS+1), dtype = np.float32)
    cdef float[:,:,:] xyz_obj = np.zeros((openmp.omp_get_max_threads(), cat_arr.shape[1],3), dtype = np.float32)
    cdef float[:,:,:] xyz_princ = np.zeros((openmp.omp_get_max_threads(), cat_arr.shape[1],3), dtype = np.float32)
    cdef int[:,:] shell = np.zeros((openmp.omp_get_max_threads(), cat_arr.shape[1]), dtype = np.int32)
    cdef complex[::1,:,:] shape_tensor = np.zeros((3, 3, openmp.omp_get_max_threads()), dtype = np.complex128, order='F')
    cdef double[::1,:] eigval = np.zeros((3, openmp.omp_get_max_threads()), dtype=np.float64, order='F')
    cdef complex[::1,:,:] eigvec = np.zeros((3,3, openmp.omp_get_max_threads()), dtype=np.complex128, order='F')
    cdef float[:,:] m_obj = np.zeros((openmp.omp_get_max_threads(), cat_arr.shape[1]), dtype = np.float32)
    cdef float[:] log_d = np.logspace(D_LOGSTART,D_LOGEND,D_BINS+1, dtype = np.float32)
    cdef bint success
    cdef int n
    cdef int r
    cdef float[:,:] centers = np.zeros((nb_objs,3), dtype = np.float32)
    for p in range(nb_objs): # Calculate centers of objects
        if obj_pass[p] == 1:
            xyz_ = respectPBCNoRef(xyz.base[cat_arr.base[idxs_compr[p],:obj_size[p]]], L_BOX)
            if CENTER == 'mode':
                centers.base[p] = calcMode(xyz_, masses.base[cat_arr.base[idxs_compr[p],:obj_size[p]]], max((max(xyz_[:,0])-min(xyz_[:,0]), max(xyz_[:,1])-min(xyz_[:,1]), max(xyz_[:,2])-min(xyz_[:,2]))))
            else:
                centers.base[p] = calcCoM(xyz_, masses.base[cat_arr.base[idxs_compr[p],:obj_size[p]]])
    for p in prange(nb_objs, schedule = 'dynamic', nogil = True):
        if obj_pass[p] == 1:
            for n in range(obj_size[p]):
                xyz_obj[openmp.omp_get_thread_num(),n,0] = xyz[cat_arr[idxs_compr[p],n],0]
                xyz_obj[openmp.omp_get_thread_num(),n,1] = xyz[cat_arr[idxs_compr[p],n],1]
                xyz_obj[openmp.omp_get_thread_num(),n,2] = xyz[cat_arr[idxs_compr[p],n],2]
                m_obj[openmp.omp_get_thread_num(),n] = masses[cat_arr[idxs_compr[p],n]]
                m[p] = m[p] + masses[cat_arr[idxs_compr[p],n]]
            xyz_obj[openmp.omp_get_thread_num(),:obj_size[p]] = CythonHelpers.respectPBCNoRef(xyz_obj[openmp.omp_get_thread_num(),:obj_size[p]], L_BOX)
            morph_info[openmp.omp_get_thread_num(),:,:] = calcObjMorphLocal(morph_info[openmp.omp_get_thread_num(),:,:], r200[p], log_d, xyz_obj[openmp.omp_get_thread_num(),:obj_size[p]], xyz_princ[openmp.omp_get_thread_num(),:obj_size[p]], m_obj[openmp.omp_get_thread_num(),:obj_size[p]], shell[openmp.omp_get_thread_num()], centers[p], shape_tensor[:,:,openmp.omp_get_thread_num()], eigval[:,openmp.omp_get_thread_num()], eigvec[:,:,openmp.omp_get_thread_num()], M_TOL, N_WALL, N_MIN)
            d[p] = morph_info[openmp.omp_get_thread_num(),0]
            q[p] = morph_info[openmp.omp_get_thread_num(),1]
            s[p] = morph_info[openmp.omp_get_thread_num(),2]
            major_x[p] = morph_info[openmp.omp_get_thread_num(),3]
            major_y[p] = morph_info[openmp.omp_get_thread_num(),4]
            major_z[p] = morph_info[openmp.omp_get_thread_num(),5]
            inter_x[p] = morph_info[openmp.omp_get_thread_num(),6]
            inter_y[p] = morph_info[openmp.omp_get_thread_num(),7]
            inter_z[p] = morph_info[openmp.omp_get_thread_num(),8]
            minor_x[p] = morph_info[openmp.omp_get_thread_num(),9]
            minor_y[p] = morph_info[openmp.omp_get_thread_num(),10]
            minor_z[p] = morph_info[openmp.omp_get_thread_num(),11]
            if obj_size[p] != 0:
                success = False
                for n in range(morph_info.shape[1]):
                    for r in range(morph_info.shape[2]):
                        if morph_info[openmp.omp_get_thread_num(),n,r] != 0.0:
                            success = True
                            break
                printf("Purpose: local. Dealing with object number %d. The number of ptcs is %d. Shape determination at R200 successful: %d\n", p, obj_size[p], success)
        
    l_succeed = []
    for p in range(nb_objs):
        if not (d.base[p] == d.base[p,0]).all():
            l_succeed += [p]
    succeed = np.array(l_succeed)
    if succeed.shape[0] != 0:
        minor = np.transpose(np.stack((minor_x.base[succeed],minor_y.base[succeed],minor_z.base[succeed])),(1,2,0))
        inter = np.transpose(np.stack((inter_x.base[succeed],inter_y.base[succeed],inter_z.base[succeed])),(1,2,0))
        major = np.transpose(np.stack((major_x.base[succeed],major_y.base[succeed],major_z.base[succeed])),(1,2,0))
        d.base[succeed][d.base[succeed]==0.0] = np.nan
        s.base[succeed][s.base[succeed]==0.0] = np.nan
        q.base[succeed][q.base[succeed]==0.0] = np.nan
        minor[minor==0.0] = np.nan
        inter[inter==0.0] = np.nan
        major[major==0.0] = np.nan
        centers.base[succeed][centers.base[succeed]==0.0] = np.nan
        m.base[succeed][m.base[succeed]==0.0] = np.nan
        return d.base[succeed], q.base[succeed], s.base[succeed], minor, inter, major, centers.base[succeed], m.base[succeed], l_succeed # Only rank = 0 content matters
    else:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), l_succeed

@cython.embedsignature(True)
@np_cache_factory(3,1)
def calcMorphGlobal(float[:,:] xyz, float[:] masses, float[:] r200, cat, float L_BOX, int MIN_NUMBER_PTCS, int M_TOL, int N_WALL, int N_MIN, str CENTER, float SAFE):
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
    :return: d, q, s, eigframe, centers, masses
    :rtype: (N3,) floats (for d, q, s, eigframe (x3)), (N3, 3) floats (for centers), (N3,) floats (for masses)
    """
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

    cdef float[:] m = np.zeros((nb_objs,), dtype = np.float32)
    cdef float[:] d = np.zeros((nb_objs,), dtype = np.float32)
    cdef float[:] q = np.zeros((nb_objs,), dtype = np.float32)
    cdef float[:] s = np.zeros((nb_objs,), dtype = np.float32)
    cdef float[:] major_x = np.zeros((nb_objs,), dtype = np.float32)
    cdef float[:] major_y = np.zeros((nb_objs,), dtype = np.float32)
    cdef float[:] major_z = np.zeros((nb_objs,), dtype = np.float32)
    cdef float[:] inter_x = np.zeros((nb_objs,), dtype = np.float32)
    cdef float[:] inter_y = np.zeros((nb_objs,), dtype = np.float32)
    cdef float[:] inter_z = np.zeros((nb_objs,), dtype = np.float32)
    cdef float[:] minor_x = np.zeros((nb_objs,), dtype = np.float32)
    cdef float[:] minor_y = np.zeros((nb_objs,), dtype = np.float32)
    cdef float[:] minor_z = np.zeros((nb_objs,), dtype = np.float32)
    cdef float[:,:] centers = np.zeros((nb_objs,3), dtype = np.float32)
    cdef float[:,:] morph_info = np.zeros((openmp.omp_get_max_threads(), 12), dtype = np.float32)
    cdef float[:,:,:] xyz_obj = np.zeros((openmp.omp_get_max_threads(), cat_arr.shape[1],3), dtype = np.float32)
    cdef float[:,:,:] xyz_princ = np.zeros((openmp.omp_get_max_threads(), cat_arr.shape[1],3), dtype = np.float32)
    cdef int[:,:] ellipsoid = np.zeros((openmp.omp_get_max_threads(), cat_arr.shape[1]), dtype = np.int32)
    cdef complex[::1,:,:] shape_tensor = np.zeros((3, 3, openmp.omp_get_max_threads()), dtype = np.complex128, order='F')
    cdef double[::1,:] eigval = np.zeros((3, openmp.omp_get_max_threads()), dtype=np.float64, order='F')
    cdef complex[::1,:,:] eigvec = np.zeros((3,3, openmp.omp_get_max_threads()), dtype=np.complex128, order='F')
    cdef float[:,:] m_obj = np.zeros((openmp.omp_get_max_threads(), cat_arr.shape[1]), dtype = np.float32)
    cdef bint success
    cdef int n
    for p in range(nb_objs): # Calculate centers of objects
        if obj_pass[p] == 1:
            xyz_ = respectPBCNoRef(xyz.base[cat_arr.base[idxs_compr[p],:obj_size[p]]], L_BOX)
            if CENTER == 'mode':
                centers.base[p] = calcMode(xyz_, masses.base[cat_arr.base[idxs_compr[p],:obj_size[p]]], max((max(xyz_[:,0])-min(xyz_[:,0]), max(xyz_[:,1])-min(xyz_[:,1]), max(xyz_[:,2])-min(xyz_[:,2]))))
            else:
                centers.base[p] = calcCoM(xyz_, masses.base[cat_arr.base[idxs_compr[p],:obj_size[p]]])
    for p in prange(nb_objs, schedule = 'dynamic', nogil = True):
        if obj_pass[p] == 1:
            for n in range(obj_size[p]):
                xyz_obj[openmp.omp_get_thread_num(),n] = xyz[cat_arr[idxs_compr[p],n]]
                m_obj[openmp.omp_get_thread_num(),n] = masses[cat_arr[idxs_compr[p],n]]
                m[p] = m[p] + masses[cat_arr[idxs_compr[p],n]]
            xyz_obj[openmp.omp_get_thread_num(),:obj_size[p]] = CythonHelpers.respectPBCNoRef(xyz_obj[openmp.omp_get_thread_num(),:obj_size[p]], L_BOX)
            morph_info[openmp.omp_get_thread_num(),:] = calcObjMorphGlobal(morph_info[openmp.omp_get_thread_num(),:], r200[p], xyz_obj[openmp.omp_get_thread_num(),:obj_size[p]], xyz_princ[openmp.omp_get_thread_num(),:obj_size[p]], m_obj[openmp.omp_get_thread_num(),:obj_size[p]], ellipsoid[openmp.omp_get_thread_num()], centers[p], shape_tensor[:,:,openmp.omp_get_thread_num()], eigval[:,openmp.omp_get_thread_num()], eigvec[:,:,openmp.omp_get_thread_num()], M_TOL, N_WALL, N_MIN, SAFE)
            d[p] = morph_info[openmp.omp_get_thread_num(),0]
            q[p] = morph_info[openmp.omp_get_thread_num(),1]
            s[p] = morph_info[openmp.omp_get_thread_num(),2]
            major_x[p] = morph_info[openmp.omp_get_thread_num(),3]
            major_y[p] = morph_info[openmp.omp_get_thread_num(),4]
            major_z[p] = morph_info[openmp.omp_get_thread_num(),5]
            inter_x[p] = morph_info[openmp.omp_get_thread_num(),6]
            inter_y[p] = morph_info[openmp.omp_get_thread_num(),7]
            inter_z[p] = morph_info[openmp.omp_get_thread_num(),8]
            minor_x[p] = morph_info[openmp.omp_get_thread_num(),9]
            minor_y[p] = morph_info[openmp.omp_get_thread_num(),10]
            minor_z[p] = morph_info[openmp.omp_get_thread_num(),11]
            if obj_size[p] != 0:
                success = False
                for n in range(12):
                    if morph_info[openmp.omp_get_thread_num(),n] != 0.0:
                        success = True
                        break
                printf("Purpose: global. Dealing with object number %d. The number of ptcs is %d. Global shape determination successful: %d\n", p, obj_size[p], success)
        
    l_succeed = []
    for p in range(nb_objs):
        if obj_pass[p] == 1:
            l_succeed += [p]
    succeed = np.array(l_succeed)
    if succeed.shape[0] == 0:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    minor = np.hstack((np.reshape(minor_x.base[succeed], (minor_x.base[succeed].shape[0],1)), np.reshape(minor_y.base[succeed], (minor_y.base[succeed].shape[0],1)), np.reshape(minor_z.base[succeed], (minor_z.base[succeed].shape[0],1))))
    inter = np.hstack((np.reshape(inter_x.base[succeed], (inter_x.base[succeed].shape[0],1)), np.reshape(inter_y.base[succeed], (inter_y.base[succeed].shape[0],1)), np.reshape(inter_z.base[succeed], (inter_z.base[succeed].shape[0],1))))
    major = np.hstack((np.reshape(major_x.base[succeed], (major_x.base[succeed].shape[0],1)), np.reshape(major_y.base[succeed], (major_y.base[succeed].shape[0],1)), np.reshape(major_z.base[succeed], (major_z.base[succeed].shape[0],1))))
    d.base[succeed][d.base[succeed]==0.0] = np.nan
    s.base[succeed][s.base[succeed]==0.0] = np.nan
    q.base[succeed][q.base[succeed]==0.0] = np.nan
    minor[minor==0.0] = np.nan
    inter[inter==0.0] = np.nan
    major[major==0.0] = np.nan
    centers.base[succeed][centers.base[succeed]==0.0] = np.nan
    m.base[succeed][m.base[succeed]==0.0] = np.nan
    return d.base[succeed], q.base[succeed], s.base[succeed], minor, inter, major, centers.base[succeed], m.base[succeed] # Only rank = 0 content matters

@cython.embedsignature(True)
@np_cache_factory(4,1)
def calcMorphLocalVelDisp(float[:,:] xyz, float[:,:] vxyz, float[:] masses, float[:] r200, cat, float L_BOX, int MIN_NUMBER_PTCS, int D_LOGSTART, int D_LOGEND, int D_BINS, int M_TOL, int N_WALL, int N_MIN, str CENTER):
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
    :return: d, q, s, eigframe, centers, masses, l_succeed: list of object indices for which morphology could be determined at R200 (length: N3)
    :rtype: (N3, ``D_BINS`` + 1) floats (for d, q, s, eigframe (x3)), (N3, 3) floats (for centers), (N3,) floats (for masses), N3-list of ints for l_succeed
    """
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

    cdef float[:] m = np.zeros((nb_objs,), dtype = np.float32)
    cdef float[:,:] d = np.zeros((nb_objs, D_BINS+1), dtype = np.float32)
    cdef float[:,:] q = np.zeros((nb_objs, D_BINS+1), dtype = np.float32)
    cdef float[:,:] s = np.zeros((nb_objs, D_BINS+1), dtype = np.float32)
    cdef float[:,:] major_x = np.zeros((nb_objs, D_BINS+1), dtype = np.float32)
    cdef float[:,:] major_y = np.zeros((nb_objs, D_BINS+1), dtype = np.float32)
    cdef float[:,:] major_z = np.zeros((nb_objs, D_BINS+1), dtype = np.float32)
    cdef float[:,:] inter_x = np.zeros((nb_objs, D_BINS+1), dtype = np.float32)
    cdef float[:,:] inter_y = np.zeros((nb_objs, D_BINS+1), dtype = np.float32)
    cdef float[:,:] inter_z = np.zeros((nb_objs, D_BINS+1), dtype = np.float32)
    cdef float[:,:] minor_x = np.zeros((nb_objs, D_BINS+1), dtype = np.float32)
    cdef float[:,:] minor_y = np.zeros((nb_objs, D_BINS+1), dtype = np.float32)
    cdef float[:,:] minor_z = np.zeros((nb_objs, D_BINS+1), dtype = np.float32)
    cdef float[:,:] centers = np.zeros((nb_objs,3), dtype = np.float32)
    cdef float[:,:] vcenters = np.zeros((nb_objs,3), dtype = np.float32)
    cdef float[:,:,:] morph_info = np.zeros((openmp.omp_get_max_threads(), 12, D_BINS+1), dtype = np.float32)
    cdef float[:,:,:] xyz_obj = np.zeros((openmp.omp_get_max_threads(), cat_arr.shape[1],3), dtype = np.float32)
    cdef float[:,:,:] vxyz_obj = np.zeros((openmp.omp_get_max_threads(), cat_arr.shape[1],3), dtype = np.float32)
    cdef float[:,:,:] xyz_princ = np.zeros((openmp.omp_get_max_threads(), cat_arr.shape[1],3), dtype = np.float32)
    cdef int[:,:] shell = np.zeros((openmp.omp_get_max_threads(), cat_arr.shape[1]), dtype = np.int32)
    cdef complex[::1,:,:] shape_tensor = np.zeros((3, 3, openmp.omp_get_max_threads()), dtype = np.complex128, order='F')
    cdef double[::1,:] eigval = np.zeros((3, openmp.omp_get_max_threads()), dtype=np.float64, order='F')
    cdef complex[::1,:,:] eigvec = np.zeros((3,3, openmp.omp_get_max_threads()), dtype=np.complex128, order='F')
    cdef float[:,:] m_obj = np.zeros((openmp.omp_get_max_threads(), cat_arr.shape[1]), dtype = np.float32)
    cdef float[:] log_d = np.logspace(D_LOGSTART,D_LOGEND,D_BINS+1, dtype = np.float32)
    cdef bint success
    cdef int n
    cdef int r
    for p in range(nb_objs): # Calculate centers of objects
        if obj_pass[p] == 1:
            xyz_ = respectPBCNoRef(xyz.base[cat_arr.base[idxs_compr[p],:obj_size[p]]], L_BOX)
            if CENTER == 'mode':
                centers.base[p] = calcMode(xyz_, masses.base[cat_arr.base[idxs_compr[p],:obj_size[p]]], max((max(xyz_[:,0])-min(xyz_[:,0]), max(xyz_[:,1])-min(xyz_[:,1]), max(xyz_[:,2])-min(xyz_[:,2]))))
            else:
                centers.base[p] = calcCoM(xyz_, masses.base[cat_arr.base[idxs_compr[p],:obj_size[p]]])
    for p in prange(nb_objs, schedule = 'dynamic', nogil = True):
        if obj_pass[p] == 1:
            for n in range(obj_size[p]):
                xyz_obj[openmp.omp_get_thread_num(),n,0] = xyz[cat_arr[idxs_compr[p],n],0]
                xyz_obj[openmp.omp_get_thread_num(),n,1] = xyz[cat_arr[idxs_compr[p],n],1]
                xyz_obj[openmp.omp_get_thread_num(),n,2] = xyz[cat_arr[idxs_compr[p],n],2]
                m_obj[openmp.omp_get_thread_num(),n] = masses[cat_arr[idxs_compr[p],n]]
                m[p] = m[p] + masses[cat_arr[idxs_compr[p],n]]
            xyz_obj[openmp.omp_get_thread_num(),:obj_size[p]] = CythonHelpers.respectPBCNoRef(xyz_obj[openmp.omp_get_thread_num(),:obj_size[p]], L_BOX)
            vcenters[p] = CythonHelpers.calcCoM(vxyz_obj[openmp.omp_get_thread_num(),:obj_size[p]], m_obj[openmp.omp_get_thread_num(),:obj_size[p]], vcenters[p])
            morph_info[openmp.omp_get_thread_num(),:,:] = calcObjMorphLocalVelDisp(morph_info[openmp.omp_get_thread_num(),:,:], r200[p], log_d, xyz_obj[openmp.omp_get_thread_num(),:obj_size[p]], vxyz_obj[openmp.omp_get_thread_num(),:obj_size[p]], xyz_princ[openmp.omp_get_thread_num(),:obj_size[p]], m_obj[openmp.omp_get_thread_num(),:obj_size[p]], shell[openmp.omp_get_thread_num()], centers[p], vcenters[p], shape_tensor[:,:,openmp.omp_get_thread_num()], eigval[:,openmp.omp_get_thread_num()], eigvec[:,:,openmp.omp_get_thread_num()], M_TOL, N_WALL, N_MIN)
            d[p] = morph_info[openmp.omp_get_thread_num(),0]
            q[p] = morph_info[openmp.omp_get_thread_num(),1]
            s[p] = morph_info[openmp.omp_get_thread_num(),2]
            major_x[p] = morph_info[openmp.omp_get_thread_num(),3]
            major_y[p] = morph_info[openmp.omp_get_thread_num(),4]
            major_z[p] = morph_info[openmp.omp_get_thread_num(),5]
            inter_x[p] = morph_info[openmp.omp_get_thread_num(),6]
            inter_y[p] = morph_info[openmp.omp_get_thread_num(),7]
            inter_z[p] = morph_info[openmp.omp_get_thread_num(),8]
            minor_x[p] = morph_info[openmp.omp_get_thread_num(),9]
            minor_y[p] = morph_info[openmp.omp_get_thread_num(),10]
            minor_z[p] = morph_info[openmp.omp_get_thread_num(),11]
            if obj_size[p] != 0:
                success = False
                for n in range(morph_info.shape[1]):
                    for r in range(morph_info.shape[2]):
                        if morph_info[openmp.omp_get_thread_num(),n,r] != 0.0:
                            success = True
                            break
                printf("Purpose: local. Dealing with object number %d. The number of ptcs is %d. Shape determination at R200 successful: %d\n", p, obj_size[p], success)
    
    l_succeed = []
    for p in range(nb_objs):
        if not (d.base[p] == d.base[p,0]).all():
            l_succeed += [p]
    succeed = np.array(l_succeed)
    if succeed.shape[0] != 0:
        minor = np.transpose(np.stack((minor_x.base[succeed],minor_y.base[succeed],minor_z.base[succeed])),(1,2,0))
        inter = np.transpose(np.stack((inter_x.base[succeed],inter_y.base[succeed],inter_z.base[succeed])),(1,2,0))
        major = np.transpose(np.stack((major_x.base[succeed],major_y.base[succeed],major_z.base[succeed])),(1,2,0))
        d.base[succeed][d.base[succeed]==0.0] = np.nan
        s.base[succeed][s.base[succeed]==0.0] = np.nan
        q.base[succeed][q.base[succeed]==0.0] = np.nan
        minor[minor==0.0] = np.nan
        inter[inter==0.0] = np.nan
        major[major==0.0] = np.nan
        centers.base[succeed][centers.base[succeed]==0.0] = np.nan
        m.base[succeed][m.base[succeed]==0.0] = np.nan
        return d.base[succeed], q.base[succeed], s.base[succeed], minor, inter, major, centers.base[succeed], m.base[succeed], l_succeed # Only rank = 0 content matters
    else:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), l_succeed # Only rank = 0 content matters

@cython.embedsignature(True)
@np_cache_factory(4,1)
def calcMorphGlobalVelDisp(float[:,:] xyz, float[:,:] vxyz, float[:] masses, float[:] r200, cat, float L_BOX, int MIN_NUMBER_PTCS, int M_TOL, int N_WALL, int N_MIN, str CENTER, float SAFE):
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
    :return: d, q, s, eigframe, centers, masses
    :rtype: (N3, ``D_BINS`` + 1) floats (for d, q, s, eigframe (x3)), (N3, 3) floats (for centers), (N3,) floats (for masses)
    """
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

    cdef float[:] m = np.zeros((nb_objs,), dtype = np.float32)
    cdef float[:] d = np.zeros((nb_objs,), dtype = np.float32)
    cdef float[:] q = np.zeros((nb_objs,), dtype = np.float32)
    cdef float[:] s = np.zeros((nb_objs,), dtype = np.float32)
    cdef float[:] major_x = np.zeros((nb_objs,), dtype = np.float32)
    cdef float[:] major_y = np.zeros((nb_objs,), dtype = np.float32)
    cdef float[:] major_z = np.zeros((nb_objs,), dtype = np.float32)
    cdef float[:] inter_x = np.zeros((nb_objs,), dtype = np.float32)
    cdef float[:] inter_y = np.zeros((nb_objs,), dtype = np.float32)
    cdef float[:] inter_z = np.zeros((nb_objs,), dtype = np.float32)
    cdef float[:] minor_x = np.zeros((nb_objs,), dtype = np.float32)
    cdef float[:] minor_y = np.zeros((nb_objs,), dtype = np.float32)
    cdef float[:] minor_z = np.zeros((nb_objs,), dtype = np.float32)
    cdef float[:,:] centers = np.zeros((nb_objs,3), dtype = np.float32)
    cdef float[:,:] vcenters = np.zeros((nb_objs,3), dtype = np.float32)
    cdef float[:,:] morph_info = np.zeros((openmp.omp_get_max_threads(), 12), dtype = np.float32)
    cdef float[:,:,:] xyz_obj = np.zeros((openmp.omp_get_max_threads(), cat_arr.shape[1],3), dtype = np.float32)
    cdef float[:,:,:] vxyz_obj = np.zeros((openmp.omp_get_max_threads(), cat_arr.shape[1],3), dtype = np.float32)
    cdef float[:,:,:] xyz_princ = np.zeros((openmp.omp_get_max_threads(), cat_arr.shape[1],3), dtype = np.float32)
    cdef int[:,:] ellipsoid = np.zeros((openmp.omp_get_max_threads(), cat_arr.shape[1]), dtype = np.int32)
    cdef complex[::1,:,:] shape_tensor = np.zeros((3, 3, openmp.omp_get_max_threads()), dtype = np.complex128, order='F')
    cdef double[::1,:] eigval = np.zeros((3, openmp.omp_get_max_threads()), dtype=np.float64, order='F')
    cdef complex[::1,:,:] eigvec = np.zeros((3,3, openmp.omp_get_max_threads()), dtype=np.complex128, order='F')
    cdef float[:,:] m_obj = np.zeros((openmp.omp_get_max_threads(), cat_arr.shape[1]), dtype = np.float32)
    cdef bint success
    cdef int n
    for p in range(nb_objs): # Calculate centers of objects
        if obj_pass[p] == 1:
            xyz_ = respectPBCNoRef(xyz.base[cat_arr.base[idxs_compr[p],:obj_size[p]]], L_BOX)
            if CENTER == 'mode':
                centers.base[p] = calcMode(xyz_, masses.base[cat_arr.base[idxs_compr[p],:obj_size[p]]], max((max(xyz_[:,0])-min(xyz_[:,0]), max(xyz_[:,1])-min(xyz_[:,1]), max(xyz_[:,2])-min(xyz_[:,2]))))
            else:
                centers.base[p] = calcCoM(xyz_, masses.base[cat_arr.base[idxs_compr[p],:obj_size[p]]])
    for p in prange(nb_objs, schedule = 'dynamic', nogil = True):
        if obj_pass[p] == 1:
            for n in range(obj_size[p]):
                xyz_obj[openmp.omp_get_thread_num(),n] = xyz[cat_arr[idxs_compr[p],n]]
                vxyz_obj[openmp.omp_get_thread_num(),n] = vxyz[cat_arr[idxs_compr[p],n]]
                m_obj[openmp.omp_get_thread_num(),n] = masses[cat_arr[idxs_compr[p],n]]
                m[p] = m[p] + masses[cat_arr[idxs_compr[p],n]]
            xyz_obj[openmp.omp_get_thread_num(),:obj_size[p]] = CythonHelpers.respectPBCNoRef(xyz_obj[openmp.omp_get_thread_num(),:obj_size[p]], L_BOX)
            vcenters[p] = CythonHelpers.calcCoM(vxyz_obj[openmp.omp_get_thread_num(),:obj_size[p]], m_obj[openmp.omp_get_thread_num(),:obj_size[p]], vcenters[p])
            morph_info[openmp.omp_get_thread_num(),:] = calcObjMorphGlobalVelDisp(morph_info[openmp.omp_get_thread_num(),:], r200[p], xyz_obj[openmp.omp_get_thread_num(),:obj_size[p]], vxyz_obj[openmp.omp_get_thread_num(),:obj_size[p]], xyz_princ[openmp.omp_get_thread_num(),:obj_size[p]], m_obj[openmp.omp_get_thread_num(),:obj_size[p]], ellipsoid[openmp.omp_get_thread_num()], centers[p], vcenters[p], shape_tensor[:,:,openmp.omp_get_thread_num()], eigval[:,openmp.omp_get_thread_num()], eigvec[:,:,openmp.omp_get_thread_num()], M_TOL, N_WALL, N_MIN, SAFE)
            d[p] = morph_info[openmp.omp_get_thread_num(),0]
            q[p] = morph_info[openmp.omp_get_thread_num(),1]
            s[p] = morph_info[openmp.omp_get_thread_num(),2]
            major_x[p] = morph_info[openmp.omp_get_thread_num(),3]
            major_y[p] = morph_info[openmp.omp_get_thread_num(),4]
            major_z[p] = morph_info[openmp.omp_get_thread_num(),5]
            inter_x[p] = morph_info[openmp.omp_get_thread_num(),6]
            inter_y[p] = morph_info[openmp.omp_get_thread_num(),7]
            inter_z[p] = morph_info[openmp.omp_get_thread_num(),8]
            minor_x[p] = morph_info[openmp.omp_get_thread_num(),9]
            minor_y[p] = morph_info[openmp.omp_get_thread_num(),10]
            minor_z[p] = morph_info[openmp.omp_get_thread_num(),11]
            if obj_size[p] != 0:
                success = False
                for n in range(12):
                    if morph_info[openmp.omp_get_thread_num(),n] != 0.0:
                        success = True
                        break
                printf("Purpose: vdisp. Dealing with object number %d. The number of ptcs is %d. VelDisp shape determination at R200 successful: %d\n", p, obj_size[p], success)
        
    l_succeed = []
    for p in range(nb_objs):
        if obj_pass[p] == 1:
            l_succeed += [p]
    succeed = np.array(l_succeed)
    if succeed.shape[0] == 0:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    minor = np.hstack((np.reshape(minor_x.base[succeed], (minor_x.base[succeed].shape[0],1)), np.reshape(minor_y.base[succeed], (minor_y.base[succeed].shape[0],1)), np.reshape(minor_z.base[succeed], (minor_z.base[succeed].shape[0],1))))
    inter = np.hstack((np.reshape(inter_x.base[succeed], (inter_x.base[succeed].shape[0],1)), np.reshape(inter_y.base[succeed], (inter_y.base[succeed].shape[0],1)), np.reshape(inter_z.base[succeed], (inter_z.base[succeed].shape[0],1))))
    major = np.hstack((np.reshape(major_x.base[succeed], (major_x.base[succeed].shape[0],1)), np.reshape(major_y.base[succeed], (major_y.base[succeed].shape[0],1)), np.reshape(major_z.base[succeed], (major_z.base[succeed].shape[0],1))))
    d.base[succeed][d.base[succeed]==0.0] = np.nan
    s.base[succeed][s.base[succeed]==0.0] = np.nan
    q.base[succeed][q.base[succeed]==0.0] = np.nan
    minor[minor==0.0] = np.nan
    inter[inter==0.0] = np.nan
    major[major==0.0] = np.nan
    centers.base[succeed][centers.base[succeed]==0.0] = np.nan
    m.base[succeed][m.base[succeed]==0.0] = np.nan
    return d.base[succeed], q.base[succeed], s.base[succeed], minor, inter, major, centers.base[succeed], m.base[succeed] # Only rank = 0 content matters

@cython.embedsignature(True)
cdef float[:,:] calcObjMorphLocal(float[:,:] morph_info, float r200, float[:] log_d, float[:,:] xyz, float[:,:] xyz_princ, float[:] masses, int[:] shell, float[:] center, complex[::1,:] shape_tensor, double[::1] eigval, complex[::1,:] eigvec, float M_TOL, int N_WALL, int N_MIN) nogil:
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
    cdef int i
    for i in range(log_d.shape[0]):
        morph_info[0,i] = r200*log_d[i]
    nb_shells = log_d.shape[0]
    for i in range(nb_shells):
        morph_info[:,i] = runEllAlgo(morph_info[:,i], xyz, xyz_princ, masses, shell, center, shape_tensor, eigval, eigvec, morph_info[0,i], M_TOL, N_WALL, N_MIN)
    
    # Discard if r200 ellipsoid did not converge
    closest_idx = 0
    for i in range(nb_shells):
        if (r200*log_d[i] - r200)**2 < (r200*log_d[closest_idx] - r200)**2:
            closest_idx = i
    if morph_info[1,closest_idx] == 0: # Return empty morph_info if R200 ellipsoid did not converge
        morph_info[:,:] = 0.0
    return morph_info

@cython.embedsignature(True)
cdef float[:] calcObjMorphGlobal(float[:] morph_info, float r200, float[:,:] xyz, float[:,:] xyz_princ, float[:] masses, int[:] ellipsoid, float[:] center, complex[::1,:] shape_tensor, double[::1] eigval, complex[::1,:] eigvec, float M_TOL, int N_WALL, int N_MIN, float SAFE) nogil:
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
    :return: ``morph_info`` containing d, q, s, eigframe info
    :rtype: (12,) float array"""
    # Return if problematic
    morph_info[:] = 0.0
    if CythonHelpers.calcLocalSpread(xyz) == 0.0: # Too low resolution = no points in this object
        morph_info[:] = 0.0
        return morph_info
    morph_info[0] = r200+SAFE
    
    # Retrieve morphology
    morph_info[:] = runEllAlgo(morph_info[:], xyz, xyz_princ, masses, ellipsoid, center, shape_tensor, eigval, eigvec, morph_info[0], M_TOL, N_WALL, N_MIN)
    return morph_info

@cython.embedsignature(True)
cdef float[:,:] calcObjMorphLocalVelDisp(float[:,:] morph_info, float r200, float[:] log_d, float[:,:] xyz, float[:,:] vxyz, float[:,:] xyz_princ, float[:] masses, int[:] shell, float[:] center, float[:] vcenter, complex[::1,:] shape_tensor, double[::1] eigval, complex[::1,:] eigvec, float M_TOL, int N_WALL, int N_MIN) nogil:
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
    cdef int i
    for i in range(log_d.shape[0]):
        morph_info[0,i] = r200*log_d[i]
    nb_shells = log_d.shape[0]
    for i in range(nb_shells):
        morph_info[:,i] = runEllVDispAlgo(morph_info[:,i], xyz, vxyz, xyz_princ, masses, shell, center, vcenter, shape_tensor, eigval, eigvec, morph_info[0,i], M_TOL, N_WALL, N_MIN)
    
    # Discard if r200 ellipsoid did not converge
    closest_idx = 0
    for i in range(nb_shells):
        if (r200*log_d[i] - r200)**2 < (r200*log_d[closest_idx] - r200)**2:
            closest_idx = i
    if morph_info[1,closest_idx] == 0: # Return empty morph_info if R200 ellipsoid did not converge
        morph_info[:,:] = 0.0
    return morph_info

@cython.embedsignature(True)
cdef float[:] calcObjMorphGlobalVelDisp(float[:] morph_info, float r200, float[:,:] xyz, float[:,:] vxyz, float[:,:] xyz_princ, float[:] masses, int[:] ellipsoid, float[:] center, float[:] vcenter, complex[::1,:] shape_tensor, double[::1] eigval, complex[::1,:] eigvec, float M_TOL, int N_WALL, int N_MIN, float SAFE) nogil:
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
    :return: ``morph_info`` containing d (= ``r200``), q, s, eigframe info
    :rtype: (12,) float array"""
    # Return if problematic
    morph_info[:] = 0.0
    if CythonHelpers.calcLocalSpread(xyz) == 0.0: # Too low resolution = no points in this object
        morph_info[:] = 0.0
        return morph_info
    morph_info[0] = r200+SAFE
    
    # Retrieve morphology
    morph_info[:] = runEllVDispAlgo(morph_info[:], xyz, vxyz, xyz_princ, masses, ellipsoid, center, vcenter, shape_tensor, eigval, eigvec, morph_info[0], M_TOL, N_WALL, N_MIN)
    return morph_info