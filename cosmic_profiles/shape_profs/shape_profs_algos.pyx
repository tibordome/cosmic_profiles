#cython: language_level=3
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 13:17:51 2021
"""

cimport openmp
import numpy as np
from cosmic_profiles.cython_helpers.helper_class cimport CythonHelpers
cimport cython
from libc.math cimport sqrt

@cython.embedsignature(True)
cdef float[:] runEllShellAlgo(float[:] morph_info, float[:,:] xyz, float[:,:] xyz_princ, float[:] masses, int[:] shell, float[:] center, complex[::1,:] shape_tensor, double[::1] eigval, complex[::1,:] eigvec, float d, float delta_d, float M_TOL, int N_WALL, int N_MIN) nogil:
    
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
        shape_tensor = CythonHelpers.getShapeTensor(xyz, shell, shape_tensor, masses, center, pts_in_shell)
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
        shape_tensor = CythonHelpers.getShapeTensor(xyz, ellipsoid, shape_tensor, masses, center, pts_in_ell)
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
        shape_tensor = CythonHelpers.getShapeTensor(vxyz, ellipsoid, shape_tensor, masses, vcenter, pts_in_ell)
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
