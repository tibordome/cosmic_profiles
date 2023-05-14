#cython: language_level=3
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 17:28:18 2022
"""

cdef double[:] runShellAlgo(double[:] morph_info, double[:,:] xyz, double[:,:] xyz_princ, double[:] masses, int[:] shell, double[:] r_ell, double[:] center, complex[::1,:] shape_tensor, double[::1] eigval, complex[::1,:] eigvec, double d, double delta_d, double M_TOL, int N_WALL, int N_MIN, bint reduced) nogil

cdef double[:] runEllAlgo(double[:] morph_info, double[:,:] xyz, double[:,:] xyz_princ, double[:] masses, int[:] ellipsoid, double[:] r_ell, double[:] center, complex[::1,:] shape_tensor, double[::1] eigval, complex[::1,:] eigvec, double d, double M_TOL, int N_WALL, int N_MIN, bint reduced) nogil

cdef double[:] runEllVDispAlgo(double[:] morph_info, double[:,:] xyz, double[:,:] vxyz, double[:,:] xyz_princ, double[:] masses, int[:] ellipsoid, double[:] r_ell, double[:] center, double[:] vcenter, complex[::1,:] shape_tensor, double[::1] eigval, complex[::1,:] eigvec, double d, double M_TOL, int N_WALL, int N_MIN, bint reduced) nogil

cdef double[:] runShellVDispAlgo(double[:] morph_info, double[:,:] xyz, double[:,:] vxyz, double[:,:] xyz_princ, double[:] masses, int[:] shell, double[:] r_ell, double[:] center, double[:] vcenter, complex[::1,:] shape_tensor, double[::1] eigval, complex[::1,:] eigvec, double d, double delta_d, double M_TOL, int N_WALL, int N_MIN, bint reduced) nogil

cdef double[:,:] calcObjMorphLocal(double[:,:] morph_info, double r200, double[:] log_d, double[:,:] xyz, double[:,:] xyz_princ, double[:] masses, int[:] shell, double[:] r_ell, double[:] center, complex[::1,:] shape_tensor, double[::1] eigval, complex[::1,:] eigvec, double M_TOL, int N_WALL, int N_MIN, bint reduced, bint shell_based) nogil

cdef double[:] calcObjMorphGlobal(double[:] morph_info, double r200, double[:,:] xyz, double[:,:] xyz_princ, double[:] masses, int[:] ellipsoid, double[:] r_ell, double[:] center, complex[::1,:] shape_tensor, double[::1] eigval, complex[::1,:] eigvec, double M_TOL, int N_WALL, int N_MIN, double SAFE, bint reduced) nogil

cdef double[:,:] calcObjMorphLocalVelDisp(double[:,:] morph_info, double r200, double[:] log_d, double[:,:] xyz, double[:,:] vxyz, double[:,:] xyz_princ, double[:] masses, int[:] shell, double[:] r_ell, double[:] center, double[:] vcenter, complex[::1,:] shape_tensor, double[::1] eigval, complex[::1,:] eigvec, double M_TOL, int N_WALL, int N_MIN, bint reduced, bint shell_based) nogil

cdef double[:] calcObjMorphGlobalVelDisp(double[:] morph_info, double r200, double[:,:] xyz, double[:,:] vxyz, double[:,:] xyz_princ, double[:] masses, int[:] ellipsoid, double[:] r_ell, double[:] center, double[:] vcenter, complex[::1,:] shape_tensor, double[::1] eigval, complex[::1,:] eigvec, double M_TOL, int N_WALL, int N_MIN, double SAFE, bint reduced) nogil