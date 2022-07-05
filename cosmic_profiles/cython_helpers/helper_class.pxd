#cython: language_level=3
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 17:28:18 2022
"""

cdef class CythonHelpers:

    @staticmethod
    cdef complex[::1,:] calcShapeTensor(float[:,:] nns, int[:] select, complex[::1,:] shape_tensor, float[:] masses, float[:] com, int nb_pts) nogil

    @staticmethod
    cdef float calcLocalSpread(float[:,:] nns) nogil

    @staticmethod
    cdef float[:] calcCoM(float[:,:] nns, float[:] masses, float[:] com) nogil

    @staticmethod
    cdef float cython_abs(float x) nogil

    @staticmethod
    cdef void ZHEEVR(complex[::1,:] H, double * eigvals, complex[::1,:] Z, int nrows) nogil

    @staticmethod
    cdef float[:,:] respectPBCNoRef(float[:,:] xyz, float L_BOX) nogil
    
    @staticmethod
    cdef float[:] calcDensProfBruteForce(float[:,:] xyz, float[:] masses, float[:] center, float r_200, float[:] ROverR200, float[:] dens_prof, int[:] shell) nogil
    
    @staticmethod
    cdef float[:] calcMenclsBruteForce(float[:,:] xyz, float[:] masses, float[:] center, float r_200, float[:] ROverR200, float[:] Mencl, int[:] ellipsoid) nogil
    
    @staticmethod
    cdef float calcKTilde(float r, float r_i, float h_i) nogil
