#cython: language_level=3
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 17:28:18 2022
"""

cdef class CythonHelpers:

    @staticmethod
    cdef complex[::1,:] getShapeTensor(float[:,:] nns, int[:] select, complex[::1,:] shape_tensor, float[:] masses, float[:] com, int nb_pts) nogil

    @staticmethod
    cdef float getLocalSpread(float[:,:] nns) nogil

    @staticmethod
    cdef float[:] getCoM(float[:,:] nns, float[:] masses, float[:] com) nogil

    @staticmethod
    cdef float cython_abs(float x) nogil

    @staticmethod
    cdef void ZHEEVR(complex[::1,:] H, double * eigvals, complex[::1,:] Z, int nrows) nogil

    @staticmethod
    cdef float[:,:] respectPBCNoRef(float[:,:] xyz, float L_BOX) nogil
