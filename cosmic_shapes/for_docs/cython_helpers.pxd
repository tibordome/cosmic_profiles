#cython: language_level=3
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 17:28:18 2022
"""

#cdef complex[::1,:] getShapeTensor(float[:,:] nns, int[:] select, complex[::1,:] shape_tensor, float[:] masses, float[:] com, int nb_pts) nogil

#cdef float getLocalSpread(float[:,:] nns) nogil

#cdef float[:] getCoM(float[:,:] nns, float[:] masses, float[:] com) nogil

#cdef float cython_abs(float x) nogil

#cdef void ZHEEVR(complex[::1,:] H, double * eigvals, complex[::1,:] Z, int nrows) nogil

#cdef float[:,:] respectPBCNoRef(float[:,:] xyz, float L_BOX) nogil
