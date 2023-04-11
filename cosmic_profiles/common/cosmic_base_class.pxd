#cython: language_level=3
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

cimport cython

@cython.embedsignature(True)
cdef class CosmicBase:
    
    cdef str SNAP
    cdef float L_BOX
    cdef double start_time
    cdef float[:] r200
    cdef str CENTER
    cdef str VIZ_DEST
    cdef str CAT_DEST
    cdef float SAFE # Units: Mpc/h. Ellipsoidal radius will be maxdist(COM,point)+SAFE where point is any point in the point cloud. The larger the better.
    cdef float MASS_UNIT
    cdef int MIN_NUMBER_PTCS
    cdef str SUFFIX