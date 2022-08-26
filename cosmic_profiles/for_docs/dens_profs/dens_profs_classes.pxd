#cython: language_level=3
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from common.cosmic_base_class cimport CosmicBase

cdef class DensProfs(CosmicBase):
    
    cdef float[:,:] xyz
    cdef float[:] masses
    cdef object idx_cat
    cdef int[:] obj_size
    
cdef class DensProfsHDF5(CosmicBase):
    
    cdef str HDF5_SNAP_DEST
    cdef str HDF5_GROUP_DEST
    cdef int MIN_NUMBER_STAR_PTCS
    cdef bint WANT_RVIR