#cython: language_level=3
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from common.cosmic_base_class cimport CosmicBase

cdef class DensProfsBase(CosmicBase):
    
    cdef float[:,:] xyz
    cdef float[:] masses
    cdef int[:] idx_cat
    cdef int[:] obj_size
    
cdef class DensProfsGadget(DensProfsBase):
    
    cdef str HDF5_SNAP_DEST
    cdef str HDF5_GROUP_DEST
    cdef str RVIR_OR_R200
    cdef str OBJ_TYPE