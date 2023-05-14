#cython: language_level=3
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

cdef class CythonHelpers:

    @staticmethod
    cdef complex[::1,:] calcShapeTensor(double[:,:] nns, int[:] select, complex[::1,:] shape_tensor, double[:] masses, double[:] center, int nb_pts, bint reduced, double[:] r_ell = *) nogil
    
    @staticmethod
    cdef float calcLocalSpread(double[:,:] nns) nogil

    @staticmethod
    cdef double[:] calcCoM(double[:,:] nns, double[:] masses, double[:] com) nogil

    @staticmethod
    cdef float cython_abs(float x) nogil

    @staticmethod
    cdef void ZHEEVR(complex[::1,:] H, double * eigvals, complex[::1,:] Z, int nrows) nogil

    @staticmethod
    cdef double[:,:] respectPBCNoRef(double[:,:] xyz, float L_BOX) nogil
    
    @staticmethod
    cdef double[:] calcDensProfBruteForceSph(double[:,:] xyz, double[:] masses, double[:] center, float r_200, double[:] ROverR200, double[:] dens_prof, int[:] shell) nogil
    
    @staticmethod
    cdef double[:] calcMenclsBruteForceSph(double[:,:] xyz, double[:] masses, double[:] center, float r_200, double[:] ROverR200, double[:] Mencl, int[:] ellipsoid) nogil
    
    @staticmethod
    cdef double[:] calcDensProfBruteForceEll(double[:,:] xyz, double[:,:] xyz_princ, double[:] masses, double[:] center, float r_200, double[:] a, double[:] b, double[:] c, double[:,:] major, double[:,:] inter, double[:,:] minor, double[:] dens_prof, int[:] shell) nogil
            
    @staticmethod
    cdef double[:] calcMenclsBruteForceEll(double[:,:] xyz, double[:,:] xyz_princ, double[:] masses, double[:] center, double[:] a, double[:] b, double[:] c, double[:,:] major, double[:,:] inter, double[:,:] minor, double[:] dens_prof, int[:] ellipsoid) nogil
        
    @staticmethod
    cdef float calcKTilde(float r, float r_i, float h_i) nogil
