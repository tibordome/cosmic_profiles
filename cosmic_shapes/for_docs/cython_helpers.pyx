#cython: language_level=3
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 12:16:48 2022
"""

cimport cython
from libc.math cimport sqrt
from scipy.linalg.cython_lapack cimport zheevr
include "array_defs.pxi"

@cython.embedsignature(True)
@cython.boundscheck(False)
@cython.wraparound(False) 
def getShapeTensor(float[:,:] nns, int[:] select, complex[::1,:] shape_tensor, float[:] masses, float[:] com, int nb_pts):
    """ Calculate shape tensor for point cloud
    
    :param nns: positions of cloud particles
    :type nns: (N,3) floats
    :param select: indices of cloud particles to consider
    :type select: (N1,3) ints
    :param shape_tensor: shape tensor array to be filled
    :type shape_tensor: (3,3) complex, zeros
    :param masses: masses of cloud particles
    :type masses: (N,) floats
    :param com: COM of point cloud
    :type com: (3,) floats
    :param nb_pts: number of points in `select` to consider
    :type nb_pts: int
    :return: shape tensor
    :rtype: (3,3) complex"""
    shape_tensor[:,:] = 0.0
    cdef float mass_tot = 0.0
    cdef int run
    for run in range(nb_pts):
        mass_tot = mass_tot+masses[select[run]]
    cdef int i
    cdef int j
    for i in range(3):
        for j in range(3):
            if i >= j:
                for run in range(nb_pts):
                    shape_tensor[i,j] = shape_tensor[i,j] + <complex>(masses[select[run]]*(nns[select[run],i]-com[i])*(nns[select[run],j]-com[j])/mass_tot)
                if i > j:
                    shape_tensor[j,i] = shape_tensor[i,j]
    return shape_tensor.base

@cython.embedsignature(True)
@cython.boundscheck(False)
@cython.wraparound(False) 
def getLocalSpread(float[:,:] nns):
    """ Calculate local spread (2nd moment) around center of volume of point cloud
    
    :param nns: positions of cloud particles
    :type nns: (N,3) floats
    :return: local spread
    :rtype: float"""
    cdef float loc_spread = 0
    cdef float nns_mean_x = 0
    cdef float nns_mean_y = 0
    cdef float nns_mean_z = 0
    cdef int i
    for i in range(nns.shape[0]):
        nns_mean_x += nns[i,0]/nns.shape[0]
        nns_mean_y += nns[i,1]/nns.shape[0]
        nns_mean_z += nns[i,2]/nns.shape[0]
    for i in range(nns.shape[0]):
        loc_spread += sqrt((nns[i,0]-nns_mean_x)**2+(nns[i,1]-nns_mean_y)**2+(nns[i,2]-nns_mean_z)**2)/nns.shape[0]
    return loc_spread

@cython.embedsignature(True)
@cython.boundscheck(False)
@cython.wraparound(False) 
def getCoM(float[:,:] nns, float[:] masses, float[:] com):
    """ Return center of mass (COM)
    
    :param nns: positions of cloud particles
    :type nns: (N,3) floats
    :param masses: masses of cloud particles
    :type masses: (N,) floats
    :param com: COM array to be filled
    :type com: (3,) floats
    :return: COM
    :rtype: (3,) floats"""
    com[:] = 0.0
    cdef int run
    cdef float mass_total = 0.0
    for run in range(nns.shape[0]):
        mass_total += masses[run]
    for run in range(nns.shape[0]):
        com[0] += masses[run]*nns[run,0]/mass_total
        com[1] += masses[run]*nns[run,1]/mass_total
        com[2] += masses[run]*nns[run,2]/mass_total
    return com

@cython.embedsignature(True)
@cython.boundscheck(False)
@cython.wraparound(False) 
def cython_abs(float x):
    """ Absolute value of float
    
    :param x: float value of interest
    :type x: float
    :return: absolute value
    :rtype: float"""
    if x >= 0.0:
        return x
    if x < 0.0:
        return -x

@cython.embedsignature(True)
def ZHEEVR(complex[::1,:] H, double[::1] eigvals, complex[::1,:] Z, int nrows):
    """
    Computes the eigenvalues and eigenvectors of a dense Hermitian matrix.
    
    Eigenvectors are returned in Z.
    
    :param H: Hermitian matrix
    :type H: Fortran-ordered 2D double complex memoryview
    :param eigvals: Input array to store eigenvalues
    :type eigvals: double array like
    :param Z: Output array of eigenvectors
    :type Z: Fortran-ordered 2D double complex memoryview
    :param nrows: Number of rows = number of columns in H
    :type nrows: int 
    :raises: Exception if not converged
    """
    cdef char jobz = b'V'
    cdef char rnge = b'A'
    cdef char uplo = b'L'
    cdef double vl=1, vu=1, abstol=0
    cdef int il=1, iu=1
    cdef int lwork = 18 * nrows
    cdef int lrwork = 24*nrows, liwork = 10*nrows
    cdef int info=0, M=0
    #These need to be freed at the end
    cdef int * isuppz = <int *>PyDataMem_NEW((2*nrows) * sizeof(int))
    cdef complex * work = <complex *>PyDataMem_NEW(lwork * sizeof(complex))
    cdef double * rwork = <double *>PyDataMem_NEW((24*nrows) * sizeof(double))
    cdef int * iwork = <int *>PyDataMem_NEW((10*nrows) * sizeof(int))
    
    if info != 0:
        if info < 0:
            raise Exception("Error in parameter : %s" & abs(info))
        else:
            raise Exception("Algorithm failed to converge")
            
@cython.embedsignature(True)
@cython.boundscheck(False)
@cython.wraparound(False) 
def respectPBCNoRef(float[:,:] xyz, float L_BOX):
    """
    Modify xyz inplace so that it respects the box periodicity
    
    If point distro xyz has particles separated in any Cartesian direction
    by more than L_BOX/2, reflect those particles along L_BOX/2
    
    :param xyz: coordinates of particles of type 1 or type 4
    :type xyz: (N^3x3) floats
    :param ref: reference particle, which does matter in this case 
        (unlike halo morphology analysis), e.g. Covol
    :type ref: int
    :return: updated coordinates of particles of type 1 or type 4
    :rtype: (N^3x3) floats"""
    
    cdef int ref = 0
    cdef float dist_x
    cdef float dist_y
    cdef float dist_z
    for i in range(xyz.shape[0]):
        dist_x = cython_abs(xyz[ref, 0]-xyz[i,0])
        if dist_x > L_BOX/2:
            xyz[i,0] = L_BOX-xyz[i,0] # Reflect x-distances along L_BOX/2
        dist_y = cython_abs(xyz[ref, 1]-xyz[i,1])
        if dist_y > L_BOX/2:
            xyz[i,1] = L_BOX-xyz[i,1] # Reflect y-distances along L_BOX/2
        dist_z = cython_abs(xyz[ref, 1]-xyz[i,1])
        if dist_z > L_BOX/2:
            xyz[i,2] = L_BOX-xyz[i,2] # Reflect z-distances along L_BOX/2
    return xyz