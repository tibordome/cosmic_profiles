#cython: language_level=3
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

cimport cython
from libc.math cimport sqrt, pi, exp

@cython.embedsignature(True)
cdef class CythonHelpers:
    
    def calcShapeTensor(float[:,:] nns, int[:] select, complex[::1,:] shape_tensor, float[:] masses, float[:] center, int nb_pts):
        """ Calculate shape tensor for point cloud
        
        :param nns: positions of cloud particles
        :type nns: (N,3) floats
        :param select: indices of cloud particles to consider
        :type select: (N1,3) ints
        :param shape_tensor: shape tensor array to be filled
        :type shape_tensor: (3,3) complex
        :param masses: masses of cloud particles
        :type masses: (N,) floats
        :param center: COM of point cloud
        :type center: (3,) floats
        :param nb_pts: number of points in `select` to consider
        :type nb_pts: int
        :return: shape tensor
        :rtype: (3,3) complex"""
        return
    
    def calcLocalSpread(float[:,:] nns):
        """ Calculate local spread (2nd moment) around center of volume of point cloud
        
        :param nns: positions of cloud particles
        :type nns: (N,3) floats
        :return: local spread
        :rtype: float"""
        return
    
    def calcCoM(float[:,:] nns, float[:] masses, float[:] com):
        """ Return center of mass (COM)
        
        :param nns: positions of cloud particles
        :type nns: (N,3) floats
        :param masses: masses of cloud particles
        :type masses: (N,) floats
        :param com: COM array to be filled
        :type com: (3,) floats
        :return: COM
        :rtype: (3,) floats"""
        return
    
    def cython_abs(float x):
        """ Absolute value of float
        
        :param x: float value of interest
        :type x: float
        :return: absolute value
        :rtype: float"""
        return

    def ZHEEVR(complex[::1,:] H, double[:] eigvals, complex[::1,:] Z, int nrows):
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
        return
    
    def respectPBCNoRef(float[:,:] xyz, float L_BOX):
        """
        Modify xyz inplace so that it respects the box periodicity.
        
        If point distro xyz has particles separated in any Cartesian direction
        by more than L_BOX/2, reflect those particles along L_BOX/2
        
        :param xyz: coordinates of particles of type 1 or type 4
        :type xyz: (N^3x3) floats
        :param ref: reference particle, which does not matter in the case of
            halo morphology analysis
        :type ref: int
        :return: updated coordinates of particles of type 1 or type 4
        :rtype: (N^3x3) floats"""
        
        return
    
    def calcDensProfBruteForce(float[:,:] xyz, float[:] masses, float[:] center, float r_200, float[:] rad_bins, float[:] dens_prof, int[:] shell):
        """ Calculates density profile for one object with coordinates `xyz` and masses `masses`
        
        :param xyz: positions of cloud particles
        :type xyz: (N,3) floats
        :param masses: masses of cloud particles
        :type masses: (N,) floats
        :param center: center of the object
        :type center: (3) floats
        :param r_200: R200 value of the object
        :type r_200: float
        :param rad_bins: radial bins (bin edges) at whose centers the density profiles
            should be calculated, normalized by R200
        :type rad_bins: float array
        :param dens_prof: array to store result in
        :type dens_prof: float array
        :param shell: array used for the calculation
        :type shell: int array
        :return: density profile
        :rtype: float array"""
        return
    
    def calcMenclsBruteForce(float[:,:] xyz, float[:] masses, float[:] center, float r_200, float[:] ROverR200, float[:] Mencl, int[:] ellipsoid):
        """ Calculates enclosed mass profile for one object with coordinates `xyz` and masses `masses`
        
        :param xyz: positions of cloud particles
        :type xyz: (N,3) floats
        :param masses: masses of cloud particles
        :type masses: (N,) floats
        :param center: center of the object
        :type center: (3) floats
        :param r_200: R200 value of the object
        :type r_200: float
        :param ROverR200: radii at which the density profiles should be calculated,
            normalized by R200
        :type ROverR200: float array
        :param Mencl: array to store result in
        :type Mencl: float array
        :param ellipsoid: array used for the calculation
        :type ellipsoid: int array
        :return: enclosed mass profile
        :rtype: float array"""
        return
    
    def calcKTilde(float r, float r_i, float h_i):
        """ Angle-averaged normalized Gaussian kernel for kernel-based density profile estimation
        
        :param r: radius in Mpc/h at which to calculate the local, spherically-averaged density
        :type r: float
        :param r_i: radius of point i whose contribution to the local, spherically-averaged density shall be determined
        :type r_i: float
        :param h_i: width of Gaussian kernel for point i
        :type h_i: float
        :return: angle-averaged normalized Gaussian kernel
        :rtype: float"""
        return