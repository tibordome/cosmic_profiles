#cython: language_level=3
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

cimport cython
from libc.math cimport sqrt, pi, exp
from scipy.linalg.cython_lapack cimport zheevr
from cython.parallel import prange
from cosmic_profiles.common.python_routines import respectPBCNoRef, calcCoM, calcMode
from cosmic_profiles.common.caching import np_cache_factory
include "array_defs.pxi"

@cython.embedsignature(True)
cdef class CythonHelpers:

    @cython.boundscheck(False)
    @cython.wraparound(False) 
    @staticmethod
    cdef complex[::1,:] calcShapeTensor(float[:,:] nns, int[:] select, complex[::1,:] shape_tensor, float[:] masses, float[:] center, int nb_pts) nogil:
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
                        shape_tensor[i,j] = shape_tensor[i,j] + <complex>(masses[select[run]]*(nns[select[run],i]-center[i])*(nns[select[run],j]-center[j])/mass_tot)
                    if i > j:
                        shape_tensor[j,i] = shape_tensor[i,j]
        return shape_tensor

    @cython.boundscheck(False)
    @cython.wraparound(False) 
    @staticmethod
    cdef float calcLocalSpread(float[:,:] nns) nogil:
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

    @cython.boundscheck(False)
    @cython.wraparound(False) 
    @staticmethod
    cdef float[:] calcCoM(float[:,:] nns, float[:] masses, float[:] com) nogil:
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

    @cython.boundscheck(False)
    @cython.wraparound(False) 
    @staticmethod
    cdef float cython_abs(float x) nogil:
        """ Absolute value of float
        
        :param x: float value of interest
        :type x: float
        :return: absolute value
        :rtype: float"""
        if x >= 0.0:
            return x
        if x < 0.0:
            return -x

    @staticmethod
    cdef void ZHEEVR(complex[::1,:] H, double * eigvals, complex[::1,:] Z, int nrows) nogil:
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

        zheevr(&jobz, &rnge, &uplo, &nrows, &H[0,0], &nrows, &vl, &vu, &il, &iu, &abstol, &M, eigvals, &Z[0,0], &nrows, isuppz, work, &lwork, rwork, &lrwork, iwork, &liwork, &info)
        PyDataMem_FREE(work)
        PyDataMem_FREE(rwork)
        PyDataMem_FREE(isuppz)
        PyDataMem_FREE(iwork)
        if info != 0:
            if info < 0:
                raise Exception("Error in parameter : %s" & abs(info))
            else:
                raise Exception("Algorithm failed to converge")
                
    @cython.boundscheck(False)
    @cython.wraparound(False) 
    @staticmethod
    cdef float[:,:] respectPBCNoRef(float[:,:] xyz, float L_BOX) nogil:
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
        
        cdef int ref = 0
        cdef float dist_x
        cdef float dist_y
        cdef float dist_z
        for i in range(xyz.shape[0]):
            dist_x = CythonHelpers.cython_abs(xyz[ref, 0]-xyz[i,0])
            if dist_x > L_BOX/2:
                xyz[i,0] = L_BOX-xyz[i,0] # Reflect x-distances along L_BOX/2
            dist_y = CythonHelpers.cython_abs(xyz[ref, 1]-xyz[i,1])
            if dist_y > L_BOX/2:
                xyz[i,1] = L_BOX-xyz[i,1] # Reflect y-distances along L_BOX/2
            dist_z = CythonHelpers.cython_abs(xyz[ref, 1]-xyz[i,1])
            if dist_z > L_BOX/2:
                xyz[i,2] = L_BOX-xyz[i,2] # Reflect z-distances along L_BOX/2
        return xyz
    
    @cython.boundscheck(False)
    @cython.wraparound(False) 
    @staticmethod
    cdef float[:] calcDensProfBruteForce(float[:,:] xyz, float[:] masses, float[:] center, float r_200, float[:] rad_bins, float[:] dens_prof, int[:] shell) nogil:
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
        cdef int corr_s = 0
        cdef int pts_in_shell = 0
        cdef int r_i
        cdef int n
        cdef int i
        for r_i in range(rad_bins.shape[0]-1):
            corr_s = 0
            pts_in_shell = 0
            for i in range(xyz.shape[0]):
                if (center[0]-xyz[i,0])**2+(center[1]-xyz[i,1])**2+(center[2]-xyz[i,2])**2 < (rad_bins[r_i+1]*r_200)**2 and (center[0]-xyz[i,0])**2+(center[1]-xyz[i,1])**2+(center[2]-xyz[i,2])**2 >= (rad_bins[r_i]*r_200)**2:
                    shell[i-corr_s] = i
                    pts_in_shell += 1
                else:
                    corr_s += 1
            if pts_in_shell != 0:
                for n in range(pts_in_shell):
                    dens_prof[r_i] = dens_prof[r_i] + masses[shell[n]]/(4/3*pi*((rad_bins[r_i+1]*r_200)**3-(rad_bins[r_i]*r_200)**3))
        return dens_prof
    
    @cython.boundscheck(False)
    @cython.wraparound(False) 
    @staticmethod
    cdef float[:] calcMenclsBruteForce(float[:,:] xyz, float[:] masses, float[:] center, float r_200, float[:] ROverR200, float[:] Mencl, int[:] ellipsoid) nogil:
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
        cdef int corr_ell = 0
        cdef int pts_in_ell = 0
        cdef int r_i
        cdef int n
        cdef int i
        for r_i in range(ROverR200.shape[0]):
            corr_ell = 0
            pts_in_ell = 0
            for i in range(xyz.shape[0]):
                if (center[0]-xyz[i,0])**2+(center[1]-xyz[i,1])**2+(center[2]-xyz[i,2])**2 < (ROverR200[r_i]*r_200)**2:
                    ellipsoid[i-corr_ell] = i
                    pts_in_ell += 1
                else:
                    corr_ell += 1
            if pts_in_ell != 0:
                for n in range(pts_in_ell):
                    Mencl[r_i] = Mencl[r_i] + masses[ellipsoid[n]]
        return Mencl
    
    @cython.boundscheck(False)
    @cython.wraparound(False) 
    @staticmethod
    cdef float calcKTilde(float r, float r_i, float h_i) nogil:
        """ Angle-averaged normalized Gaussian kernel for kernel-based density profile estimation
        
        :param r: radius in Mpc/h at which to calculate the local, spherically-averaged density
        :type r: float
        :param r_i: radius of point i whose contribution to the local, spherically-averaged density shall be determined
        :type r_i: float
        :param h_i: width of Gaussian kernel for point i
        :type h_i: float
        :return: angle-averaged normalized Gaussian kernel
        :rtype: float"""
        return 1/(2*(2*pi)**(3/2))*(r*r_i/(h_i**2))**(-1)*(exp(-(r_i-r)**2/(2*h_i**2))-exp(-(r_i+r)**2/(2*h_i**2)))