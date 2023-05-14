#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng
import math
from scipy.linalg import cholesky
from numpy.linalg import inv

def eTo10(st):
    """Replace e+{xy} by "10^{xy}" etc..
    
    Note that "st" should be handed over with {:.2E}.
    Works for exponents 00 to 19, plus and minus, e and E.
    
    :Example: 
    
    >>> area = 2
    >>> print("E.g. internally we write '\u00b2' for 'square'. The area of your rectangle is {} cm\u00b2".format(area))
    >>> test = "2.88E-19"
    >>> print("The string", test, "becomes", eTo10(test))"""
    st_l = list(st)
    i = 0
    drop_last_char = True
    while i < len(st_l):
        if st_l[i] == "e" or st_l[i] == "E":
            if st_l[i+2] == "0" and st_l[i+3] == "0":
                if st_l[i] == "e":
                    split_string = st.split("e", 1)
                else:
                    split_string = st.split("E", 1)
                return split_string[0]
            if st_l[i+1] == "+":
                st_l[i] = "1"
                st_l[i+1] = "0"
                if st_l[i+2] == "0" and st_l[i+3] == "1":
                    st_l[i+2] = "\u00B9"
                elif st_l[i+2] == "0" and st_l[i+3] == "2":
                    st_l[i+2] = "\u00b2"
                elif st_l[i+2] == "0" and st_l[i+3] == "3":
                    st_l[i+2] = "\u00B3"
                elif st_l[i+2] == "0" and st_l[i+3] == "4":
                    st_l[i+2] = "\u2074"
                elif st_l[i+2] == "0" and st_l[i+3] == "5":
                    st_l[i+2] = "\u2075"
                elif st_l[i+2] == "0" and st_l[i+3] == "6":
                    st_l[i+2] = "\u2076"
                elif st_l[i+2] == "0" and st_l[i+3] == "7":
                    st_l[i+2] = "\u2077"
                elif st_l[i+2] == "0" and st_l[i+3] == "8":
                    st_l[i+2] = "\u2078"
                elif st_l[i+2] == "0" and st_l[i+3] == "9":
                    st_l[i+2] = "\u2079"
                elif st_l[i+2] == "1" and st_l[i+3] == "0":
                    st_l[i+2] = "\u00B9"
                    st_l[i+3] = "\u2070"
                    drop_last_char = False
                elif st_l[i+2] == "1" and st_l[i+3] == "1":
                    st_l[i+2] = "\u00B9"
                    st_l[i+3] = "\u00B9"
                    drop_last_char = False
                elif st_l[i+2] == "1" and st_l[i+3] == "2":
                    st_l[i+2] = "\u00B9"
                    st_l[i+3] = "\u00b2"
                    drop_last_char = False
                elif st_l[i+2] == "1" and st_l[i+3] == "3":
                    st_l[i+2] = "\u00B9"
                    st_l[i+3] = "\u00B3"
                    drop_last_char = False
                elif st_l[i+2] == "1" and st_l[i+3] == "4":
                    st_l[i+2] = "\u00B9"
                    st_l[i+3] = "\u2074"
                    drop_last_char = False
                elif st_l[i+2] == "1" and st_l[i+3] == "5":
                    st_l[i+2] = "\u00B9"
                    st_l[i+3] = "\u2075"
                    drop_last_char = False
                elif st_l[i+2] == "1" and st_l[i+3] == "6":
                    st_l[i+2] = "\u00B9"
                    st_l[i+3] = "\u2076"
                    drop_last_char = False
                elif st_l[i+2] == "1" and st_l[i+3] == "7":
                    st_l[i+2] = "\u00B9"
                    st_l[i+3] = "\u2077"
                    drop_last_char = False
                elif st_l[i+2] == "1" and st_l[i+3] == "8":
                    st_l[i+2] = "\u00B9"
                    st_l[i+3] = "\u2078"
                    drop_last_char = False
                else:
                    assert st_l[i+2] == "1" and st_l[i+3] == "9"
                    st_l[i+2] = "\u00B9"
                    st_l[i+3] = "\u2079"
                    drop_last_char = False
                    
                st_l.insert(i, "\u2800") # Add blank character
                st_l.insert(i+1, "\u2715") # Add \times character
                st_l.insert(i+2, "\u2800") # Add blank character
                if drop_last_char == True:
                    st_l = st_l[:-1]
            else:
                assert st_l[i+1] == "-"
                st_l[i] = "1"
                st_l[i+1] = "0"
                    
                if st_l[i+2] == "0" and st_l[i+3] == "1":
                    st_l[i+3] = "\u00B9"
                elif st_l[i+2] == "0" and st_l[i+3] == "2":
                    st_l[i+3] = "\u00b2"
                elif st_l[i+2] == "0" and st_l[i+3] == "3":
                    st_l[i+3] = "\u00B3"
                elif st_l[i+2] == "0" and st_l[i+3] == "4":
                    st_l[i+3] = "\u2074"
                elif st_l[i+2] == "0" and st_l[i+3] == "5":
                    st_l[i+3] = "\u2075"
                elif st_l[i+2] == "0" and st_l[i+3] == "6":
                    st_l[i+3] = "\u2076"
                elif st_l[i+2] == "0" and st_l[i+3] == "7":
                    st_l[i+3] = "\u2077"
                elif st_l[i+2] == "0" and st_l[i+3] == "8":
                    st_l[i+3] = "\u2078"
                elif st_l[i+2] == "0" and st_l[i+3] == "9":
                    st_l[i+3] = "\u2079"
                elif st_l[i+2] == "1" and st_l[i+3] == "0":
                    st_l[i+3] = "\u00B9"
                    st_l += "\u2070"
                elif st_l[i+2] == "1" and st_l[i+3] == "1":
                    st_l[i+3] = "\u00B9"
                    st_l += "\u00B9"
                elif st_l[i+2] == "1" and st_l[i+3] == "2":
                    st_l[i+3] = "\u00B9"
                    st_l += "\u00b2"
                elif st_l[i+2] == "1" and st_l[i+3] == "3":
                    st_l[i+3] = "\u00B9"
                    st_l += "\u00B3"
                elif st_l[i+2] == "1" and st_l[i+3] == "4":
                    st_l[i+3] = "\u00B9"
                    st_l += "\u2074"
                elif st_l[i+2] == "1" and st_l[i+3] == "5":
                    st_l[i+3] = "\u00B9"
                    st_l += "\u2075"
                elif st_l[i+2] == "1" and st_l[i+3] == "6":
                    st_l[i+3] = "\u00B9"
                    st_l += "\u2076"
                elif st_l[i+2] == "1" and st_l[i+3] == "7":
                    st_l[i+3] = "\u00B9"
                    st_l += "\u2077"
                elif st_l[i+2] == "1" and st_l[i+3] == "8":
                    st_l[i+3] = "\u00B9"
                    st_l += "\u2078"
                else:
                    assert st_l[i+2] == "1" and st_l[i+3] == "9"
                    st_l[i+3] = "\u00B9"
                    st_l += "\u2079"
                
                st_l[i+2] = "\u207b" # Minus in the exponent
                st_l.insert(i, "\u2800") # Add blank character space
                st_l.insert(i+1, "\u2715") # Add \times character
                st_l.insert(i+2, "\u2800") # Add blank character space
        i += 1
    if st_l[0] == "1" and st_l[1] == "\u002E" and st_l[2] == "0" and st_l[3] == "0": # Remove 1.00 x 
        return "".join(st_l[7:])
    return "".join(st_l)

def print_status(rank,start_time,message,**kwargs):
    """ Routine to print script status to command line, with elapsed time
    
    :param rank: rank of process
    :type rank: int
    :param start_time: time of start of script (or specific function)
    :type start_time: float
    :param message: message to be printed
    :type message: string
    :param color: for optional ``color`` argument, we allow for 'green', 'blue' and 'red'
    :type color: string"""
    if rank == 0 or ("allowed_any" in kwargs and kwargs['allowed_any'] == True):
        elapsed_time = time.time() - start_time
        CEND = ' \033[0m'
        CRED = '\33[33m '
        CGREEN = '\033[32m '
        CBLUE = '\33[34m '
        if not ("color" in kwargs):
            print('%d\ts: %s' % (elapsed_time,message))
        elif ("color" in kwargs and kwargs['color'] == 'green'):
            print(CGREEN + '%d\ts: %s' % (elapsed_time,message) + CEND)
        elif ("color" in kwargs and kwargs['color'] == 'red'):
            print(CRED + '%d\ts: %s' % (elapsed_time,message) + CEND)
        else:
            assert ("color" in kwargs and kwargs['color'] == 'blue')
            print(CBLUE + '%d\ts: %s' % (elapsed_time,message) + CEND)
            
def getRhoCrit():
    """ Returns critical comoving density of the universe
    
    Return value is in units of [solar masses/(cMpc)^3*h^2]
    
    :return: critical comoving density
    :rtype: float"""
    G = 6.674*10**(-29)/(3.086*10**16)**3 # in (cMpc)^3/(kg*s^2)
    H_z = 100/(3.086*10**19) # in 1/s
    solar_mass = 2*10**30 # in kg
    return 3*H_z**2/(8*np.pi*G)/solar_mass

def getMassDMParticle(N, L):
    """Returns the mass of each DM particle,
    assuming it is constant, given N and L
    
    :param N: Simulation resolution parameter, usually power of 2
    :type N: int
    :param L: Simulation box size in cMpc/h
    :type L: float
    :return: Mass of each DM particle in solar masses*h^-1
    :rtype: float"""
    return getRhoCrit()*(L/N)**3

def getDelta(z, OMEGA_M, OMEGA_L):
    """ Calculates the overdensity required for spherical virialization 
    relative to the mean background of the universe
    
    The result is taken from Bryan, Norman, 1998, ApJ 
    (Vir. Theorem + Spher. Collapse). Note that this quantity is z-dependent.
    
    :param z: redshift of interest
    :type z: float
    :param OMEGA_M: fractional matter density of the universe
    :type OMEGA_M: float
    :param OMEGA_L: fractional dark energy density of the universe
    :type OMEGA_L: float
    :return: spherical virialization overdensity
    :rtype: float"""
    x = (OMEGA_M*(1+z)**3)/(OMEGA_M*(1+z)**3+OMEGA_L)-1
    DELTA = (18*np.pi**2 + 82*x - 39*x**2)/(x+1) - 1
    return np.float64(DELTA)

def set_axes_equal(ax: plt.Axes):
    """Set 3D plot axes to equal scale.

    Make axes of 3D plot have equal scale so that spheres appear as
    spheres and cubes as cubes.  Required since ``ax.axis('equal')``
    and ``ax.set_aspect('equal')`` don't work on 3D.
    """
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)

def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])

def fibonacci_sphere(samples=1):
    """ Creating "evenly" distributed points on a unit sphere
    
    This distribution of points on the unit sphere is called a Fibonacci sphere
    
    :param samples: number of points to be put onto the unit sphere
    :type samples: int
    :return: N points on the unit sphere
    :rtype: list of N (3,) floats"""

    points = []
    phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        points.append((x, y, z))

    return points

def calcMode(xyz, masses, rad):
    """ Find mode (point of highest local density) of point distribution xyz
    
    :param xyz: coordinates of particles of type 1 or type 4
    :type xyz: (N,3) floats
    :param masses: masses of the particles
    :type masses: (N,) floats
    :param rad: initial radius to consider from CoM of object
    :type rad: float
    :return: mode of distro
    :rtype: (3,) floats"""
    com = np.sum(xyz*np.reshape(masses, (masses.shape[0],1)), axis = 0)/masses.sum()
    distances_all = np.linalg.norm(xyz-com,axis=1)
    xyz_constrain = xyz[distances_all < rad]
    masses_constrain = masses[distances_all < rad]
    if xyz_constrain.shape[0] < 5: # If only < 5 particles left, return
        return com
    else:
        rad *= 0.83 # Reduce radius by 17 %
        return calcMode(xyz_constrain, masses_constrain, rad)
        
def fibonacci_ellipsoid(a, b, c, samples=1):
    """ Creating "evenly" distributed points on an ellipsoid surface
    
    Here, this distribution of points on the ellipsoid surface is 
    called a Fibonacci ellipsoid
    
    :param a: major axis of ellipsoid surface
    :type a: float
    :param b: intermediate axis of ellipsoid surface
    :type b: float
    :param c: minor axis of ellipsoid surface
    :type c: float
    :param samples: number of points to be put onto the unit sphere
    :type samples: int
    :return: N points on the unit sphere
    :rtype: list of N (3,) floats"""

    points = []
    phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        points.append((x*a, y*b, z*c))

    return points

def drawUniformFromEllipsoid(dims, a, b, c, Nptc, ell_nb):
    """ Draw points uniformly from an ellipsoid volume, centered at the origin
    
    This function is primarily used to generate synthetic halos.
    The approach is taken from Rubinstein & Kroese 2007:\n
    Part 1: Generating uniform random vectors inside the 3-ball\n
    Part 2: Lower Cholesky decomposition\n
    Part 3: Draw uniformly from ellipsoid
    
    :param dims: number of dimensions of 'ellipsoid', either 2,
        in which case c remains 'None', or 3
    :type dims: int
    :param a: major axis array
    :type a: float array, units are Mpc/h
    :param b: intermediate axis array
    :type b: float array, units are Mpc/h
    :param c: minor axis array
    :type c: float array, units are Mpc/h
    :param Nptc: number of particles in each ellipsoid Ell(a[idx], b[idx], c[idx]).
    :type Nptc: int array
    :param ell_nb: ellipsoid number of interest
    :type ell_nb: int
    :return: points drawn uniformly from an ellipsoid volume, ellipsoid number
    :rtype: (N_reals,3) floats, 1 int"""
    
    # Part 1: Generating uniform random vectors inside the 3-ball
    a_val = a[ell_nb]
    b_val = b[ell_nb]
    c_val = c[ell_nb]
    N_reals = Nptc[ell_nb]
    X_tmp = np.zeros((N_reals, dims))
    Z = np.zeros((N_reals, dims))
    np.random.seed(ell_nb)
    for i in range(N_reals):
        X_tmp[i] = np.random.normal(0,1,dims)
        R = (np.random.uniform(0,1,1))**(1/dims)
        Z[i] = R*X_tmp[i]/np.linalg.norm(X_tmp[i])
    
    # Part 2: lower Cholesky decomposition
    if c_val == None:
        Sigma = np.array([[1/a_val**2, 0],[0,1/b_val**2]])
    else:
        Sigma = np.array([[1/a_val**2, 0,0],[0,1/b_val**2,0],[0,0,1/c_val**2]])
    L = cholesky(Sigma, lower=True)
    
    # Part 3: Uniform from Ellipsoid
    X = np.zeros((N_reals, dims))
    for i in range(N_reals):
        X[i] = np.dot(inv(L.T),Z[i])
    return X, ell_nb

def inShell(X, a, b, c, shell_nb):
    """ Determine whether point `X` is in Shell(a[shell_nb], b[shell_nb], c[shell_nb])
    
    :param X: point of interest
    :type X: (3,) float array
    :param a: major axis array
    :type a: float array, units are Mpc/h
    :param b: intermediate axis array
    :type b: float array, units are Mpc/h
    :param c: minor axis array
    :type c: float array, units are Mpc/h
    :param shell_nb: ellipsoid number of interest
    :type shell_nb: int
    :return: True if `X` is in Shell(a[shell_nb], b[shell_nb], c[shell_nb]), False otherwise
    :rtype: boolean
    """
    if shell_nb > 0: # The principal frame coincides with the Cartesian xyz-frame, so no need to rotate
        if (X[0]/a[shell_nb])**2 + (X[1]/b[shell_nb])**2 + (X[2]/c[shell_nb])**2 <= 1 and (X[0]/a[shell_nb-1])**2 + (X[1]/b[shell_nb-1])**2 + (X[2]/c[shell_nb-1])**2 > 1:
            return True
        else:
            return False
    else:
        if (X[0]/a[shell_nb])**2 + (X[1]/b[shell_nb])**2 + (X[2]/c[shell_nb])**2 <= 1:
            return True
        else:
            return False

def drawUniformFromShell(dims, a, b, c, Nptc, shell_nb):
    """ Draw points uniformly from an ellipsoidal shell volume, centered at the origin
    
    This function is primarily used to generate synthetic halos.
    The approach is taken from Rubinstein & Kroese 2007:\n
    Part 1: Generating uniform random vectors inside the 3-ball\n
    Part 2: Lower Cholesky decomposition\n
    Part 3: Draw uniformly from Ball(a[shell_nb])
    Part 4: Move ptcs from ball Ball(a[shell_nb]) into ellipsoidal
    shell Shell(a[shell_nb-1],b[shell_nb-1],c[shell_nb-1],a[shell_nb],b[shell_nb],c[shell_nb])
    
    :param dims: number of dimensions of 'ellipsoid', either 2,
        in which case c remains 'None', or 3
    :type dims: int
    :param a: major axis array
    :type a: float array, units are Mpc/h
    :param b: intermediate axis array
    :type b: float array, units are Mpc/h
    :param c: minor axis array
    :type c: float array, units are Mpc/h
    :param Nptc: number of particles in each shell 
        Shell(a[shell_nb-1],b[shell_nb-1],c[shell_nb-1],a[shell_nb],b[shell_nb],c[shell_nb]).
    :type Nptc: int array
    :param shell_nb: shell number of interest
    :type shell_nb: int
    :return: points drawn uniformly from an ellipsoidal shell volume, shell number
    :rtype: (N_reals,3) floats, 1 int"""
    
    # Part 0: Interpolation functions for deformation parameters
    x = np.arange(a.shape[0]+1)/(a.shape[0])
    a_inter = interp1d(x, np.hstack((0.0, a)))
    b_inter = interp1d(x, np.hstack((0.0, b)))
    c_inter = interp1d(x, np.hstack((0.0, c)))

    # Part 1: Generating uniform random vectors inside the 3-ball
    a_val = a[shell_nb]
    c_val = c[shell_nb]
    N_reals = Nptc[shell_nb]
    X_tmp = np.zeros((N_reals, dims))
    Z = np.zeros((N_reals, dims))
    np.random.seed(shell_nb)
    for i in range(N_reals):
        X_tmp[i] = np.random.normal(0,1,dims)
        R = (np.random.uniform(0,1,1))**(1/dims)
        Z[i] = R*X_tmp[i]/np.linalg.norm(X_tmp[i])
    
    # Part 2: Lower Cholesky decomposition
    if c_val == None:
        Sigma = np.array([[1/a_val**2, 0],[0,1/a_val**2]])
    else:
        Sigma = np.array([[1/a_val**2, 0,0],[0,1/a_val**2,0],[0,0,1/a_val**2]])
    L = cholesky(Sigma, lower=True)
    
    # Part 3: Draw uniform random vectors from Ball
    X = np.zeros((N_reals, dims))
    inv_ = inv(L.T)
    for pt in range(N_reals):
        X[pt] = np.dot(inv_,Z[pt])
        
    # Part 4: Move particles from Ball into Ellipsoidal Shell
    def transformCartToSpher(xyz):
        ptsnew = np.zeros(xyz.shape)
        xy = xyz[:,0]**2 + xyz[:,1]**2
        ptsnew[:,0] = np.sqrt(xy + xyz[:,2]**2)
        ptsnew[:,1] = np.arctan2(np.sqrt(xy), xyz[:,2]) # For elevation angle defined from Z-axis down
        ptsnew[:,2] = np.arctan2(xyz[:,1], xyz[:,0])
        return ptsnew
    
    X_sph = transformCartToSpher(X)
    rand = np.random.uniform(x[shell_nb], x[shell_nb+1], size = X.shape[0])
    a_target = a_inter(rand)
    b_target = b_inter(rand)
    c_target = c_inter(rand)
    x_new = np.sin(X_sph[:,1])*np.cos(X_sph[:,2])*a_target
    y_new = np.sin(X_sph[:,1])*np.sin(X_sph[:,2])*b_target
    z_new = np.cos(X_sph[:,1])*c_target
    Y = np.hstack((np.reshape(x_new, (x_new.shape[0],1)), np.reshape(y_new, (y_new.shape[0],1)), np.reshape(z_new, (z_new.shape[0],1))))
    # All ptcs pt in Y fulfill inShell(Y[pt], a, b, c, shell_nb) == True.
    return Y

def respectPBCNoRef(xyz, L_BOX = None):
    """
    Return modified positions xyz_out of an object that respect the box periodicity
    
    If point distro xyz has particles separated in any Cartesian direction
    by more than L_BOX/2, translate those particles accordingly.
    
    :param xyz: coordinates of particles
    :type xyz: (N^3x3) floats
    :param L_BOX: periodicity of box (0.0 if non-periodic)
    :type L_BOX: float
    :return: updated coordinates of particles
    :rtype: (N^3x3) floats"""
    if L_BOX != 0.0:
        xyz_out = xyz.copy() # Otherwise changes would be reflected in outer scope (np.array is mutable).
        rng = default_rng(seed=0) # Reference particle does not matter, i.e. ref = 0 is an option, but it is better to average over some random particles
        choose = rng.choice(np.arange(len(xyz)), (min(50,len(xyz)),), replace = False)
        ref_xyz = np.average(xyz_out[choose], axis = 0)
        dist_x = xyz_out[:,0]-ref_xyz[0]
        dist_y = xyz_out[:,1]-ref_xyz[1]
        dist_z = xyz_out[:,2]-ref_xyz[2]
        xyz_out[:,0][dist_x > L_BOX/2] = xyz_out[:,0][dist_x > L_BOX/2]-L_BOX
        xyz_out[:,0][dist_x < -L_BOX/2] = xyz_out[:,0][dist_x < -L_BOX/2]+L_BOX
        xyz_out[:,1][dist_y > L_BOX/2] = xyz_out[:,1][dist_y > L_BOX/2]-L_BOX
        xyz_out[:,1][dist_y < -L_BOX/2] = xyz_out[:,1][dist_y < -L_BOX/2]+L_BOX
        xyz_out[:,2][dist_z > L_BOX/2] = xyz_out[:,2][dist_z > L_BOX/2]-L_BOX
        xyz_out[:,2][dist_z < -L_BOX/2] = xyz_out[:,2][dist_z < -L_BOX/2]+L_BOX
        return xyz_out
    else:
        return xyz

def calcCoM(xyz, masses):
    """ Calculate center of mass of point distribution
    
    :param xyz: coordinates of particles of type 1 or type 4
    :type xyz: (N,3) floats
    :param masses: masses of the particles
    :type masses: (N,3) floats
    :return: com, center of mass
    :rtype: (3,) floats"""
    # Average over some random particles and recentre with respect to that to avoid large numbers
    rng = default_rng(seed=0)
    choose = rng.choice(np.arange(len(xyz)), (min(50,len(xyz)),), replace = False)
    ref_xyz = np.average(xyz[choose], axis = 0)
    delta_xyz = xyz.copy()-ref_xyz
    mass_total = np.sum(masses)
    com = np.sum(np.reshape(masses, (len(masses),1))*delta_xyz/mass_total, axis = 0)
    com = com+ref_xyz
    return com

def recentreObject(xyz, L_BOX):
    """ Recentre object if fallen outside [L_BOX]^3 due to e.g. respectPBCNoRef()
    
    :param xyz: coordinates of particles of type 1 or type 4
    :type xyz: (N,3) floats
    :param L_BOX: periodicity of box (0.0 if non-periodic)
    :type L_BOX: float
    :return: updated coordinates of particles
    :rtype: (N^3x3) floats"""
    xyz_out = xyz.copy()
    xyz_out[:,0][xyz_out[:,0] >= L_BOX] = xyz_out[:,0][xyz_out[:,0] >= L_BOX]-L_BOX
    xyz_out[:,0][xyz_out[:,0] < 0.0] = xyz_out[:,0][xyz_out[:,0] < 0.0]+L_BOX
    xyz_out[:,1][xyz_out[:,1] >= L_BOX] = xyz_out[:,1][xyz_out[:,1] >= L_BOX]-L_BOX
    xyz_out[:,1][xyz_out[:,1] < 0.0] = xyz_out[:,1][xyz_out[:,1] < 0.0]+L_BOX
    xyz_out[:,2][xyz_out[:,2] >= L_BOX] = xyz_out[:,2][xyz_out[:,2] >= L_BOX]-L_BOX
    xyz_out[:,2][xyz_out[:,2] < 0.0] = xyz_out[:,2][xyz_out[:,2] < 0.0]+L_BOX
    return xyz_out
    
def getCatWithinFracR200(cat_in, obj_size_in, xyz, masses, L_BOX, CENTER, r200, frac_r200):
    """ Cleanse index catalogue ``cat_in`` of particles beyond R200 ``r200``
    
    :param cat_in: contains indices of particles belonging to an object
    :type cat_in: (N3) integers
    :param obj_size_in: indicates how many particles are in each object
    :type obj_size_in: (N1,) integers
    :param xyz: coordinates of particles of type 1 or type 4
    :type xyz: (N2,3) floats
    :param masses: masses of the particles
    :type masses: (N2,3) floats
    :param L_BOX: periodicity of the box
    :type L_BOX: float
    :param CENTER: shape quantities will be calculated with respect to CENTER = 'mode' (point of highest density)
            or 'com' (center of mass) of each halo
    :type CENTER: str
    :param r200: R_200 radii of the parent halos
    :type r200: (N1,) floats
    :param frac_r200: depth of objects to plot ellipticity, in units of R200
    :type frac_r200: float
    :return cat_out, obj_size_out: updated cat_in and obj_size_out, with particles
        beyond R200 ``r200`` removed
    :rtype: list of length N1"""
    cat_out = np.empty(0, dtype = np.int32)
    obj_size_out = np.zeros((len(obj_size_in),), dtype = np.int32)
    centers = np.zeros((len(obj_size_in),3), dtype = np.float64)
    for idx in range(len(obj_size_in)): # Calculate centers of objects
        xyz_ = respectPBCNoRef(xyz[cat_in[np.sum(obj_size_in[:idx]):np.sum(obj_size_in[:idx+1])]], L_BOX)
        if CENTER == 'mode':
            centers.base[idx] = calcMode(xyz_, masses[cat_in[np.sum(obj_size_in[:idx]):np.sum(obj_size_in[:idx+1])]], max((max(xyz_[:,0])-min(xyz_[:,0]), max(xyz_[:,1])-min(xyz_[:,1]), max(xyz_[:,2])-min(xyz_[:,2]))))
        else:
            centers.base[idx] = calcCoM(xyz_, masses[cat_in[np.sum(obj_size_in[:idx]):np.sum(obj_size_in[:idx+1])]])
    remnant = []
    for idx in range(len(obj_size_in)):
        obj = cat_in[np.sum(obj_size_in[:idx]):np.sum(obj_size_in[:idx+1])]
        xyz_ = respectPBCNoRef(xyz[obj])
        tree = cKDTree(xyz_, leafsize=2, balanced_tree = False)
        all_nn_idxs = tree.query_ball_point(centers[idx], r=r200[idx]*frac_r200, n_jobs=-1)
        if all_nn_idxs != []:
            cat_out = np.hstack((cat_out, np.array(obj)[all_nn_idxs]))
            obj_size_out[idx] = len(all_nn_idxs)
            remnant.append(idx)            
    return cat_out, obj_size_out[remnant]

def isValidSelection(obj_numbers, nb_objects):
    """ Trivial function to check whether selection of objects is valid
    
    :param obj_numbers: list of object indices of interest
    :type obj_numbers: list of int
    :param nb_objects: number of objects in inventory
    :type nb_objects: integer
    :return valid: True if selection of objects is valid one, False otherwise
    :rtype: boolean
    :raises: ``ValueError`` if selection of objects is invalid"""
    # Check that obj_numbers has at least 1 object
    if len(obj_numbers) == 0:
        raise ValueError("Choose at least one halo for your object selection (obj_numbers). Please provide a list of halo indices.")
    # Check for non-integers
    if len(np.where(obj_numbers != obj_numbers.round())[0]) != 0:
        raise ValueError("Please supply integers only for your object selection!")
    # Check for negative integers
    if len(np.where(obj_numbers < 0)[0]) != 0:
        raise ValueError("No negative indices allowed in your object selection!")
    # Check for repeated non-negative integers
    if len(np.unique(obj_numbers)) != len(obj_numbers):
        raise ValueError("No repeated indices allowed in your object selection!")
    # Check for too large integers
    if np.max(obj_numbers) >= nb_objects:
        raise ValueError("Index / indices in your object selection too large. There aren't that many objects in the inventory.")
    return True

def getSubSetIdxCat(idx_cat, obj_size, obj_numbers):
    """ Get the indices from idx_cat that correspond to object numbers ``obj_numbers``
    
    :param idx_cat: contains indices of particles belonging to an object
    :type idx_cat: (N3,) integers
    :param obj_size: indicates how many particles are in each object
    :type obj_size: (N1,) integers
    :param obj_numbers: list of object indices of interest
    :type obj_numbers: list of int
    :return subset_idx_cat: indices from idx_cat that correspond to requested object numbers
    :rtype: (N4,) integers"""
    
    offsets = np.hstack((np.array([0]), np.cumsum(obj_size)))
    subset_idx_cat = np.empty((0,), np.int32)
    for p in obj_numbers:
        subset_idx_cat = np.hstack((subset_idx_cat, idx_cat[offsets[p]:offsets[p+1]]))
    return subset_idx_cat

def checkKatzConfig(katz_config):
    """ Check (for types etc) and return configuration parameters for Katz algorithm
    
    :param katz_config: dictionary with parameters to the Katz algorithm, with fields 'ROverR200', 'IT_TOL', 'IT_WALL', 'IT_MIN', 'REDUCED', 'SHELL_BASED'
    :type katz_config: dictionary
    :return ROverR200, IT_TOL, IT_WALL, IT_MIN, REDUCED, SHELL_BASED: configuration parameters
    :rtype: (r_res,) doubles, double, int, int, boolean, boolean"""
    ROverR200 = katz_config['ROverR200']
    IT_TOL = katz_config['IT_TOL']
    IT_WALL = katz_config['IT_WALL']
    IT_MIN = katz_config['IT_MIN']
    REDUCED = katz_config['REDUCED']
    SHELL_BASED = katz_config['SHELL_BASED']
    assert type(SHELL_BASED) == bool, "SHELL_BASED should be boolean"
    assert type(REDUCED) == bool, "REDUCED should be boolean"
    assert hasattr(ROverR200, "__len__"), "ROverR200 should be a list or array with more than one element" 
    ROverR200 = np.float64(ROverR200)
    IT_TOL = np.float64(IT_TOL)
    IT_WALL = np.int32(IT_WALL)
    IT_MIN = np.int32(IT_MIN)
    return ROverR200, IT_TOL, IT_WALL, IT_MIN, REDUCED, SHELL_BASED

def checkDensFitMethod(method):
    """ Check validity of density profile fitting method
    
    :param method: describes density profile model assumed for fitting, if parameter should be kept fixed during fitting then it needs to be provided, e.g. method['alpha'] = 0.18
    :type method: dictionary, method['profile'] is either `einasto`, `alpha_beta_gamma`, `hernquist`, `nfw`, minimum requirement"""
    assert type(method) == dict, "Note: method must be a dictionary"
    assert 'profile' in method, "Note: method must have at least the 'profile' field"
    assert method['profile'] == 'einasto' or method['profile'] == 'alpha_beta_gamma' or method['profile'] == 'hernquist' or method['profile'] == 'nfw', "Note: method['profile'] must be one of `einasto`, `alpha_beta_gamma`, `hernquist`, `nfw`"
    if method['profile'] == 'einasto':
        allowed_fields = {'profile', 'rho_s', 'alpha', 'r_s', 'min_method'}
        assert method.keys() <= allowed_fields, "Since you have chosen the Einasto profile, the only fields allowed for the method dict are 'rho_s', 'alpha', 'r_s'"
    elif method['profile'] == 'alpha_beta_gamma':
        allowed_fields = {'profile', 'rho_s', 'alpha', 'beta', 'gamma', 'r_s', 'min_method'}
        assert method.keys() <= allowed_fields, "Since you have chosen the generalized NFW profile (aka alpha-beta-gamma profile), the only fields allowed for the method dict are 'rho_s', 'alpha', 'beta', 'gamma', 'r_s'"
    else:
        allowed_fields = {'profile', 'rho_s', 'r_s', 'min_method'}
        assert method.keys() <= allowed_fields, "Since you have chosen the Hernquist or NFW profile, the only fields allowed for the method dict are 'rho_s', 'r_s'"
    if 'min_method' in method:
        assert method['min_method'] == 'Nelder-Mead' or method['min_method'] == 'L-BFGS-B' or method['min_method'] == 'TNC' or method['min_method'] == 'SLSQP' or method['min_method'] == 'Powell' or method['min_method'] == 'trust-constr', "Note: method['min_method'] must be one of `Nelder-Mead`, `L-BFGS-B`, `TNC`, `SLSQP`, `Powell`, and `trust-constr` methods"
    
default_katz_config = {'ROverR200': np.logspace(-1.5,0,70), 'IT_TOL': 1e-2, 'IT_WALL': 100, 'IT_MIN': 10, 'REDUCED': False, 'SHELL_BASED': False}