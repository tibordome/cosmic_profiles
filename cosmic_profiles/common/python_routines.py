#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy as np
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
    return np.float32(DELTA)

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

def respectPBCNoRef(xyz, L_BOX):
    """
    Return positions xyz that respect the box periodicity
    
    If point distro xyz has particles separated in any Cartesian direction
    by more than L_BOX/2, reflect those particles along L_BOX/2
    
    :param xyz: coordinates of particles of type 1 or type 4
    :type xyz: (N,3) floats
    :param L_BOX: periodicity of the box
    :type L_BOX: float
    :return: updated coordinates of particles of type 1 or type 4
    :rtype: (N,3) floats"""
    xyz_out = xyz.copy() # Otherwise changes would be reflected in outer scope (np.array is mutable).
    ref = 0 # Reference particle does not matter
    dist_x = abs(xyz_out[ref, 0]-xyz_out[:,0])
    xyz_out[:,0][dist_x > L_BOX/2] = L_BOX-xyz_out[:,0][dist_x > L_BOX/2] # Reflect x-xyz_outition along L_BOX/2
    dist_y = abs(xyz_out[ref, 1]-xyz_out[:,1])
    xyz_out[:,1][dist_y > L_BOX/2] = L_BOX-xyz_out[:,1][dist_y > L_BOX/2] # Reflect y-xyz_outition along L_BOX/2
    dist_z = abs(xyz_out[ref, 2]-xyz_out[:,2])
    xyz_out[:,2][dist_z > L_BOX/2] = L_BOX-xyz_out[:,2][dist_z > L_BOX/2] # Reflect z-xyz_outition along L_BOX/2
    return xyz_out

def calcCoM(xyz, masses):
    """ Calculate center of mass of point distribution
    
    :param xyz: coordinates of particles of type 1 or type 4
    :type xyz: (N,3) floats
    :param masses: masses of the particles
    :type masses: (N,3) floats
    :return: com, center of mass
    :rtype: (3,) floats"""
    com = np.zeros((3,), dtype = np.float32)
    mass_total = 0.0
    for run in range(xyz.shape[0]):
        mass_total += masses[run]
    for run in range(xyz.shape[0]):
        com[0] += masses[run]*xyz[run,0]/mass_total
        com[1] += masses[run]*xyz[run,1]/mass_total
        com[2] += masses[run]*xyz[run,2]/mass_total
    return com

def getCatWithinFracR200(cat_in, xyz, masses, L_BOX, CENTER, r200, frac_r200):
    """ Cleanse index catalogue ``cat_in`` of particles beyond R200 ``r200``
    
    :param cat_in: each entry of the list is a list containing indices of particles belonging to an object
    :type cat_in: list of length N1, N1 << N2
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
    :return cat_out: updated cat_in, with particles beyond R200 ``r200`` removed
    :rtype: list of length N1"""
    cat_out = [[] for i in range(len(cat_in))]
    idxs_compr = np.zeros((len(cat_in),), dtype = np.int32)
    h_pass = np.array([1 if x != [] else 0 for x in cat_in])
    idxs_compr[h_pass.nonzero()[0]] = np.arange(np.sum(h_pass))
    centers = np.zeros((len(cat_in),3), dtype = np.float32)
    for idx in range(len(cat_in)): # Calculate centers of objects
        if cat_in[idx] != []:
            xyz_ = respectPBCNoRef(xyz[cat_in[idxs_compr[idx]]], L_BOX)
            if CENTER == 'mode':
                centers.base[idx] = calcMode(xyz_, masses[cat_in[idxs_compr[idx]]], max((max(xyz_[:,0])-min(xyz_[:,0]), max(xyz_[:,1])-min(xyz_[:,1]), max(xyz_[:,2])-min(xyz_[:,2]))))
            else:
                centers.base[idx] = calcCoM(xyz_, masses[cat_in[idxs_compr[idx]]])
    for idx, obj in enumerate(cat_in):
        if obj != []:
            xyz_ = respectPBCNoRef(xyz[obj])
            tree = cKDTree(xyz_, leafsize=2, balanced_tree = False)
            all_nn_idxs = tree.query_ball_point(centers[idxs_compr[idx]], r=r200[idx]*frac_r200, n_jobs=-1)
            if all_nn_idxs != []:
                cat_out[idx] = list(np.array(obj)[all_nn_idxs])
    return cat_out