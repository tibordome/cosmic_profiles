#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 17:39:15 2022
"""

import time
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.linalg import cholesky
from numpy.linalg import inv

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
            
def getRhoCrit(h):
    """ Returns critical comoving density of the universe
    
    Return value is in units of [solar masses/(cMpc)^3*h^2]
    
    :param h: little H
    :type h: float
    :return: critical comoving density
    :rtype: float"""
    G = 6.674*10**(-29)/(3.086*10**16)**3*h**3 # in (cMpc/h)^3/(kg*s^2)
    H_z = 100*h/(3.086*10**19) # in 1/s
    solar_mass = 2*10**30 # in kg
    return 3*H_z**2/(8*np.pi*G)/solar_mass*h

def getMassDMParticle(N, L, h):
    """Returns the mass of each DM particle,
    assuming it is constant, given N and L
    
    :param N: Simulation resolution parameter, usually power of 2
    :type N: int
    :param L: Simulation box size in cMpc/h
    :type L: float
    :return: Mass of each DM particle in solar masses*h^-1
    :rtype: float"""
    return getRhoCrit(h)*(L/N)**3

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
    spheres and cubes as cubes.  Required since `ax.axis('equal')`
    and `ax.set_aspect('equal')` don't work on 3D.
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

def drawUniformFromEllipsoid(N_reals, dims, a, b, c=None):
    """ Draw points uniformly from an ellipsoid volume
    
    This function is primarily used to generate synthetic halos.
    The approach is taken from Rubinstein & Kroese 2007:\n
    Part 1: Generating uniform random vectors inside the 3-ball\n
    Part 2: Lower Cholesky decomposition\n
    Part 3: Draw uniformly from ellipsoid
    
    :param N_reals: number of points to draw from ellipsoid volume
    :type N_reals: int
    :param dims: number of dimensions of 'ellipsoid', either 2,
        in which case c remains `None`, or 3
    :type dims: int
    :type a: float
    :param b: intermediate axis of ellipsoid surface
    :type b: float
    :param c: optional, minor axis of ellipsoid surface
    :type c: float
    :return: points drawn uniformly from an ellipsoid volume
    :rtype: (N_reals,3) floats"""
    
    # Part 1: Generating uniform random vectors inside the 3-ball
    X_tmp = np.zeros((N_reals, dims))
    Z = np.zeros((N_reals, dims))
    for i in range(N_reals):
        X_tmp[i] = np.random.normal(0,1,dims)
        R = (np.random.uniform(0,1,1))**(1/dims)
        Z[i] = R*X_tmp[i]/np.linalg.norm(X_tmp[i])
    
    # Part 2: lower Cholesky decomposition
    if c == None:
        Sigma = np.array([[1/a**2, 0],[0,1/b**2]])
    else:
        Sigma = np.array([[1/a**2, 0,0],[0,1/b**2,0],[0,0,1/c**2]])
    L = cholesky(Sigma, lower=True)
    
    # Part 3: Uniform from Ellipsoid
    X = np.zeros((N_reals, dims))
    for i in range(N_reals):
        X[i] = np.dot(inv(L.T),Z[i])
    return X