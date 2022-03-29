#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 17:39:15 2022

@author: tibor
"""

import time
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.linalg import cholesky
from numpy.linalg import inv

def print_status(rank,start_time,message,**kwargs):
    """ Routine to print script status to command line, with elapsed time
    
    For optional ``color`` argument, we allow for 'green', 'blue' and 'red'."""
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
    in [solar masses/(cMpc)^3*h^2] units"""
    G = 6.674*10**(-29)/(3.086*10**16)**3*h**3 # in (cMpc/h)^3/(kg*s^2)
    H_z = 100*h/(3.086*10**19) # in 1/s
    solar_mass = 2*10**30 # in kg
    return 3*H_z**2/(8*np.pi*G)/solar_mass*h

def getMassDMParticle(N, L, h):
    """Returns the mass of each DM particle,
    assuming it is constant, given N and L
    Arguments:
    -------------
    N: Simulation resolution parameter, power of 2
    L: Simulation box size in cMpc/h
    Returns:
    -------------
    Mass of each DM particle in solar masses*h^-1"""
    return getRhoCrit(h)*(L/N)**3

def getDelta(z, OMEGA_M, OMEGA_L):
    """ Calculates the overdensity required for spherical virialization relative to the mean BG of the universe
    
    From Bryan, Norman, 1998, ApJ (Vir. Theorem + Spher. Collapse). Note that this quantity is z-dependent."""
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
    """ Creating "evenly" distributed points on a unit sphere"""

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
    """ Creating "evenly" distributed points on an ellipsoid's surface"""

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
    # Step 2: Rubinstein & Kroese 2007
    # Part 1: Generating Uniform Random Vectors inside the 3-ball
    
    print("The major axis is", a, "the intermediate one is", b, "and the minor axis is", c)
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

def projectIntoOuterShell3D_biased(X, a, b, c, delta_r):
    Y = np.zeros((X.shape[0], X.shape[1]))
    for i in range(X.shape[0]):
        Y[i,0] = delta_r/a*X[i,0]+(a-delta_r)*X[i,0]/np.linalg.norm(X[i,:])
        Y[i,1] = delta_r/a*X[i,1]+(a-delta_r)*X[i,1]/np.linalg.norm(X[i,:])*b/a
        Y[i,2] = delta_r/a*X[i,2]+(a-delta_r)*X[i,2]/np.linalg.norm(X[i,:])*c/a
    return Y


def projectIntoOuterShell3D(X, a, b, c, delta_a):
    Y = np.zeros((X.shape[0], X.shape[1]))
    for i in range(X.shape[0]):
        uni = np.random.uniform(0.0,1.0)
        Y[i,0] = (a-delta_a)*X[i,0]/np.linalg.norm(X[i,:])+delta_a*uni*X[i,0]/np.linalg.norm(X[i,:])
        Y[i,1] = (a-delta_a)*X[i,1]/np.linalg.norm(X[i,:])*b/a+delta_a*uni*X[i,1]/np.linalg.norm(X[i,:])*b/a # delta_r introduces some level of error here
        Y[i,2] = (a-delta_a)*X[i,2]/np.linalg.norm(X[i,:])*c/a+delta_a*uni*X[i,2]/np.linalg.norm(X[i,:])*c/a # delta_r introduces some level of error here
    return Y

def projectIntoOuterShell2D(X, a, b, delta_r):
    Y = np.zeros((X.shape[0], X.shape[1]))
    for i in range(X.shape[0]):
        Y[i,0] = delta_r/a*X[i,0]+(a-delta_r)*X[i,0]/np.linalg.norm(X[i,:])
        Y[i,1] = delta_r/a*X[i,1]+(a-delta_r)*X[i,1]/np.linalg.norm(X[i,:])*b/a
    return Y
 

