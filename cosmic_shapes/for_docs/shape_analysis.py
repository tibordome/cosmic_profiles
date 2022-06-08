#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 17:39:15 2022
"""

from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
from math import isnan
from sklearn.utils import resample
import matplotlib
matplotlib.rcParams.update({'font.size': 13})
from python_helpers import eTo10, print_status
from scipy.spatial.transform import Rotation as R
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def findMode(xyz, masses, rad):
    """ Find mode of point distribution xyz
    
    :param xyz: coordinates of particles of type 1 or type 4
    :type xyz: (N^3x3) floats
    :param masses: masses of particles of type 1 or type 4
    :type masses: (N^3x1) floats
    :param rad: initial radius to consider away from COM of object
    :type rad: float
    :return: mode of point distribution
    :rtype: (3,) floats"""
    com = np.sum(xyz*np.reshape(masses, (masses.shape[0],1)), axis = 0)/masses.sum()
    distances_all = np.linalg.norm(xyz-com,axis=1)
    xyz_constrain = xyz[distances_all < rad]
    masses_constrain = masses[distances_all < rad]
    if xyz_constrain.shape[0] < 5: # If only < 5 particles left, return
        return com
    else:
        rad *= 0.83 # Reduce radius by 17 %
        return findMode(xyz_constrain, masses_constrain, rad)

def respectPBC(xyz, L_BOX):
    """
    Return positions xyz that respect the box periodicity
    
    If point distro xyz has particles separated in any Cartesian direction
    by more than L_BOX/2, reflect those particles along L_BOX/2
    
    :param xyz: coordinates of particles of type 1 or type 4
    :type xyz: (N^3x3) floats
    :param L_BOX: simulation box side length
    :type L_BOX: float, units: cMpc/h
    :return: updated coordinates of particles of type 1 or type 4
    :rtype: (N^3x3) floats"""
    xyz_out = xyz.copy() # Otherwise changes would be reflected in outer scope (np.array is mutable).
    ref = 0 # Reference particle does not matter
    dist_x = abs(xyz_out[ref, 0]-xyz_out[:,0])
    xyz_out[:,0][dist_x > L_BOX/2] = L_BOX-xyz_out[:,0][dist_x > L_BOX/2] # Reflect x-xyz_outition along L_BOX/2
    dist_y = abs(xyz_out[ref, 1]-xyz_out[:,1])
    xyz_out[:,1][dist_y > L_BOX/2] = L_BOX-xyz_out[:,1][dist_y > L_BOX/2] # Reflect y-xyz_outition along L_BOX/2
    dist_z = abs(xyz_out[ref, 2]-xyz_out[:,2])
    xyz_out[:,2][dist_z > L_BOX/2] = L_BOX-xyz_out[:,2][dist_z > L_BOX/2] # Reflect z-xyz_outition along L_BOX/2
    return xyz_out

def getEpsilon(cat, xyz, masses, L_BOX, angle=0.0):
    """ Calculate the complex ellipticity (z-projected)
    
    It is obtained from the shape tensor = centred (wrt mode) second mass moment tensor
    
    :param cat: catalogue of objects (halos/gxs)
    :type cat: list of lists of ints
    :param xyz: coordinates of particles of type 1 or type 4
    :type xyz: (N^3x3) floats
    :param masses: masses of particles of type 1 or type 4
    :type masses: (N^3x1) floats
    :param L_BOX: simulation box side length
    :type L_BOX: float, units: cMpc/h
    :param angle: rotation of objects around z-axis before ellipticity is calculated (z-projected)
    :type angle: float
    :return: complex ellipticity
    :rtype: complex scalar
    """
    if rank == 0:
        eps = []
        rot_matrix = R.from_rotvec(angle * np.array([0, 0, 1])).as_matrix()
        for obj in cat:
            if obj != []:
                xyz_ = respectPBC(xyz[obj], L_BOX)
                masses_ = masses[obj]
                mode = findMode(xyz_, masses_, max((max(xyz_[:,0])-min(xyz_[:,0]), max(xyz_[:,1])-min(xyz_[:,1]), max(xyz_[:,2])-min(xyz_[:,2]))))
                xyz_new = np.zeros((xyz_.shape[0],3))
                for i in range(xyz_new.shape[0]):
                    xyz_new[i] = np.dot(rot_matrix, xyz_[i]-mode)
                shape_tensor = np.sum((masses_)[:,np.newaxis,np.newaxis]*(np.matmul(xyz_new[:,:,np.newaxis],xyz_new[:,np.newaxis,:])),axis=0)/np.sum(masses_)
                qxx = shape_tensor[0,0]
                qyy = shape_tensor[1,1]
                qxy = shape_tensor[0,1]
                eps.append((qxx-qyy)/(qxx+qyy) + complex(0,1)*2*qxy/(qxx+qyy))
        eps = np.array(eps)
        return eps
    else:
        return None
    
def chiSquareToIMultiDim(x, y):  
    """ Calculate chi-square test for multidimensional categorical data
    
    Both x and y are multi-dimensional (D_BINS+1, 
    varying numbers of elements in each bin), binning into 1D-bins is done
    before handing over to chiSquareToICore()
    :param x: sample 1, multi-dimensional
    :type x: list of float lists
    :param y: sample 1, multi-dimensional
    :type y: list of float lists
    :return: chi-square, p-value, degrees of freedom
    :rtype: float, float, int"""
    assert type(x) == list and type(y) == list, \
        'x and y must be lists'
    assert len(x) == len(y), \
        'Length of x and y must match'
    nbins = 10 # Choosing 10 bins for q-,s-,T-,SS- and SP-range [0.0,1.0]
    occs = np.zeros((2,len(x)*nbins), dtype = np.int32)
    y_ax = np.linspace(0.0,1.0,nbins)
    for bin_ in range(len(x)):
        for pt in range(len(x[bin_])):
            closest_idx = (np.abs(y_ax - x[bin_][pt])).argmin() # Determine which point in y_ax is closest
            occs[0,nbins*bin_+closest_idx] += 1
        for pt in range(len(y[bin_])):
            closest_idx = (np.abs(y_ax - y[bin_][pt])).argmin() # Determine which point in y_ax is closest
            occs[1,nbins*bin_+closest_idx] += 1
    chi_square, p_chi_square, dof = chiSquareToICore(occs)
    return chi_square, p_chi_square, dof
    
def chiSquareToIShapeCurve(d_cdm, param_interest_cdm, d_fdm, param_interest_fdm, D_LOGSTART, D_LOGEND, D_BINS):
    """ Chi-square test for shape curves
    
    Hands over to ``chiSquareToIMultiDim()``
    
    :param d_cdm: array of ellipsoidal radii, sample 1
    :type d_cdm: float array
    :param param_interest_cdm: shape quantitity of interest, sample 1
    :type param_interest_cdm: float array
    :param d_fdm: array of ellipsoidal radii, sample 2
    :type d_fdm: float array
    :param param_interest_fdm: shape quantitity of interest, sample 2
    :type param_interest_fdm: float array
    :param D_LOGSTART: logarithm of minimum ellipsoidal radius of interest, in units of R200 of parent halo
    :type D_LOGSTART: int
    :param D_LOGEND: logarithm of maximum ellipsoidal radius of interest, in units of R200 of parent halo
    :type D_LOGEND: int
    :param D_BINS: number of ellipsoidal radii of interest minus 1 (i.e. number of bins)
    :type D_BINS: int
    :return: chi-square, p-value, degrees of freedom
    :rtype: float, float, int"""
    R = np.logspace(D_LOGSTART,D_LOGEND,D_BINS+1)
    y_cdm = [[] for i in range(D_BINS+1)]
    for obj in range(len(param_interest_cdm)):
        for rad in range(D_BINS+1):
            closest_idx = (np.abs(R - d_cdm[obj][rad]/d_cdm[obj][-int(D_LOGEND/((D_LOGEND-D_LOGSTART)/D_BINS))-1])).argmin() # Determine which point in R is closest
            if isnan(param_interest_cdm[obj][rad]) or np.log10(d_cdm[obj][rad]/d_cdm[obj][-int(D_LOGEND/((D_LOGEND-D_LOGSTART)/D_BINS))-1]) > D_LOGEND:
                continue
            else:
                y_cdm[closest_idx].append(param_interest_cdm[obj][rad])
    y_fdm = [[] for i in range(D_BINS+1)]
    for obj in range(len(param_interest_fdm)):
        for rad in range(D_BINS+1):
            closest_idx = (np.abs(R - d_fdm[obj][rad]/d_fdm[obj][-int(D_LOGEND/((D_LOGEND-D_LOGSTART)/D_BINS))-1])).argmin() # Determine which point in R is closest
            if isnan(param_interest_fdm[obj][rad]) or np.log10(d_fdm[obj][rad]/d_fdm[obj][-int(D_LOGEND/((D_LOGEND-D_LOGSTART)/D_BINS))-1]) > D_LOGEND:
                continue
            else:
                y_fdm[closest_idx].append(param_interest_fdm[obj][rad])
    chi_square, p_chi_square, dof = chiSquareToIMultiDim(y_cdm, y_fdm)
    return chi_square, p_chi_square, dof
    
def chiSquareToICore(occs): # https://libguides.library.kent.edu/spss/chisquare
    """ Chi Square Test of Independence for Binned Data
    
    Get chi square and p-value, assuming occs[0,:]
    and occs[1,:] are drawn from the same distribution.
    Q we could ask: Is the PDF independent of mass?
    
    :param occs: spine of shaded curves, but 
        rescaled to be cell counts (!or weighted cell counts!), not PDF
    :type occs: (2 = # mass groups, # mass or sep bins) floats
    :return: chi-square, p-value, degrees of freedom
    :rtype: float, float, int"""
    if (occs == 0.0).all():
        return np.nan, np.nan, np.nan
    chi_square = 0.0
    for group in range(occs.shape[0]): 
        for d_bin in range(occs.shape[1]):
            expect = occs[group].sum()*occs[:,d_bin].sum()/occs.flatten().sum()
            if expect != 0.0:
                chi_square += (occs[group, d_bin] - expect)**2/expect
    dof = (occs.shape[1]-1)*(occs.shape[0]-1)
    p_chi_square = 1 - stats.chi2.cdf(chi_square, dof) # p value for chi_square and dof dof
    return chi_square, p_chi_square, dof
    
def KSTest(x,y):
    """ 2-Sample KS Test of Independence for Continuous Distributions
    
    Get KS test statistic and p-value, assuming x
    and y are drawn from the same distribution.
    Q we could ask: Is the PDF independent of mass?
    
    :param x: 1st sample, all sample points, not binned,
        whether normalized or not is irrelevant
    :type x: (# sample values) floats, can be both np.array and list
    :param y: 2nd sample, all sample points, not binned,
        whether normalized or not is irrelevant
    :type y: (# sample values) floats
    :return: KS-stat, p-value
    :rtype: float, float"""
    if len(list(x)) == 0 or len(list(y)) == 0:
        return np.nan, np.nan
    ks_stats = stats.ks_2samp(x, y)
    return ks_stats[0], ks_stats[1]  

            
def getMeanOrMedianAndError(y, ERROR_METHOD, N_REAL=10):
    """Return mean (if ERROR_METHOD == "bootstrap" or "SEM") or median
    (if ERROR_METHOD == "median_quantile") and the +- 1 sigma error attached
    
    :param y: data
    :type y: 1D float array
    :param ERROR_METHOD: error method, either 'bootstrap', 'SEM' or 'median_quantile'
    :type ERROR_METHOD: string
    :param N_REAL: number of bootstraps, only relevant if ``ERROR_METHOD`` == 'bootstrap'
    :type N_REAL: int
    :return: mean_median, err_low, err_high
    :rtype: float, float, float"""
    if ERROR_METHOD == "bootstrap":
        mean_median = np.array([np.average(z) if z != [] else np.nan for z in y])
        mean_l = [[] for i in range(len(y))]
        err_low = np.empty(len(y), dtype = np.float32)
        err_high = np.empty(len(y), dtype = np.float32)
        for random_state in range(N_REAL):
            for d_bin in range(len(y)):
                boot = resample(y[d_bin], replace=True, n_samples=len(y[d_bin]), random_state=random_state)
                mean_l[d_bin].append(np.average(boot))
        for d_bin in range(len(y)):
            err_low[d_bin] = np.std(mean_l[d_bin], ddof=1) # Says thestatsgeek.com
            err_high[d_bin] = err_low[d_bin]
    elif ERROR_METHOD == "SEM":
        mean_median = np.array([np.average(z) if z != [] else np.nan for z in y])
        err_low = np.array([np.std(z, ddof=1)/(np.sqrt(len(z))) for z in y])
        err_high = err_low
    else:
        assert ERROR_METHOD == "median_quantile"
        mean_median = np.array([np.median(z) if z != [] else np.nan for z in y])
        err_low = np.array([np.quantile(np.array(z), 0.25)/(np.sqrt(len(z))) if z != [] else np.nan for z in y])
        err_high = np.array([np.quantile(np.array(z), 0.75)/(np.sqrt(len(z))) if z != [] else np.nan for z in y])
    return mean_median, err_low, err_high
            
def getShape(R, d, param_interest, ERROR_METHOD, D_LOGSTART, D_LOGEND, D_BINS):
    """ Get average profile for param_interest (which is defined at all values of d)
    at all elliptical radii R
    
    :param R: elliptical radii of interest
    :type R: (N,) floats
    :param d: param_interest is defined at all elliptical radii d
    :type d: (N2,) floats
    :param_interest: the quantity of interest defined at all elliptical radii d
    :param param_interest: (N2,) floats
    :return: mean, err_low, err_high
    :rtype: float, float, float"""
    y = [[] for i in range(D_BINS+1)]
    for obj in range(param_interest.shape[0]):
        for rad in range(D_BINS+1):
            closest_idx = (np.abs(R - d[obj,rad]/d[obj,-int(D_LOGEND/((D_LOGEND-D_LOGSTART)/D_BINS))-1])).argmin() # Determine which point in R is closest
            if isnan(param_interest[obj][rad]) or np.log10(d[obj][rad]/d[obj,-int(D_LOGEND/((D_LOGEND-D_LOGSTART)/D_BINS))-1]) > D_LOGEND:
                continue
            else:
                y[closest_idx].append(param_interest[obj][rad])
    mean, err_low, err_high = getMeanOrMedianAndError(y, ERROR_METHOD)
    return mean, err_low, err_high

def getShapeMs(R, d, idx_groups, group, param_interest, ERROR_METHOD, D_LOGSTART, D_LOGEND, D_BINS):
    """ Similar to getShape, but with mass-splitting"""
    y = [[] for i in range(D_BINS+1)]
    for obj in idx_groups[group]:
        for rad in range(D_BINS+1):
            closest_idx = (np.abs(R - d[obj,rad]/d[obj,-int(D_LOGEND/((D_LOGEND-D_LOGSTART)/D_BINS))-1])).argmin() # Determine which point in R is closest
            if isnan(param_interest[obj][rad]) or np.log10(d[obj][rad]/d[obj,-int(D_LOGEND/((D_LOGEND-D_LOGSTART)/D_BINS))-1]) > D_LOGEND:
                continue
            else:
                y[closest_idx].append(param_interest[obj][rad])
    mean, err_low, err_high = getMeanOrMedianAndError(y, "median_quantile")
    return mean, err_low, err_high
            
def getBlocks(lst, n):
    """ Yield successive ``n``-sized blocks from lst (list or np.array).
    
    :param lst: array-like object to draw successive blocks from
    :type lst: list or np.array
    :param n: size of blocks, last block might be smaller
    :type n: int
    :return: blocks of size ``n``
    :rtype: generator function"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def M_split(m, center, start_time, v = None, M_SPLIT_TYPE = "const_occ", TWO_SPLIT = 9, NB_BINS = 2):
    """Mass-splitting of quantities center, v
    
    Mass units are determined by ``m``
    
    :param m: total mass of objects
    :type m: list of floats
    :param center: centers of objects
    :type center: list of (3,) float arrays
    :param start_time: time of start of shape analysis
    :type start_time: float
    :param M_SPLIT_TYPE: either "log_slice", where masses in log space are split, 
        or "const_occ", where masses are split ensuring equal number of points in each bin
        out of the ``NB_BINS`` bins, or "fixed_bins",
        where bins will be 10^7 to 10^8, 10^8 to 10^9 etc.
    :type M_SPLIT_TYPE: string
    :param TWO_SPLIT: In case of "2_fixed_bins", around which mass to split (10** thereof)
    :type TWO_SPLIT: float
    :param NB_BINS: In case of "const_occ" and "log_slice", number of bins
    :type NB_BINS: float
    :param v: major axis of objects (optional) or any other vectorial quantity
    :type v: list of (3,) float arrays
    :return: max_min_m (mass bin edges), m_groups (total mass in bins), center_groups (center average in each bin), 
        v_groups (v average in each bin), idx_groups (indices of all objects in each bin)
    :rtype: (N,) floats, (N-1,) floats, (N-1,) floats, (N-1,) floats, list of lists (for each bin) of ints
    """
    
    if v is not None:
        vdim = v.ndim
    args_sort = np.argsort(m)
    m_ordered = np.sort(m)
    max_min_m = []
    max_min_mlog = []
    m_groups = []
    gx_center_groups = []
    v_groups = []
    if len(m) % NB_BINS == 0:
        chunk_size = len(m)//NB_BINS
    else:
        chunk_size = len(m)//NB_BINS + 1
    if M_SPLIT_TYPE == "const_occ":
        if chunk_size == 0:
            idx_groups = []
            if v == None:
                return max_min_m, m_groups, gx_center_groups, idx_groups
            return max_min_m, m_groups, gx_center_groups, v_groups, idx_groups 
        m_groups = list(getBlocks(list(m_ordered), chunk_size)) # List of lists
        gx_center_groups = list(getBlocks(center[args_sort], chunk_size)) # List of arrays
        if v is not None:
            v_groups = list(getBlocks(v[args_sort], chunk_size)) # List of arrays
        print_status(rank, start_time, "The mass bins (except maybe last) have size {0}".format(chunk_size))
        print_status(rank, start_time, "The number of mass bins is {0}".format(len(m_groups)))
        for i in range(len(m_groups)+1):
            if i == len(m_groups):
                max_min_m.append(np.float32(m_ordered[-1]))
            else:
                max_min_m.append(np.float32(m_ordered[i*chunk_size]))
    elif M_SPLIT_TYPE == "log_slice":
        log_m_min = np.log10(m.min())
        log_m_max = np.log10(m.max())
        delta_m = (log_m_max - log_m_min)/(NB_BINS)
        for i in range(NB_BINS + 1):
            if i == NB_BINS:
                max_min_m.append(np.float32(10**(log_m_max)))
                max_min_mlog.append(np.float32(log_m_max))
            else:
                max_min_m.append(np.float32(10**(log_m_min+delta_m*i)))
                max_min_mlog.append(np.float32(log_m_min+delta_m*i))
        split_occ = np.zeros((NB_BINS,))
        for m in range(len(m_ordered)):
            for split in range(NB_BINS):
                if np.log10(m_ordered[m]) >= max_min_mlog[split] and np.log10(m_ordered[m]) <= max_min_mlog[split+1]:
                    split_occ[split] += 1
        for split in range(NB_BINS):
            m_groups.append([np.float32(m_ordered[i]) for i in np.arange(int(np.array([split_occ[j] for j in range(split)]).sum()), int(np.array([split_occ[j] for j in range(split+1)]).sum()))])
            center_add = np.array([np.float32(center[args_sort][i]) for i in np.arange(int(np.array([split_occ[j] for j in range(split)]).sum()), int(np.array([split_occ[j] for j in range(split+1)]).sum()))])
            gx_center_groups.append(np.float32(np.reshape(center_add, (center_add.shape[0], 3))))
            if v is not None:
                v_add = np.array([np.float32(v[args_sort][i]) for i in np.arange(int(np.array([split_occ[j] for j in range(split)]).sum()), int(np.array([split_occ[j] for j in range(split+1)]).sum()))])
                if vdim == 1:
                    v_groups.append(np.float32(np.reshape(v_add, (v_add.shape[0],))))
                else:
                    assert vdim == 2
                    v_groups.append(np.float32(np.reshape(v_add, (v_add.shape[0], 3))))
        print_status(rank, start_time, "Split occupancies: {0}".format(split_occ))
    elif M_SPLIT_TYPE == "fixed_bins":
        log_m_min = 7
        log_m_max = 15
        delta_m = 1
        for i in range(9):
            if i == 8:
                max_min_m.append(np.float32(10**(log_m_max)))
                max_min_mlog.append(np.float32(log_m_max))
            else:
                max_min_m.append(np.float32(10**(log_m_min+delta_m*i)))
                max_min_mlog.append(np.float32(log_m_min+delta_m*i))
        split_occ = np.zeros((8,))
        for m in range(len(m_ordered)):
            for split in range(8):
                if np.log10(m_ordered[m]) >= max_min_mlog[split] and np.log10(m_ordered[m]) <= max_min_mlog[split+1]:
                    split_occ[split] += 1
        for split in range(8):
            m_groups.append([np.float32(m_ordered[i]) for i in np.arange(int(np.array([split_occ[j] for j in range(split)]).sum()), int(np.array([split_occ[j] for j in range(split+1)]).sum()))])
            center_add = np.array([np.float32(center[args_sort][i]) for i in np.arange(int(np.array([split_occ[j] for j in range(split)]).sum()), int(np.array([split_occ[j] for j in range(split+1)]).sum()))])
            gx_center_groups.append(np.float32(np.reshape(center_add, (center_add.shape[0], 3))))
            if v is not None:
                v_add = np.array([np.float32(v[args_sort][i]) for i in np.arange(int(np.array([split_occ[j] for j in range(split)]).sum()), int(np.array([split_occ[j] for j in range(split+1)]).sum()))])
                if vdim == 1:
                    v_groups.append(np.float32(np.reshape(v_add, (v_add.shape[0],))))
                else:
                    assert vdim == 2
                    v_groups.append(np.float32(np.reshape(v_add, (v_add.shape[0], 3))))
        print_status(rank, start_time, "Split occupancies: {0}".format(split_occ))
    else:
        assert M_SPLIT_TYPE == "2_fixed_bins"
        max_min_m.append(np.float32(10**(7)))
        max_min_mlog.append(np.float32(7))
        max_min_m.append(np.float32(10**(TWO_SPLIT)))
        max_min_mlog.append(np.float32(TWO_SPLIT))
        max_min_m.append(np.float32(10**(15)))
        max_min_mlog.append(np.float32(15))
        split_occ = np.zeros((2,))
        for m in range(len(m_ordered)):
            for split in range(2):
                if np.log10(m_ordered[m]) >= max_min_mlog[split] and np.log10(m_ordered[m]) <= max_min_mlog[split+1]:
                    split_occ[split] += 1
        for split in range(2):
            m_groups.append([np.float32(m_ordered[i]) for i in np.arange(int(np.array([split_occ[j] for j in range(split)]).sum()), int(np.array([split_occ[j] for j in range(split+1)]).sum()))])
            center_add = np.array([np.float32(center[args_sort][i]) for i in np.arange(int(np.array([split_occ[j] for j in range(split)]).sum()), int(np.array([split_occ[j] for j in range(split+1)]).sum()))])
            gx_center_groups.append(np.float32(np.reshape(center_add, (center_add.shape[0], 3))))
            if v is not None:
                v_add = np.array([np.float32(v[args_sort][i]) for i in np.arange(int(np.array([split_occ[j] for j in range(split)]).sum()), int(np.array([split_occ[j] for j in range(split+1)]).sum()))])
                if vdim == 1:
                    v_groups.append(np.float32(np.reshape(v_add, (v_add.shape[0],))))
                else:
                    assert vdim == 2
                    v_groups.append(np.float32(np.reshape(v_add, (v_add.shape[0], 3))))
        print_status(rank, start_time, "Split occupancies: {0}".format(split_occ))
        
    idx_groups = [[args_sort[i] for i in np.arange(int(np.array([len(m_groups[k]) for k in range(j)]).sum()), int(np.array([len(m_groups[k]) for k in range(j)]).sum())+len(m_groups[j]))] for j in range(len(m_groups))]
    if v_groups == []:
        assert v == None
        return max_min_m, m_groups, gx_center_groups, idx_groups
    return max_min_m, m_groups, gx_center_groups, v_groups, idx_groups

def readShapeData(CAT_DEST, SNAP, D_BINS, local, suffix):
    """ Read in all relevant shape-related data
    
    :param CAT_DEST: catalogue destination
    :type CAT_DEST: string
    :param SNAP: e.g. '024'
    :type SNAP: string
    :param D_BINS: number of ellipsoidal radii of interest minus 1 (i.e. number of bins)
    :type D_BINS: int
    :param local: whether to read in local or global shape data
    :type local: boolean
    :param suffix: either '_dm_' or '_gx_' or '' (latter for CosmicShapesDirect)
    :type suffix: string
    :return: obj_masses, obj_centers, d, q, s, major_full
    :rtype: 1D float array, 1D float array, 1D float array, 3x ((number_of_objs,) float array or (number_of_objs, D_BINS+1) float array), 
        (number_of_objs,3) float array or (number_of_objs, D_BINS+1, 3) float array"""
    
    d = np.loadtxt('{0}/d_{1}{2}{3}.txt'.format(CAT_DEST, "local" if local == True else "global", suffix, SNAP)) # Has shape (number_of_objs, D_BINS+1)
    q = np.loadtxt('{0}/q_{1}{2}{3}.txt'.format(CAT_DEST, "local" if local == True else "global", suffix, SNAP))
    s = np.loadtxt('{0}/s_{1}{2}{3}.txt'.format(CAT_DEST, "local" if local == True else "global", suffix, SNAP))
    if local == False:
        d = np.array(d, ndmin=1)
        d = d.reshape(d.shape[0], 1) # Has shape (number_of_objs, 1)
        q = np.array(q, ndmin=1)
        q = q.reshape(q.shape[0], 1) # Has shape (number_of_objs, 1)
        s = np.array(s, ndmin=1)
        s = s.reshape(s.shape[0], 1) # Has shape (number_of_objs, 1)
    else:
        # Dealing with the case of 1 obj
        if d.ndim == 1 and d.shape[0] == D_BINS+1:
            d = d.reshape(1, D_BINS+1)
            q = q.reshape(1, D_BINS+1)
            s = s.reshape(1, D_BINS+1)
    major_full = np.loadtxt('{0}/major_{1}{2}{3}.txt'.format(CAT_DEST, "local" if local == True else "global", suffix, SNAP))
    if major_full.ndim == 2:
        major_full = major_full.reshape(major_full.shape[0], major_full.shape[1]//3, 3) # Has shape (number_of_objs, D_BINS+1, 3)
    else:
        if local == True:
            if major_full.shape[0] == (D_BINS+1)*3:
                major_full = major_full.reshape(1, D_BINS+1, 3)
        else:
            if major_full.shape[0] == 3:
                major_full = major_full.reshape(1, 1, 3)
    obj_masses = np.loadtxt('{0}/m_{1}{2}{3}.txt'.format(CAT_DEST, "local" if local == True else "global", suffix, SNAP)) # Has shape (number_of_hs,)
    obj_centers = np.loadtxt('{0}/centers_{1}{2}{3}.txt'.format(CAT_DEST, "local" if local == True else "global", suffix, SNAP)) # Has shape (number_of_hs,3)
    return obj_masses, obj_centers, d, q, s, major_full

def getShapeCurves(CAT_DEST, VIZ_DEST, SNAP, D_LOGSTART, D_LOGEND, D_BINS, start_time, MASS_UNIT=1e10, suffix = '_'):
    """
    Create a series of plots to analyze object shapes
    
    Plot intertial tensor axis ratios, triaxialities and ellipticity histograms.
    
    :param CAT_DEST: catalogue destination
    :type CAT_DEST: string
    :param VIZ_DEST: visualisation folder destination
    :type VIZ_DEST: string
    :param SNAP: e.g. '024'
    :type SNAP: string
    :param D_LOGSTART: logarithm of minimum ellipsoidal radius of interest, in units of R200 of parent halo
    :type D_LOGSTART: int
    :param D_LOGEND: logarithm of maximum ellipsoidal radius of interest, in units of R200 of parent halo
    :type D_LOGEND: int
    :param D_BINS: number of ellipsoidal radii of interest minus 1 (i.e. number of bins)
    :type D_BINS: int
    :param start_time: time of start of shape analysis
    :type start_time: float
    :param MASS_UNIT: conversion factor from previous mass unit to M_sun/h
    :type MASS_UNIT: float
    :param suffix: either '_dm_' or '_gx_' or '' (latter for CosmicShapesDirect)
    :type suffix: string"""
    
    print_status(rank,start_time,'Starting getShapeCurves() with snap {0}'.format(SNAP))
    
    if rank == 0:
        # Reading
        try:
            obj_masses, obj_centers, d, q, s, major_full = readShapeData(CAT_DEST, SNAP, D_BINS, True, suffix)
        except OSError: # Components for snap are not available 
            print_status(rank,start_time,'Calling readShapeData() for snap {0} threw OSError. Skip rest'.format(SNAP))
            return None
        print_status(rank, start_time, "The number of objects considered is {0}".format(d.shape[0]))
        
        # Mass splitting
        max_min_m, obj_m_groups, obj_center_groups, idx_groups = M_split(MASS_UNIT*obj_masses, obj_centers, start_time)
            
        # Maximal elliptical radii
        R = np.logspace(D_LOGSTART,D_LOGEND,D_BINS+1)
        ERROR_METHOD = "median_quantile"
        
        # Q
        plt.figure()
        mean_median, err_low, err_high = getShape(R, d, q, ERROR_METHOD, D_LOGSTART, D_LOGEND, D_BINS)
        plt.semilogx(R, mean_median)
        plt.fill_between(R, mean_median-err_low, mean_median+err_high, edgecolor='g', alpha = 0.5)
        # Formatting
        plt.xlabel(r"$r/R_{200}$")
        plt.ylabel(r"q")
        plt.ylim(0.0, 1.0)
        plt.savefig("{0}/q{1}{2}.pdf".format(VIZ_DEST, suffix, SNAP), bbox_inches="tight")
        
        # S
        plt.figure()
        mean_median, err_low, err_high = getShape(R, d, s, ERROR_METHOD, D_LOGSTART, D_LOGEND, D_BINS)
        plt.semilogx(R, mean_median)
        plt.fill_between(R, mean_median-err_low, mean_median+err_high, edgecolor='g', alpha = 0.5)
        # Formatting
        plt.xlabel(r"$r/R_{200}$")
        plt.ylabel(r"s")
        plt.ylim(0.0, 1.0)
        plt.savefig("{0}/s{1}{2}.pdf".format(VIZ_DEST, suffix, SNAP), bbox_inches="tight")
        
        # T
        plt.figure()
        if q.ndim == 2:
            T = np.zeros((q.shape[0], q.shape[1]))
            for obj in range(q.shape[0]):
                T[obj] = (1-q[obj]**2)/(1-s[obj]**2) # Triaxiality
        else:
            T = np.empty(0)
        mean_median, err_low, err_high = getShape(R, d, T, ERROR_METHOD, D_LOGSTART, D_LOGEND, D_BINS)
        plt.semilogx(R, mean_median)
        plt.fill_between(R, mean_median-err_low, mean_median+err_high, edgecolor='g', alpha = 0.5)
        
        # Formatting
        plt.xlabel(r"$r/R_{200}$")
        plt.ylabel(r"T")
        plt.axhline(2/3, label=r"$T$ > 2/3: prolate", linestyle='--', color = "y")
        plt.legend(loc="upper right", fontsize="x-small")
        plt.ylim(0.0, 1.0)
        plt.savefig("{0}/T{1}{2}.pdf".format(VIZ_DEST, suffix, SNAP), bbox_inches="tight")
        
        # Q: M-splitting
        for group in range(len(obj_m_groups)):
            plt.figure()
            mean_median, err_low, err_high = getShapeMs(R, d, idx_groups, group, q, ERROR_METHOD, D_LOGSTART, D_LOGEND, D_BINS)
            if len(idx_groups[group]) != 0:
                plt.semilogx(R, mean_median)
                plt.fill_between(R, mean_median-err_low, mean_median+err_high, label = r"$M: {0} - {1} \ M_{{\odot}}/h$".format(eTo10("{:.2E}".format(max_min_m[group])), eTo10("{:.2E}".format(max_min_m[group+1]))), alpha = 0.5)
            plt.legend(loc="upper right", fontsize="x-small")
            plt.xlabel(r"$r/R_{200}$")
            plt.ylabel(r"q")
            plt.ylim(0.0, 1.0)
            plt.savefig("{0}/qM{1}{2}{3}.pdf".format(VIZ_DEST, int(np.log10(max_min_m[group])), suffix, SNAP), bbox_inches="tight")
        
        # S: M-splitting
        for group in range(len(obj_m_groups)):
            plt.figure()
            mean_median, err_low, err_high = getShapeMs(R, d, idx_groups, group, s, ERROR_METHOD, D_LOGSTART, D_LOGEND, D_BINS)
            if len(idx_groups[group]) != 0:
                plt.semilogx(R, mean_median)
                plt.fill_between(R, mean_median-err_low, mean_median+err_high, label = r"$M: {0} - {1} \ M_{{\odot}}/h$".format(eTo10("{:.2E}".format(max_min_m[group])), eTo10("{:.2E}".format(max_min_m[group+1]))), alpha = 0.5)
            plt.legend(loc="upper right", fontsize="x-small")
            plt.xlabel(r"$r/R_{200}$")
            plt.ylabel(r"s")
            plt.ylim(0.0, 1.0)
            plt.savefig("{0}/sM{1}{2}{3}.pdf".format(VIZ_DEST, int(np.log10(max_min_m[group])), suffix, SNAP), bbox_inches="tight")
        
        # T: M-splitting
        for group in range(len(obj_m_groups)):
            plt.figure()
            mean_median, err_low, err_high = getShapeMs(R, d, idx_groups, group, T, ERROR_METHOD, D_LOGSTART, D_LOGEND, D_BINS)
            if len(idx_groups[group]) != 0:
                plt.semilogx(R, mean_median)
                plt.fill_between(R, mean_median-err_low, mean_median+err_high, label = r"$M: {0} - {1} \ M_{{\odot}}/h$".format(eTo10("{:.2E}".format(max_min_m[group])), eTo10("{:.2E}".format(max_min_m[group+1]))), alpha = 0.5)
            plt.axhline(2/3, label=r"$T$ > 2/3: prolate", linestyle='--', color = "y")
            plt.legend(loc="upper right", fontsize="x-small")            
            plt.xlabel(r"$r/R_{200}$")
            plt.ylabel(r"T")
            plt.ylim(0.0, 1.0)
            plt.savefig("{0}/TM{1}{2}{3}.pdf".format(VIZ_DEST, int(np.log10(max_min_m[group])), suffix, SNAP), bbox_inches="tight")
    
    
def getLocalTHisto(CAT_DEST, VIZ_DEST, SNAP, D_LOGSTART, D_LOGEND, D_BINS, start_time, HIST_NB_BINS, MASS_UNIT=1e10, suffix = '_', inner = False):
    """ Plot triaxiality T histogram
    
    :param CAT_DEST: catalogue destination
    :type CAT_DEST: string
    :param VIZ_DEST: visualisation folder destination
    :type VIZ_DEST: string
    :param SNAP: e.g. '024'
    :type SNAP: string
    :param D_LOGSTART: logarithm of minimum ellipsoidal radius of interest, in units of R200 of parent halo
    :type D_LOGSTART: int
    :param D_LOGEND: logarithm of maximum ellipsoidal radius of interest, in units of R200 of parent halo
    :type D_LOGEND: int
    :param D_BINS: number of ellipsoidal radii of interest minus 1 (i.e. number of bins)
    :type D_BINS: int
    :param start_time: time of start of shape analysis
    :type start_time: float
    :param HIST_NB_BINS: Number of histogram bins
    :type HIST_NB_BINS: int
    :param MASS_UNIT: conversion factor from previous mass unit to M_sun/h
    :type MASS_UNIT: float
    :param suffix: either '_dm_' or '_gx_' or '' (latter for CosmicShapesDirect)
    :type suffix: string
    :param inner: whether T histogram is to be plotted for R200 (False) or at R200*0.15 (True, e.g. 
        Milky Way radius is about R200*0.15)
    :type inner: boolean
    """
    
    if rank == 0:
        # Read & Assemble
        try:
            obj_masses, obj_centers, d, q, s, major_full = readShapeData(CAT_DEST, SNAP, D_BINS, True, suffix)
        except OSError: # Components for snap are not available 
            print_status(rank,start_time,'Calling readShapeData() for snap {0} threw OSError. Skip rest'.format(SNAP))
            return None
        print_status(rank, start_time, "The number of objects considered is {0}".format(d.shape[0]))
        
        idx = np.zeros((d.shape[0],), dtype = np.int32)
        for obj in range(idx.shape[0]):
            if inner == True:
                idx[obj] = np.argmin(abs(d[obj] - d[obj,-int(D_LOGEND/((D_LOGEND-D_LOGSTART)/D_BINS))-1]*0.15))
            else:
                idx = np.array([np.int32(x) for x in list(np.ones((d.shape[0],))*(-1))])
        
        t = np.zeros((d.shape[0],))
        for obj in range(d.shape[0]):
            t[obj] = (1-q[obj,idx[obj]]**2)/(1-s[obj,idx[obj]]**2) # Triaxiality
        t = np.nan_to_num(t)
            
        # T counting
        plt.figure()
        t[t == 0.] = np.nan
        n, bins, patches = plt.hist(x=t, bins = np.linspace(0, 1, HIST_NB_BINS), alpha=0.7, density=True)
        plt.axvline(1/3, label="oblate-triaxial transition", color = "g")
        plt.axvline(2/3, label="triaxial-prolate transition", color = "r")
        plt.xlabel(r"T")
        plt.ylabel('Normalized Bin Count')
        plt.grid(axis='y', alpha=0.75)
        plt.xlim(0.0, 1.0)
        plt.legend(loc="upper left", fontsize="x-small")
        plt.savefig("{0}/TCount{1}{2}.pdf".format(VIZ_DEST, suffix, SNAP), bbox_inches="tight")
        
        t = t[np.logical_not(np.isnan(t))]
        print_status(rank, start_time, "In degrees: The average T value for the objects is {0} and the standard deviation (assuming T is Gaussian distributed) is {1}".format(round(np.average(t),2), round(np.std(t),2)))
     
def getGlobalEpsHisto(cat, xyz, masses, L_BOX, VIZ_DEST, SNAP, suffix = '_', HIST_NB_BINS = 11):
    """ Plot ellipticity histogram
    
    :param cat: catalogue of objects (objs/gxs)
    :type cat: list of lists of ints
    :param xyz: coordinates of particles of type 1 or type 4
    :type xyz: (N^3x3) floats
    :param masses: masses of particles of type 1 or type 4
    :type masses: (N^3x1) floats
    :param L_BOX: simulation box side length
    :type L_BOX: float, units: cMpc/h
    :param VIZ_DEST: visualisation folder destination
    :type VIZ_DEST: string
    :param SNAP: e.g. '024'
    :type SNAP: string
    :param suffix: either '_dm_' or '_gx_' or '' (latter for CosmicShapesDirect)
    :type suffix: string
    :param HIST_NB_BINS: Number of histogram bins
    :type HIST_NB_BINS: int"""
    
    if rank == 0:
        eps = getEpsilon(cat, xyz, masses, L_BOX)
        plt.figure()
        n, bins, patches = plt.hist(x=abs(eps), bins = np.linspace(0, 1, HIST_NB_BINS), alpha=0.7, density=True)
        plt.xlabel(r"$\epsilon$")
        plt.ylabel('Normalized Bin Count')
        plt.grid(axis='y', alpha=0.75)
        plt.xlim(0.0, 1.0)
        plt.savefig("{0}/EpsCount{1}{2}.pdf".format(VIZ_DEST, suffix, SNAP), bbox_inches="tight")
