#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 17:39:15 2022
"""

from scipy import stats
import numpy as np
from math import isnan
from sklearn.utils import resample
import matplotlib
matplotlib.rcParams.update({'font.size': 13})
from cosmic_profiles.common.python_routines import print_status
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def calcMode(xyz, masses, rad):
    """ Find mode of point distribution xyz
    
    :param xyz: coordinates of particles of type 1 or type 4
    :type xyz: (N,3) floats
    :param masses: masses of particles of type 1 or type 4
    :type masses: (N,) floats
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
        return calcMode(xyz_constrain, masses_constrain, rad)

def respectPBC(xyz, L_BOX):
    """
    Return positions xyz that respect the box periodicity
    
    If point distro xyz has particles separated in any Cartesian direction
    by more than L_BOX/2, reflect those particles along L_BOX/2
    
    :param xyz: coordinates of particles of type 1 or type 4
    :type xyz: (N,3) floats
    :param L_BOX: simulation box side length
    :type L_BOX: float, units: Mpc/h
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
    :param v: major axis of objects (optional) or any other vectorial quantity
    :type v: list of (3,) float arrays
    :param M_SPLIT_TYPE: either "log_slice", where masses in log space are split, 
        or "const_occ", where masses are split ensuring equal number of points in each bin
        out of the ``NB_BINS`` bins, or "fixed_bins",
        where bins will be 10^7 to 10^8, 10^8 to 10^9 etc.
    :type M_SPLIT_TYPE: string
    :param TWO_SPLIT: In case of "2_fixed_bins", around which mass to split (10** thereof)
    :type TWO_SPLIT: float
    :param NB_BINS: In case of "const_occ" and "log_slice", number of bins
    :type NB_BINS: float
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
