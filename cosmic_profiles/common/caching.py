#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from functools import lru_cache, wraps
import numpy as np

def np_cache_factory(nb_arrs, nb_lists):
    """ Decorator for caching functions with numpy array and/or lists of lists arguments
    
    Note that the arguments of the to-be-decorated function have to be as follows: 
    ``f(np_arr1, np_arr2, ..., list_of_lists1, list_of_lists2, ..., regular_args)``.
    
    :param nb_arrs: number of arguments that are numpy arrays
    :type nb_arrs: int
    :param nb_lists: number of subsequent arguments that are lists of lists
    :type nb_lists: int
    :return: decorator
    :rtype: function"""
    
    def np_cache(function):
        @lru_cache(0)
        def cached_wrapper(*args_cached, dims):
            args_l = []
            for arg in args_cached:
                args_l.append(arg)
            for arr_nb in range(nb_arrs):
                args_l[arr_nb] = np.array(args_l[arr_nb]).reshape(dims[arr_nb])
            for list_nb in np.arange(nb_arrs, nb_arrs + nb_lists):
                args_l[list_nb] = [list(args_l[list_nb][run]) for run in range(len(args_l[list_nb]))]
            return function(*args_l)
    
        @wraps(function)
        def wrapper(*args):
            args_l = []
            dims = []
            for arg in args:
                args_l.append(arg)
            for arr_nb in range(nb_arrs):
                dims.append(args_l[arr_nb].shape)
                args_l[arr_nb] = tuple(args_l[arr_nb].flatten())
            for list_nb in np.arange(nb_arrs, nb_arrs + nb_lists):
                args_l[list_nb] = tuple([tuple(args_l[list_nb][run]) for run in range(len(args_l[list_nb]))])
            dims = tuple(dims)
            args_l = tuple(args_l)
            return cached_wrapper(*args_l, dims=dims)
    
        # Copy lru_cache attributes over too
        wrapper.cache_info = cached_wrapper.cache_info
        wrapper.cache_clear = cached_wrapper.cache_clear
    
        return wrapper
    return np_cache