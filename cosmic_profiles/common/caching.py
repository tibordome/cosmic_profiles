#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import psutil
from functools import RLock, update_wrapper, namedtuple, wraps
import cosmic_profiles.common.config as config
import numpy as np

_CacheInfo = namedtuple("CacheInfo", ["hits", "misses", "maxsize", "currsize"])

class _HashedSeq(list):
    """ This class guarantees that hash() will be called no more than once
        per element.  This is important because the lru_cache() will hash
        the key multiple times on a cache miss.
    """

    __slots__ = 'hashvalue'

    def __init__(self, tup, hash=hash):
        self[:] = tup
        self.hashvalue = hash(tup)

    def __hash__(self):
        return self.hashvalue

def _make_key(args, kwds, typed,
             kwd_mark = (object(),),
             fasttypes = {int, str, frozenset, type(None)},
             sorted=sorted, tuple=tuple, type=type, len=len):
    """Make a cache key from optionally typed positional and keyword arguments
    
    The key is constructed in a way that is flat as possible rather than
    as a nested structure that would take more memory. If there is only a 
    single argument and its data type is known to cache
    its hash value, then that argument is returned without a wrapper.  This
    saves space and improves lookup speed.
    """
    key = args
    if kwds:
        sorted_items = sorted(kwds.items())
        key += kwd_mark
        for item in sorted_items:
            key += item
    if typed:
        key += tuple(type(v) for v in args)
        if kwds:
            key += tuple(type(v) for k, v in sorted_items)
    elif len(key) == 1 and type(key[0]) in fasttypes:
        return key[0]
    return _HashedSeq(key)

def lru_cache(maxsize=128, typed=False, use_memory_up_to=False):
    """Least-recently-used cache decorator.

    Arguments to the cached function must be hashable.
    View the cache statistics named tuple (hits, misses, maxsize, currsize)
    with f.cache_info().  Clear the cache and statistics with f.cache_clear().
    Access the underlying function with f.__wrapped__.
    See:  http://en.wikipedia.org/wiki/Cache_algorithms#Least_Recently_Used

    :param maxsize: if set to None, the LRU features are disabled and the cache
        can grow without bound
    :type maxsize: int
    :param typed: If True, arguments of different types will be cached separately.
        For example, f(3.0) and f(3) will be treated as distinct calls with
        distinct results.
    :param use_memory_up_to: represents the number of bytes of memory
        that must be available on the system in order for a value to be cached. If
        it is set (i.e. not ``False``), ``maxsize`` has no effect.
    :type use_memory_up_to: int, or float
    :return: decorator
    """
    if use_memory_up_to:
        maxsize=None

    # Users should only access the lru_cache through its public API:
    #       cache_info, cache_clear, and f.__wrapped__
    # The internals of the lru_cache are encapsulated for thread safety and
    # to allow the implementation to change (including a possible C version).

    # Constants shared by all lru cache instances:
    sentinel = object()          # unique object used to signal cache misses
    make_key = _make_key         # build a key from the function arguments
    PREV, NEXT, KEY, RESULT = 0, 1, 2, 3   # names for the link fields

    def decorating_function(user_function):
        cache = {}
        hits = misses = 0
        full = False
        cache_get = cache.get    # bound method to lookup a key or return None
        lock = RLock()           # because linkedlist updates aren't threadsafe
        root = []                # root of the circular doubly linked list
        root[:] = [root, root, None, None]     # initialize by pointing to self

        if use_memory_up_to:

            def wrapper(*args, **kwds):
                # Size limited caching that tracks accesses by recency
                nonlocal root, hits, misses, full
                key = make_key(args, kwds, typed)
                with lock:
                    link = cache_get(key)
                    if link is not None:
                        # Move the link to the front of the circular queue
                        link_prev, link_next, _key, result = link
                        link_prev[NEXT] = link_next
                        link_next[PREV] = link_prev
                        last = root[PREV]
                        last[NEXT] = root[PREV] = link
                        link[PREV] = last
                        link[NEXT] = root
                        hits += 1
                        return result
                result = user_function(*args, **kwds)
                with lock:
                    if key in cache:
                        # Getting here means that this same key was added to the
                        # cache while the lock was released.  Since the link
                        # update is already done, we need only return the
                        # computed result and update the count of misses.
                        pass
                    elif full:
                        # Use the old root to store the new key and result.
                        oldroot = root
                        oldroot[KEY] = key
                        oldroot[RESULT] = result
                        # Empty the oldest link and make it the new root.
                        # Keep a reference to the old key and old result to
                        # prevent their ref counts from going to zero during the
                        # update. That will prevent potentially arbitrary object
                        # clean-up code (i.e. __del__) from running while we're
                        # still adjusting the links.
                        root = oldroot[NEXT]
                        oldkey = root[KEY]
                        oldresult = root[RESULT]
                        root[KEY] = root[RESULT] = None
                        # Now update the cache dictionary.
                        del cache[oldkey]
                        # Save the potentially reentrant cache[key] assignment
                        # for last, after the root and links have been put in
                        # a consistent state.
                        cache[key] = oldroot
                    else:
                        # Put result in a new link at the front of the queue.
                        last = root[PREV]
                        link = [last, root, key, result]
                        last[NEXT] = root[PREV] = cache[key] = link
                        full = (psutil.virtual_memory().available
                                < use_memory_up_to)
                    misses += 1
                return result

        elif maxsize == 0:

            def wrapper(*args, **kwds):
                # No caching -- just a statistics update after a successful call
                nonlocal misses
                result = user_function(*args, **kwds)
                misses += 1
                return result

        elif maxsize is None:

            def wrapper(*args, **kwds):
                # Simple caching without ordering or size limit
                nonlocal hits, misses
                key = make_key(args, kwds, typed)
                result = cache_get(key, sentinel)
                if result is not sentinel:
                    hits += 1
                    return result
                result = user_function(*args, **kwds)
                cache[key] = result
                misses += 1
                return result

        else:

            def wrapper(*args, **kwds):
                # Size limited caching that tracks accesses by recency
                nonlocal root, hits, misses, full
                key = make_key(args, kwds, typed)
                with lock:
                    link = cache_get(key)
                    if link is not None:
                        # Move the link to the front of the circular queue
                        link_prev, link_next, _key, result = link
                        link_prev[NEXT] = link_next
                        link_next[PREV] = link_prev
                        last = root[PREV]
                        last[NEXT] = root[PREV] = link
                        link[PREV] = last
                        link[NEXT] = root
                        hits += 1
                        return result
                result = user_function(*args, **kwds)
                with lock:
                    if key in cache:
                        # Getting here means that this same key was added to the
                        # cache while the lock was released.  Since the link
                        # update is already done, we need only return the
                        # computed result and update the count of misses.
                        pass
                    elif full:
                        # Use the old root to store the new key and result.
                        oldroot = root
                        oldroot[KEY] = key
                        oldroot[RESULT] = result
                        # Empty the oldest link and make it the new root.
                        # Keep a reference to the old key and old result to
                        # prevent their ref counts from going to zero during the
                        # update. That will prevent potentially arbitrary object
                        # clean-up code (i.e. __del__) from running while we're
                        # still adjusting the links.
                        root = oldroot[NEXT]
                        oldkey = root[KEY]
                        oldresult = root[RESULT]
                        root[KEY] = root[RESULT] = None
                        # Now update the cache dictionary.
                        del cache[oldkey]
                        # Save the potentially reentrant cache[key] assignment
                        # for last, after the root and links have been put in
                        # a consistent state.
                        cache[key] = oldroot
                    else:
                        # Put result in a new link at the front of the queue.
                        last = root[PREV]
                        link = [last, root, key, result]
                        last[NEXT] = root[PREV] = cache[key] = link
                        full = (len(cache) >= maxsize)
                    misses += 1
                return result

        def cache_info():
            """Report cache statistics"""
            with lock:
                return _CacheInfo(hits, misses, maxsize, len(cache))

        def cache_clear():
            """Clear the cache and cache statistics"""
            nonlocal hits, misses, full
            with lock:
                cache.clear()
                root[:] = [root, root, None, None]
                hits = misses = 0
                full = False

        wrapper.cache_info = cache_info
        wrapper.cache_clear = cache_clear
        return update_wrapper(wrapper, user_function)

    return decorating_function

def np_cache_factory(nb_arrs, nb_lists):
    """ Decorator for caching functions with numpy array and/or lists of lists arguments
    
    Note that the arguments of the to-be-decorated function have to be as follows: 
    ``f(np_arr1, np_arr2, ..., list_of_lists1, list_of_lists2, ..., regular_args)``.
    
    :param nb_arrs: number of arguments that are numpy arrays
    :type nb_arrs: int
    :param nb_lists: number of subsequent arguments that are lists of lists
    :type nb_lists: int
    :param nb_GBs: if only nb_GBs are left (according to psutil.virtual_memory().available),
        cache will be considered full
    :type nb_GBs: float
    :return: decorator
    :rtype: function"""
    def np_cache(function):
        
        GB = 1024**3 # in bytes
        @lru_cache(maxsize = config.CACHE_MAXSIZE, use_memory_up_to=(config.GBs * GB))
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
