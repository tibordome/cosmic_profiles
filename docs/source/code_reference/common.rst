.. _Cosmic Profiles Code Reference:

Common Functionalities
========================

The ``profile_classes`` module defines the classes ``CosmicProfilesDirect`` and ``CosmicProfilesGadgetHDF5`` that constitute the heart of the ``CosmicProfiles`` package, providing methods to perform shape and density profile estimation and fitting. 

.. automodule:: common.profile_classes
   :members:
   :undoc-members:
   :show-inheritance:

The ``cosmo_tools`` module provides simple Python tools that are useful for the analysis of cosmological datasets, such as KS tests, chi-square tests etc.

.. automodule:: common.cosmo_tools
   :members:
   :undoc-members:
   :show-inheritance:

The ``python_routines`` module helps in calculating the center of mass of a point cloud, drawing uniformly from an ellipsoid, drawing uniformly from an ellipsoidal shell, drawing Fibonacci sphere samples etc.

.. automodule:: common.python_routines
   :members:
   :undoc-members:
   :show-inheritance:

The ``caching`` module implements the caching of function outputs to avoid recalculating/reloading repeatedly. A modified decorator has been added that takes an additional optional argument `use_memory_up_to`. If set, the cache will be considered full if there are fewer than `use_memory_up_to bytes` of memory available (according to `psutil.virtual_memory().available`). Note that the `maxsize` argument will have no effect in that case. The motivation behind a memory-aware LRU caching function is that caching too many values causes thrashing, which should be avoided if possible.

.. automodule:: common.caching
   :members:
   :undoc-members:
   :show-inheritance:
