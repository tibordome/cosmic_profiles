.. _Cosmic Profiles Code Reference:

Common Functionalities
=======================

Base Profiling Class
*********************

The ``cosmic_base_class`` module defines the ``CosmicBase`` class that constitutes the parent class of many profiling classes in the *CosmicProfiles* package.

.. automodule:: common.cosmic_base_class
   :members:
   :undoc-members:
   :show-inheritance:

Cosmo Tools
**************

The ``cosmo_tools`` module provides simple Python tools that are useful for the analysis of cosmological datasets, such as KS tests, chi-square tests etc.

.. automodule:: common.cosmo_tools
   :members:
   :undoc-members:
   :show-inheritance:

Python Routines
****************

The ``python_routines`` module helps in calculating the center of mass of a point cloud, drawing uniformly from an ellipsoid, drawing uniformly from an ellipsoidal shell, drawing Fibonacci sphere samples etc.

.. automodule:: common.python_routines
   :members:
   :undoc-members:
   :show-inheritance:

.. _Caching Routines:

Caching Routines
*****************

The ``caching`` module implements the caching of function outputs to avoid recalculating/reloading repeatedly. A modified decorator has been added that in addition to the standard ``maxsize`` argument takes the optional argument ``use_memory_up_to``. 
If set (i.e. ``use_memory_up_to`` is not ``False``), the cache will be considered full if there are fewer than ``use_memory_up_to`` free bytes of memory available (according to ``psutil.virtual_memory().available``). Note that the ``maxsize`` argument will have no effect in that case. 
The motivation behind a memory-aware LRU caching function is that caching too many values causes thrashing, which should be avoided if possible.

.. automodule:: common.caching
   :members:
   :undoc-members:
   :show-inheritance:

The user has the liberty to update ``use_memory_up_to`` and ``maxsize`` by calling ``updateCachingMaxGBs()`` or ``updateCachingMaxSize()``, respectively.

.. automodule:: common.config
   :members:
   :undoc-members:
   :show-inheritance:

The default caching parameters are

.. literalinclude :: ../../../cosmic_profiles/common/config.py
   :language: python
   :lines: 47

To disable caching, use::

    from cosmic_profiles import updateCachingMaxSize, updateCachingMaxGBs
    updateCachingMaxGBs(False)
    updateCachingMaxSize(None)
