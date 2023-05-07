Unit System and Caching
========================

**************************
Unit System
**************************

To avoid ambiguities in the units that are provided by the user vs the units that are returned by public methods such as ``estDensProfs()``, it is best to explicitly declare the 'incoming' and 'outgoing' unit system::

    from cosmic_profiles import updateInUnitSystem, updateOutUnitSystem
    
    # Unit System
    updateInUnitSystem(length_in_cm = 3.085678e21, mass_in_g = 1.989e43, velocity_in_cm_per_s = 1e5, little_h = 0.6774)
    updateOutUnitSystem(length_in_cm = 3.085678e24, mass_in_g = 1.989e33, velocity_in_cm_per_s = 1e5, little_h = 0.6774)

Here, the combination of ``length_in_cm``, ``mass_in_g`` and ``velocity_in_cm_per_s`` determines the unit system of the data provided by the user. For instance, the combination above is the unit sytem that is often used in cosmological simulations. By analogy, the combination of ``length_in_cm``, ``mass_in_g`` and ``velocity_in_cm_per_s`` fully determines the unit system of the CosmicProfiles outputs (such as density profiles).

.. warning:: When handing over density profiles as arguments to ``fitDensProfs()``, ``estConcentrations()`` or ``plotDensProfs()``, please make sure the units are as determined by the 'outgoing' unit system. This is important when a user employs the functionalities ``fitDensProfs()``, ``estConcentrations()`` or ``plotDensProfs()`` directly, with density profiles obtained in some other way.

**************************
Caching
**************************

As explained in the :ref:`Caching Routines section<Caching Routines>` in more detail, caching for various internal functions is implemented by CosmicProfiles to avoid recalculating / reloading repeatedly. The cache will be considered full if there are fewer than ``use_memory_up_to`` bytes of RAM available. 
In other words, it's the minimum size of free RAM (i.e. not occupied by cached objects). The user can update ``use_memory_up_to`` by calling ``updateCachingMaxGBs()``::

    from cosmic_profiles import updateCachingMaxGBs
    
    # Caching Settings
    use_memory_up_to = 2 # In GBs
    updateCachingMaxGBs(use_memory_up_to)

On systems with a small RAM, CosmicProfiles will cache fewer objects, and calculations will be repeated more often. In case the user sets ``use_memory_up_to`` to ``False``, the ``maxsize`` argument will take effect, which can be updated by calling ``updateCachingMaxSize()``::

    from cosmic_profiles import updateCachingMaxSize
    
    # Caching Settings
    maxsize = 128 # New maximum cache size
    updateCachingMaxSize(maxsize)

This will result in standard, memory-agnostic LRU-caching as implemented with the ``lru_cache`` decorator. ``maxsize`` specifies the maximum number of items that can be stored in the cache. 
When the cache reaches its maximum size, the least recently used (LRU) item is evicted to make room for a new item. ``maxsize`` is set to 128 by default, but it can be set to any non-negative integer or ``None`` to indicate an unbounded cache size.
Setting ``use_memory_up_to`` to ``False`` and ``maxsize`` to zero might make sense if you want to ensure CosmicProfiles uses the least amount of RAM possible (i.e. this disables caching altogether).