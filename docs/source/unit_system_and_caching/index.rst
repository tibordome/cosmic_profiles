Unit System and Caching
========================

**************************
Unit System
**************************

To avoid ambiguities in the units that are provided by the user vs the units that are returned by public methods such as ``estDensProfs()``, it is best to explicitly declare the 'incoming' and 'outgoing' unit system::

    from cosmic_profiles import updateInUnitSystem, updateOutUnitSystem
    
    # Unit System
    updateInUnitSystem(in_unit_length_in_cm = 3.085678e21, in_unit_mass_in_g = 1.989e43, in_unit_velocity_in_cm_per_s = 1e5)
    updateOutUnitSystem(out_unit_length_in_cm = 3.085678e24, out_unit_mass_in_g = 1.989e33, out_unit_velocity_in_cm_per_s = 1e5)

Here, the combination of ``in_unit_length_in_cm``, ``in_unit_mass_in_g`` and ``in_unit_velocity_in_cm_per_s`` determines the unit system of the data provided by the user. For instance, the combination above is the unit sytem that is often used in cosmological simulations. By analogy, the combination of ``out_unit_length_in_cm``, ``out_unit_mass_in_g`` and ``out_unit_velocity_in_cm_per_s`` fully determines the unit system of the data provided by *Cosmic Profiles*.

.. note:: The plotting functionalities of *Cosmic Profiles* will disregard the 'outgoing' unit system. For instance, ``plotDensProfs()`` will always plot density profiles in units of M_sun*h^2/(Mpc)**3 vs normalized radius.

.. note:: When handing over density profiles as arguments to ``fitDensProfs()``, ``estConcentrations()`` or ``plotDensProfs()``, please make sure the units are as determined by the 'outgoing' unit system. This is important when a user employs the functionalities ``fitDensProfs()``, ``estConcentrations()`` or ``plotDensProfs()`` directly, with density profiles obtained in some other way.

**************************
Caching
**************************

As explained in the :ref:`Caching Routines section<Caching Routines>` in more detail, caching of function outputs is implemented by *Cosmic Profiles* to avoid recalculating / reloading repeatedly. The cache will be considered full if there are fewer than ``use_memory_up_to`` bytes of memory available. The user can update ``use_memory_up_to`` by calling ``updateCachingMaxGBs()``::

    from cosmic_profiles import updateCachingMaxGBs
    
    # Caching Settings
    use_memory_up_to = 2 # In GBs
    updateCachingMaxGBs(use_memory_up_to)

On systems with a small RAM, the user can increase ``use_memory_up_to`` to make *Cosmic Profiles* consume less RAM. In case the user sets ``use_memory_up_to`` to ``False``, the ``maxsize`` argument will take effect, which can be updated by calling ``updateCachingMaxSize()``::

    from cosmic_profiles import updateCachingMaxSize
    
    # Caching Settings
    maxsize = 128 # New maximum cache size
    updateCachingMaxSize(maxsize)

This will result in regular, memory-agnostic LRU-caching as implemented with the ``lru_cache`` decorator.
