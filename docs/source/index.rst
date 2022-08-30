.. cosmic_profiles documentation master file, created by
   sphinx-quickstart on Wed Feb  2 22:15:33 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

The *Cosmic Profiles* project
=============================

|pic0|

.. |pic0| image:: ../../CProfiles.png
   :width: 50%

*Cosmic Profiles* provides shape and density profile analysis tools for cosmological simulations (and beyond). Its features include

- overall halo shape determination, i.e. major, intermediate, minor axis vectors and shape quantities such as intermediate-to-major axis ratio or sphericity
- halo shape profile determination

  - iterative shell-based shape profile determination algorithm for high-resolution halos
  - iterative ellipsoid-based shape profile determination algorithm for lower-resolution halos
  - user can choose between reduced shape tensor and non-reduced shape tensor
- supports
    
  - 'direct' datasets (i.e. index catalogue provided by user) and
  - *GADGET*-style FoF / SUBFIND HDF5 snapshot files

    - density and shape profile determination for dark matter halos, gas particle halos and star particle halos
    - velocity dispersion tensor eigenaxes determination
- halo density profile estimation using direct binning and kernel-based approaches
  - user can choose between direct binning into spherical shells and
  - direct binning into ellipsoidal shells
- density profile fitting assuming either NFW, Hernquist 1990, Einasto or :math:`\alpha \beta \gamma`-profile model
  - concentration-mass relationship of halos easy to calculate
- mock halo generator: ellipsoidal or spherical, compatible with the 4 density profile models
- easy to interface with `pynbody` to work with halos identified in a cosmological simulation (see example scripts)
- easy to interface with `nbodykit` to harness large-scale structure capabilities (see example scripts)
- various profile plotting and 3D point cloud plotting tools
- efficient caching capabilities to accelerate look-ups

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   support
   data_structures/index
   shape_estimation/index
   density_profiles/index
   mock_halo_generator/index
   unit_system_and_caching/index
   code_reference/index
   example_scripts/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
