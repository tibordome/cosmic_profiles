|Logo|

CosmicProfiles is a Cython package for Point Cloud Profiling

|Documentation Status| |PyPI| |Build Status|

The CosmicProfiles project
****************************

This repository provides shape and density profile analysis tools for cosmological simulations (and beyond). Its features include

- overall halo shape determination, i.e. major, intermediate, minor axis vectors and shape quantities such as intermediate-to-major axis ratio or sphericity
- halo shape profile determination

  - iterative shell-based shape profile determination algorithm for high-resolution halos
  - iterative ellipsoid-based shape profile determination algorithm for lower-resolution halos
  - user can choose between reduced shape tensor and non-reduced shape tensor
- works with

  - 'direct' datasets (i.e. index catalogue provided by user) and
  - Gadget-style HDF5 snapshot files
  
    - additional velocity dispersion tensor eigenaxes determination
    - galaxy density and shape profile determination also works out of the box
- halo density profile estimation using direct binning and kernel-based approaches

  - user can choose between direct binning into spherical shells and
  - direct binning into ellipsoidal shells
  
- density profile fitting assuming either `NFW <https://ui.adsabs.harvard.edu/abs/1997ApJ...490..493N/abstract>`_, `Hernquist 1990 <https://ui.adsabs.harvard.edu/abs/1990ApJ...356..359H/abstract>`_, `Einasto <https://ui.adsabs.harvard.edu/abs/1965TrAlm...5...87E/abstract>`_ or `alpha-beta-gamma <https://arxiv.org/abs/1107.5582>`_ profile model

  - concentration-mass relationship of halos easy to calculate
- mock halo generator: ellipsoidal or spherical, compatible with the 4 density profile models
- easy to interface with `pynbody` to work with halos identified in a cosmological simulation (see example scripts)
- easy to interface with `nbodykit` to harness large-scale structure capabilities (see example scripts)
- 3D point cloud plotting tools
- efficient caching capabilities to accelerate look-ups

The documentation can be found `here <https://cosmic-profiles.readthedocs.io/en/latest/index.html>`_.

Contributions are welcome. Please raise an issue or open a PR.


.. |PyPI| image:: https://badge.fury.io/py/cosmic_profiles.svg
   :target: https://badge.fury.io/py/cosmic_profiles
.. |Logo| image:: https://cosmic-profiles.readthedocs.io/en/latest/_images/CProfiles.png
   :target: https://github.com/tibordome/cosmic_profiles
   :width: 400
.. |Documentation Status| image:: https://readthedocs.org/projects/cosmic-profiles/badge/?version=latest
   :target: https://cosmic-profiles.readthedocs.io/en/latest/?badge=latest
.. |Build status| image:: https://app.travis-ci.com/tibordome/cosmic_profiles.svg?branch=master
   :target: https://app.travis-ci.com/tibordome/cosmic_profiles
