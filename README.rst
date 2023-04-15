|Logo|

CosmicProfiles is a Cython package for Point Cloud Profiling

|Build Status| |Platforms|

The CosmicProfiles project
****************************

This repository provides shape and density profile analysis tools for cosmological simulations (and beyond). Its features include

- overall halo shape determination, i.e. major, intermediate, minor axis vectors and shape quantities such as intermediate-to-major axis ratio or sphericity
- halo shape profile determination

  - iterative shell-based shape profile determination algorithm for high-resolution halos
  - iterative ellipsoid-based shape profile determination algorithm for lower-resolution halos
  - user can choose between reduced shape tensor and non-reduced shape tensor
- supports

  - 'direct' datasets (i.e. index catalogue provided by user) and
  - FoF / SUBFIND halo catalogues
  - Gadget-style I, II and HDF5 snapshot files
  
    - all functionalities available for dark matter halos, gas particle halos and star particle halos
    - in addition, allows for velocity dispersion tensor eigenaxes determination
- halo density profile estimation using direct binning and kernel-based approaches

  - user can choose between direct binning into spherical shells and
  - direct binning into ellipsoidal shells
- density profile fitting assuming either `NFW <https://ui.adsabs.harvard.edu/abs/1997ApJ...490..493N/abstract>`_, `Hernquist 1990 <https://ui.adsabs.harvard.edu/abs/1990ApJ...356..359H/abstract>`_, `Einasto <https://ui.adsabs.harvard.edu/abs/1965TrAlm...5...87E/abstract>`_ or `alpha-beta-gamma <https://arxiv.org/abs/1107.5582>`_ profile model

  - concentration-mass relationship of halos easy to calculate
- mock halo generator: ellipsoidal or spherical, compatible with the 4 density profile models
- easy to interface with `pynbody` to work with halos identified in a cosmological simulation (see example scripts)
- easy to interface with `nbodykit` to harness large-scale structure capabilities (see example scripts)
- various profile plotting and 3D point cloud plotting tools
- efficient caching capabilities to accelerate look-ups

Documentation
****************************

|Documentation Status|

The documentation for CosmicProfiles is hosted on `Read the Docs
<https://cosmic-profiles.readthedocs.io/en/latest/>`__.

Installation and Dependencies
******************************

|PyPI| |Name| |Downloads| |Version|

The easiest way to get CosmicProfiles is to install it with conda using the conda-forge channel::

    conda install cosmic_profiles --channel conda-forge
    
Alternatively, you can use pip::

   pip install cosmic-profiles

See the `installation
instructions <https://cosmic-profiles.readthedocs.io/en/latest/support.html>`_ in the
`documentation <https://cosmic-profiles.readthedocs.io/en/latest/>`__ for more information.

License
****************************

|License|

Copyright 2020-2023 Tibor Dome.

CosmicProfiles is free software made available under the MIT License.

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
.. |Name| image:: https://img.shields.io/badge/recipe-cosmic_profiles-green.svg
   :target: https://anaconda.org/conda-forge/cosmic_profiles
.. |Downloads| image:: https://img.shields.io/conda/dn/conda-forge/cosmic_profiles.svg
   :target: https://anaconda.org/conda-forge/cosmic_profiles
.. |Version| image:: https://img.shields.io/conda/vn/conda-forge/cosmic_profiles.svg
   :target: https://anaconda.org/conda-forge/cosmic_profiles
.. |Platforms| image:: https://img.shields.io/conda/pn/conda-forge/cosmic_profiles.svg
   :target: https://anaconda.org/conda-forge/cosmic_profiles
.. |License| image:: https://anaconda.org/conda-forge/cosmic_profiles/badges/license.svg
   :target: https://anaconda.org/conda-forge/cosmic_profiles
