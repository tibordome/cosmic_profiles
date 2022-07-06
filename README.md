<img src="./info/CProfiles.png" alt="Cosmic Profiles logo" style="height: 100px; width:400px;"/>

CosmicProfiles is a Python package for Point Cloud Profiling

[![Documentation Status](https://readthedocs.org/projects/cosmic-profiles/badge/?version=latest)](https://cosmic-profiles.readthedocs.io/en/latest/?badge=latest) [![PyPI version](https://badge.fury.io/py/cosmic-profiles.svg)](https://badge.fury.io/py/cosmic-profiles)

# The *CosmicProfiles* project

This repository provides shape and density profile analysis tools for cosmological simulations (and beyond). Its features include

- overall halo shape determination, i.e. major, intermediate, minor axis vectors and shape quantities such as intermediate-to-major axis ratio or sphericity
- halo shape profile determination
  - iterative shell-based shape profile determination algorithm for high-resolution halos
  - iterative ellipsoid-based shape profile determination algorithm for lower-resolution halos
- works with
  - 'direct' datasets (i.e. index catalogue provided by user) and
  - Gadget-style HDF5 snapshot files
    - additional velocity dispersion tensor eigenaxes determination
    - galaxy density and shape profile determination also works out of the box
- halo density profile estimation using direct binning and kernel-based approaches
- density profile fitting assuming either NFW, Hernquist 1990, Einasto or $\alpha \beta \gamma$-profile model
  - concentration-mass relationship of halos easy to calculate
- mock halo generator: ellipsoidal or spherical, compatible with the 4 density profile models
- easy to interface with `pynbody` to work with halos identified in a cosmological simulation (see example scripts)
- easy to interface with `nbodykit` to harness large-scale structure capabilities (see example scripts)
- 3D point cloud plotting tools

The documentation can be found [here](https://cosmic-profiles.readthedocs.io/en/latest/index.html).

Contributions are welcome. Please raise a PR. Comments/recommendations/complaints can be sent [here](mailto:tibor.doeme@gmail.com).
