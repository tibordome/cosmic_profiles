[![Documentation Status](https://readthedocs.org/projects/cosmic-profiles/badge/?version=latest)](https://cosmic-profiles.readthedocs.io/en/latest/?badge=latest) [![PyPI version](https://badge.fury.io/py/cosmic-profiles.svg)](https://badge.fury.io/py/cosmic-profiles)

# The *Cosmic Profiles* project

This repository provides shape and density profile analysis tools for cosmological simulations (and beyond). Its features include

- iterative shell-based shape determination algorithm for high-resolution halos/galaxies
- iterative ellipsoid-based shape determination algorithm for lower-resolution halos/galaxies
- overall halo/galaxy shape determination, shape profiles determination, velocity dispersion tensor eigenaxes determination
- halo/galaxy density profile estimation using direct binning and kernel-based approaches
- density profile fitting assuming either NFW, Hernquist 1990, Einasto or $\alpha \beta \gamma$-profile model
- wrapper to the Amiga Halo Finder (AHF) via `pynbody` to identify halos in a cosmological simulation (or mock universe)
- mock point cloud generator: ellipsoidal or spherical, compatible with the 4 density profile models 
- mock log-normal ''universe'' generator
- example scripts.

The documentation can be found [here](https://cosmic-profiles.readthedocs.io/en/latest/index.html).

Contributions are welcome. Please raise a PR. Comments/recommendations/complaints can be sent [here](mailto:tibor.doeme@gmail.com).
