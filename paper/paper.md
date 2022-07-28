---
title: 'CosmicProfiles: A Python package for radial profiling of finitely sampled dark matter halos and galaxies'
tags:
  - Python
  - astronomy
  - cosmology
  - simulations
  - point clouds
authors:
  - name: Tibor Dome
    affiliation: 1
affiliations:
 - name: PhD Student, University of Cambridge, Institute of Astronomy, United Kingdom
   index: 1
date: 7 July 2022
bibliography: paper.bib
---

# Summary

During the evolution of the Universe, dark matter clumps under its own gravitational
influence and forms quasi-equilibrium halos. Their density and shape profiles are
key to understanding the impact of exotic dark matter scenarios and the role baryonic
feedback plays in the central regions of dark halos. Substantial simulation-based effort
has been invested to model the approximately universal density profiles and to qualitatively
track the evolution of shape profiles. The shapes of galaxies and their statistical correlations
have recently received increased attention since the results of the next generation of weak
lensing surveys such as Euclid will be contaminated by intrinsic alignment effects
without a proper treatment thereof.

# Statement of need

The analysis of simulation outputs that inform observational searches requires
reliable and fast numerical tools. `CosmicProfiles` is a Python package with a
substantial Cython component to enable quick and easy calculation of global and
local density as well as shape properties of finitely resolved objects. The shape
profiles of the objects can in turn be used to improve the fidelity of density
profiles by considering ellipsoidal shells instead of the popular choice of
spherical shells. While `Colossus` [@Diemer_2018] provides some density profile
fitting functionalities, `CosmicProfiles` in addition offers spherical shell-based
and ellipsoidal shell-based density profile estimation techniques to find such
profiles in the first place.

The objects under consideration can either be halos or galaxies from a cosmological
simulation but also point clouds from any other scientific research field. The API
for `CosmicProfiles` was designed to provide a class-based and user-friendly interface
to Cython-optimized implementations of common operations such as the estimation of
density profiles and subsequent fitting thereof to the user's preferred density
profile model. Interfaces to `pynbody` [@Pontzen_2013] and `nbodykit` [@Hand_2018]
have been rendered very simple with detailed example scripts, such that e.g.
halos that have been identified via `pynbody` can be fed to `CosmicProfiles`
for radial profiling.

If no halos are available to the user, `CosmicProfiles` offers a versatile mock
halo generator that will sample particles from a target density profile model and
ellipsoidal shape distribution, all provided by the user. `CosmicProfiles`
was designed to be used by both astronomical researchers and researchers of other
fields that come across ellipsoidal point clouds. It has already been used in
a scientific publication [@Dome_2022]. The combination of speed and design
will hopefully ease the post-processing of simulation snapshots such as Gadget-style
HDF5 files [@Springel_2010] and contribute to exciting scientific explorations
by students and experts alike.

# Acknowledgements

It is a pleasure to thank my PhD supervisor Anastasia Fialkov for the patient guidance, encouragement and advice during the genesis of this project. This work received support from STFC under grant number ST/V50659X/1.

# References
