Mock Universe and Mock Halo Functions
======================================


Mock universe generator
**************************

The ``mock_uni`` module contains a function based on the third-party ``nbodykit`` package to generate a mock Universe whose density distribution is lognormal-distributed given an input redshift, box size and dark matter number density.

.. note:: Additional package to install: `nbodykit <https://nbodykit.readthedocs.io/en/latest/>`_.

.. automodule:: mock_tools.mock_uni
   :members:
   :undoc-members:
   :show-inheritance:


Mock halo generator
**************************

The ``mock_halo_gen`` module provides a function to generate mock halos given a total mass and resolution as well as halo density and shape profile specifications.

.. automodule:: mock_tools.mock_halo_gen
   :members:
   :undoc-members:
   :show-inheritance:
