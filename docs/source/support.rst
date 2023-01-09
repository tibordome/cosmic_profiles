*************************
Installation and Support
*************************

*CosmicProfiles* makes use of the following packages:

- numpy>=1.19.2
- scipy
- cython
- scikit-learn
- mpi4py
- h5py
- pathos
- matplotlib<3.5

  - versions >=3.5 will bicker over ``auto_add_to_figure=False`` not being an admissible argument for ``Axes3D``

- psutil

Source code for these is included with *CosmicProfiles* and is built automatically, so you do not need to install them yourself. However, beware of MPI and OpenMP (see below).

Binary installation with conda
*********************************

This is the recommended installation path for Anaconda and Miniconda users. Conda Forge provides a `conda channel <https://anaconda.org/conda-forge/cosmic_profiles>`_ with a pre-compiled version of *CosmicProfiles* for linux 64bit and MAC OS X platforms. You can install it in Anaconda with::

    conda config --add channels conda-forge
    conda install cosmic_profiles

.. note:: Running *CosmicProfiles* relies on a functioning MPI implementation. For macOS and Linux, MPICH can be installed in the command line using ``brew install mpich`` or ``sudo apt install mpich``, respectively.

Binary installation with pip
*********************************

Installing *CosmicProfiles* from the Python Package Index using `pip <https://pypi.org/project/cosmic-profiles/>`_ is recommended for most other users. For most common architectures and platforms (Linux x86-64, Linux i686, and macOS x86-64), pip will download and install a pre-built binary. For other platforms, it will automatically try to build *CosmicProfiles* from source. 

To install the latest version of *CosmicProfiles* with pip, simply run::
    
    pip install --user cosmic-profiles

If you have installed with pip, you can keep your installation up to date by upgrading from time to time::
    
    pip install --user --upgrade cosmic-profiles

.. note:: Pip will most likely rely on OpenMP support being enabled for the compiler (gcc for Linux or clang for macOS) during the installation process, and while this is easy to establish on Linux (via ``sudo apt install libomp-dev``), it is challenging on macOS (albeit possible, see `here <https://blog.llvm.org/2015/05/openmp-support_22.html>`_).

Contributing
****************************

The easiest way to get help with the code or to contribute is to raise an issue or open a pull request (PR) on GitHub.

Github: https://github.com/tibordome/cosmic_profiles
