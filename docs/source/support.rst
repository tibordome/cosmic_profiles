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
- matplotlib
- psutil

Source code for these is included with *CosmicProfiles* and is built automatically, so you do not need to install them yourself.


Binary installation with conda
*********************************

This is the recommended installation path for Anaconda and Miniconda users. Conda Forge provides a `conda channel <https://anaconda.org/conda-forge/cosmic_profiles>`_ with a pre-compiled version of *CosmicProfiles* for linux 64bit and MAC OS X platforms. You can install it in Anaconda with::

    conda config --add channels conda-forge
    conda install cosmic_profiles

Binary installation with pip
*********************************

Installing *CosmicProfiles* from the Python Package Index using `pip <https://pypi.org/project/cosmic-profiles/>`_ is recommended for most other users. For most common architectures and platforms (Linux x86-64, Linux i686, and macOS x86-64), pip will download and install a pre-built binary. For other platforms, it will automatically try to build *CosmicProfiles* from source. 

To install the latest version of *CosmicProfiles* with pip, simply run::
    
    pip install --user cosmic-profiles

If you have installed with pip, you can keep your installation up to date by upgrading from time to time::
    
    pip install --user --upgrade cosmic-profiles


Contributing
****************************

The easiest way to get help with the code or to contribute is to raise an issue or open a pull request (PR) on GitHub.

Github: https://github.com/tibordome/cosmic_profiles
