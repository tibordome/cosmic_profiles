#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 12:12:27 2020
"""

from distutils.core import setup, Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import os
import sysconfig
import numpy as np

def get_ext_filename_without_platform_suffix(filename):
    """ Retrieve filename of default cython files without the machine-dependent suffixes"""
    name, ext = os.path.splitext(filename)
    ext_suffix = sysconfig.get_config_var('EXT_SUFFIX')

    if ext_suffix == ext:
        return filename

    ext_suffix = ext_suffix.replace(ext, '')
    idx = name.find(ext_suffix)

    if idx == -1:
        return filename
    else:
        return name[:idx] + ext
    

class BuildExtWithoutPlatformSuffix(build_ext):
    """ Build Cython with naming convention that discards the machine-dependent suffixes"""
    def get_ext_filename(self, ext_name):
        filename = super().get_ext_filename(ext_name)
        return get_ext_filename_without_platform_suffix(filename)

extension1 = [Extension(
                "cython_helpers.helper_class",
                sources=['cython_helpers/helper_class.pyx'],
                extra_compile_args=["-fopenmp"],
                extra_link_args=["-fopenmp"],
                include_dirs=[np.get_include(), '.']
            )]
    


setup(
    cmdclass={'build_ext': BuildExtWithoutPlatformSuffix},
    ext_modules = cythonize(extension1)
)

extension2 = [Extension(
                "gadget_hdf5.gen_catalogues",
                sources=['gadget_hdf5/gen_catalogues.pyx'],
                extra_compile_args=["-fopenmp"],
                extra_link_args=["-fopenmp"],
                include_dirs=[np.get_include(), '.']
            )]
    


setup(
    cmdclass={'build_ext': BuildExtWithoutPlatformSuffix},
    ext_modules = cythonize(extension2)
)

extension3 = [Extension(
                "dens_profs.dens_profs_algos",
                sources=['dens_profs/dens_profs_algos.pyx'],
                extra_compile_args=["-fopenmp"],
                extra_link_args=["-fopenmp"],
                include_dirs=[np.get_include(), '.']
            )]
    


setup(
    cmdclass={'build_ext': BuildExtWithoutPlatformSuffix},
    ext_modules = cythonize(extension3)
)

extension4 = [Extension(
                "shape_profs.shape_profs_algos",
                sources=['shape_profs/shape_profs_algos.pyx'],
                extra_compile_args=["-fopenmp"],
                extra_link_args=["-fopenmp"],
                include_dirs=[np.get_include(), '.']
            )]
    


setup(
    cmdclass={'build_ext': BuildExtWithoutPlatformSuffix},
    ext_modules = cythonize(extension4)
)

extension5 = [Extension(
                "common.profile_classes",
                sources=['common/profile_classes.pyx'],
                extra_compile_args=["-fopenmp"],
                extra_link_args=["-fopenmp"],
                include_dirs=[np.get_include(), '.']
            )]
    


setup(
    cmdclass={'build_ext': BuildExtWithoutPlatformSuffix},
    ext_modules = cythonize(extension5)
)
