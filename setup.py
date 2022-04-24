import pathlib
from setuptools import find_packages, setup
from distutils.core import Extension
import numpy as np
import os
import sysconfig
from Cython.Build import cythonize
from Cython.Distutils import build_ext

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

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

ext_modules=[Extension(
                "cosmic_shapes.cython_helpers",
                sources=['cosmic_shapes/cython_helpers.c'],
                extra_compile_args=["-fopenmp"],
                extra_link_args=["-fopenmp"],
                include_dirs=[np.get_include(), '.']
            ), Extension(
                "cosmic_shapes.gen_csh_gx_cat",
                sources=['cosmic_shapes/gen_csh_gx_cat.pyx'],
                extra_compile_args=["-fopenmp"],
                extra_link_args=["-fopenmp"],
                include_dirs=[np.get_include(), '.']
            ), Extension(
                "cosmic_shapes.cosmic_shapes",
                sources=['cosmic_shapes/cosmic_shapes.c'],
                extra_compile_args=["-fopenmp"],
                extra_link_args=["-fopenmp"],
                include_dirs=[np.get_include(), '.']
            )]

# This call to setup() does all the work
setup(
    name="cosmic_shapes",
    version="1.9.0",
    description="Implements various ellipsoidal shape identification algorithms for 3D particle data",
    long_description=README,
    long_description_content_type="text/markdown",
    project_urls={
    'Documentation': 'https://cosmic-shapes.readthedocs.io/en/latest/',
    'Source': "https://github.com/tibordome/cosmic_shapes"
    },
    author="Tibor Dome",
    author_email="tibor.doeme@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    cmdclass={'build_ext': BuildExtWithoutPlatformSuffix},
    ext_modules = cythonize(ext_modules),
    packages=find_packages(),
    include_package_data=True,
    include_dirs=np.get_include(),
    install_requires=["cython", "numpy>=1.19.2", "scikit-learn", "mpi4py", "h5py", "matplotlib"]
)

