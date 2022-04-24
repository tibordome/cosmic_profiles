import pathlib
from setuptools import find_packages, setup
from distutils.core import Extension

import numpy as np
from Cython.Build import cythonize

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="cosmic_shapes",
    version="1.7.0",
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
    ext_modules=[
        Extension(
            "cosmic_shapes.cosmic_shapes",
            ["cosmic_shapes/cosmic_shapes.c"],
            extra_compile_args=["-fopenmp"],
            extra_link_args=["-fopenmp"],
            include_dirs=[np.get_include()]
        )
    ],
    packages=["cosmic_shapes"],
    include_package_data=True,
    include_dirs=np.get_include(),
    install_requires=["cython", "numpy>=1.19.2", "scikit-learn", "mpi4py", "h5py", "matplotlib"]
)

