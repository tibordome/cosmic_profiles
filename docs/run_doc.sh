#!/bin/bash

cd ../cosmic_profiles/for_docs
python3 setup.py build_ext --inplace
cd ../../docs
sphinx-build -b html source build
cd ../cosmic_profiles/for_docs
rm cosmic_profiles.c
rm cosmic_profiles.so
rm cython_helpers.c
rm cython_helpers.so
rm gen_csh_gx_cat.c
rm gen_csh_gx_cat.so
rm -r __pycache__
rm -r build
cd ../../docs
