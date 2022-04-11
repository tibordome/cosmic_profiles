#!/bin/bash

cd ../cosmic_shapes/for_docs
python3 setup.py build_ext --inplace
cd ../../docs
sphinx-build -b html source build
cd ../cosmic_shapes/for_docs
#rm cosmic_shapes.c
#rm cosmic_shapes.so
#rm cython_helpers.c
#rm cython_helpers.so
#rm gen_csh_gx_cat.c
#rm gen_csh_gx_cat.so
rm -r __pycache__
rm -r build
cd ../../docs
