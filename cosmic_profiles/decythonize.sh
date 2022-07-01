#!/bin/bash

cd common
rm profile_classes.c
rm profile_classes.so
rm -r __pycache__
rm -r build
cd ../cython_helpers
rm helper_class.c
rm helper_class.so
rm -r __pycache__
rm -r build
cd ../gadget_hdf5
rm gen_catalogues.c
rm gen_catalogues.so
rm -r __pycache__
rm -r build
cd ../shape_profs
rm shape_profs_algos.c
rm shape_profs_algos.so
rm -r __pycache__
rm -r build
cd ../dens_profs
rm dens_profs_algos.c
rm dens_profs_algos.so
rm -r __pycache__
rm -r build
cd ../tests
rm -r __pycache__
rm -r build
cd cat
rm *
cd ../viz
rm *
cd ../..
rm -r __pycache__
rm -r build