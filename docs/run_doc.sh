#!/bin/bash

cd ../cosmic_profiles/for_docs
python3 setup.py build_ext --inplace
cd ../../docs
sphinx-build -b html source build
cd ../cosmic_profiles/for_docs/common
rm cosmic_base_class.c
rm cosmic_base_class.so
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
rm shape_profs_classes.c
rm shape_profs_classes.so
rm shape_profs_algos.c
rm shape_profs_algos.so
rm -r __pycache__
rm -r build
cd ../dens_profs
rm dens_profs_classes.c
rm dens_profs_classes.so
rm dens_profs_algos.c
rm dens_profs_algos.so
rm -r __pycache__
rm -r build
cd ../mock_tools
rm -r __pycache__
rm -r build
cd ../tests
rm -r __pycache__
rm -r build
rm -r cat
rm -r viz
cd ..
rm -r __pycache__
rm -r build
cd ../../docs
