#!/bin/bash

cd ../cosmic_profiles
python3 setup.py build_ext --inplace
source decythonize.sh
cd ../docs