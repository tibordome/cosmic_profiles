cd ../cosmic_profiles/for_docs
python3 setup_compile.py build_ext --inplace
cd ../../docs
sphinx-build -b html source build
