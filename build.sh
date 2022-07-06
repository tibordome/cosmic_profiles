cd cosmic_profiles
$PYTHON setup.py build_ext --inplace
cd ..
$PYTHON setup.py install --single-version-externally-managed --record=record.txt
