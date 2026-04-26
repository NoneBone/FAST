#!/bin/bash

# For TA sampler
rm -rf ./*.so
cd ./getSamplerCore
python setup.py build_ext --inplace
mv *.so ../
cd ../

# For TE graph operators
cd ./getFastAgg
python cuda_operators/setup.py install
