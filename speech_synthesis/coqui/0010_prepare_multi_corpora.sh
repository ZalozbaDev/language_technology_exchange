#!/bin/bash

rm -rf vctk_export/

mkdir -p vctk_export/txt
mkdir -p vctk_export/wav48

pushd pythonenv
source bin/activate

# pip install alive_progress

python3 ../scripts/vctk_multi.py

popd
