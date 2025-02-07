#!/bin/bash

rm -rf vctk_finetune_export/

mkdir -p vctk_finetune_export/txt
mkdir -p vctk_finetune_export/wav48

pushd pythonenv
source bin/activate

# pip install alive_progress

python3 ../scripts/vctk_finetune.py

popd
