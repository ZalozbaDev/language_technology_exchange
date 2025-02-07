#!/bin/bash

pushd pythonenv
source bin/activate

# pip install -e ../TTS/[all,dev,notebooks]

python3 ../scripts/multi_vits.py

popd

# search for this line in the output and start in venv
# tensorboard --logdir=
# then go to
# localhost:6006
