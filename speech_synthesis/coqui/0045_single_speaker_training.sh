#!/bin/bash

pushd pythonenv
source bin/activate

python3 ../scripts/train_vits.py \
--coqpit.run_name "vits_single"

popd

# search for this line in the output and start in venv
# tensorboard --logdir=
# then go to
# localhost:6006
