#!/bin/bash

export BASE_MODEL_PATH="/home/lucija/coqui_tts_training/models/multispeaker/2025_02_05/best_model.pth"

pushd pythonenv
source bin/activate

# the default learning rate(s) is/are 0.0002
# the recommended learning rate as per documentation is 0.00001
# 
# trying out various settings between the two

python3 ../scripts/train_vits.py \
--restore_path $BASE_MODEL_PATH \
--coqpit.run_name "vits_finetune" \
--coqpit.lr_gen 0.0001 \
--coqpit.lr_disc 0.0001

popd

# search for this line in the output and start in venv
# tensorboard --logdir=
# then go to
# localhost:6006
