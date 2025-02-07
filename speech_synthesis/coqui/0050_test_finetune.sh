#!/bin/bash

pushd pythonenv
source bin/activate

rm -f ../test_finetune.wav

tts --text "Witajće k nam lubi ludźo. Ja so jara wjeselu, zo je so mi poradźiło, tajki syntetiski hłós wutworić. Nadźijomnje budźe to tež w přichodźe tak." \
    --model_path ../models/finetune/2025_01_31_weronika/best_model_57475.pth \
    --config_path ../models/finetune/2025_01_31_weronika/config.json \
    --out_path ../test_finetune.wav

popd
