#!/bin/bash

pushd pythonenv
source bin/activate

rm -f ../test_multispeaker.wav

tts --model_path ../models/multispeaker/2025_02_05/best_model.pth \
    --config_path ../models/multispeaker/2025_02_05/config.json \
    --list_speaker_idxs

tts --text "Witajće k nam lubi ludźo. Ja so jara wjeselu, zo je so mi poradźiło, tajki syntetiski hłós wutworić. Nadźijomnje budźe to tež w přichodźe tak." \
    --model_path ../models/multispeaker/2025_02_05/best_model.pth \
    --config_path ../models/multispeaker/2025_02_05/config.json \
    --speaker_idx "VCTK_old_2" \
    --out_path ../test_multispeaker.wav

popd
