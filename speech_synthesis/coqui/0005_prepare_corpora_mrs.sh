#!/bin/bash

rm -rf corpora/temp/speech_corpus_michal_recordings
mkdir -p corpora/temp/speech_corpus_michal_recordings

# rearrange data from repo to resemble mimic recording studio session --> required for conversion

mkdir -p corpora/temp/speech_corpus_michal_recordings/backend/audio_files/michalcyz_1977-03-29/
mkdir -p corpora/temp/speech_corpus_michal_recordings/backend/db/

cp corpora/speech_corpus_michal_recordings/mimic/michalcyz_1977-03-29/*.wav corpora/temp/speech_corpus_michal_recordings/backend/audio_files/michalcyz_1977-03-29/
cp corpora/speech_corpus_michal_recordings/mimic/db/mimicstudio.db          corpora/temp/speech_corpus_michal_recordings/backend/db/

pushd pythonenv
source bin/activate

# pip install alive_progress

python3 ../scripts/MRS2LJSpeech.py --mrs_dir ../corpora/temp/speech_corpus_michal_recordings/

popd

rm -rf corpora/export/
mkdir -p corpora/export/

mkdir -p corpora/export/speech_corpus_michal_recordings/
cp -r scripts/dataset/LJSpeech-1.1/ corpora/export/speech_corpus_michal_recordings/
