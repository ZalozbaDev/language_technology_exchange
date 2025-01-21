""" Created 2024

    Fine-Tune Open AI whisper, Data preparation

    @author: ivan

Copyright (c) 2024 Fraunhofer IKTS

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
import torch
import torchaudio
from datasets import DatasetDict, Dataset
from tqdm import tqdm  # Progress bar
from transformers import WhisperProcessor, WhisperTokenizer, WhisperFeatureExtractor

# Adjust accordingly
WORK_DIRECTORY = './'
#WORK_DIRECTORY = 'uasr-data/db-hsb-asr/ZalozbaDev/'
CORPUS_PATH = 'hsbcorpus/release_1.0/'

# Paths to your data folders
AUDIO_DIR = WORK_DIRECTORY + CORPUS_PATH + '/sig/'
TRANS_DIR = WORK_DIRECTORY + CORPUS_PATH + '/trl/'

# Paths to file lists for train, test, and dev
TRAIN_FILELIST = WORK_DIRECTORY + CORPUS_PATH + '/train.flst'
TEST_FILELIST = WORK_DIRECTORY + CORPUS_PATH + '/test.flst'
DEV_FILELIST = WORK_DIRECTORY + CORPUS_PATH + '/dev.flst'

EXPECTED_SAMPLE_RATE = 16000  # Whisper expects 16kHz
MAX_PAD_LENGHT = EXPECTED_SAMPLE_RATE * 30

model_name = "openai/whisper-small"
tokenizer = WhisperTokenizer.from_pretrained(model_name, task="transcribe", language ='czech')
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name, task="transcribe", language ='czech')

# Function to load file list
def load_filelist(filelist_path):
    with open(filelist_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

# Load audio and transcription data
def load_data(audio_dir, trans_dir, filelist):
    data = {'audio': [], 'transcription': []}

    for rel_path in tqdm(filelist, desc="Loading data"):
        audio_path = os.path.join(audio_dir, rel_path + '.wav')

        try:
            waveform, sample_rate = torchaudio.load(audio_path)
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            continue

        # wveform should be valid
        if waveform.numel() == 0:
            print(f"Skipping empty audio file: {audio_path}")
            continue

        # Convert audio if needed
        if sample_rate != EXPECTED_SAMPLE_RATE:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=EXPECTED_SAMPLE_RATE)(waveform)

        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Load the transcription
        trans_path = os.path.join(trans_dir, rel_path + '.trl')

        if os.path.exists(trans_path):
            with open(trans_path, 'r', encoding='utf-8') as f:
                transcription = ' '.join([line.strip() for line in f if line.strip()])

            if not transcription:
                #print(f"Skipping empty transcription file: {trans_path}")
                continue

            data['audio'].append(waveform)
            data['transcription'].append(transcription.lower())
        else:
            print(f"Warning: Transcription not found for {audio_path}")

    return data

# Load file lists
train_filelist = load_filelist(TRAIN_FILELIST)
test_filelist = load_filelist(TEST_FILELIST)
dev_filelist = load_filelist(DEV_FILELIST)

# Load the datasets based on the file lists
train_data = load_data(AUDIO_DIR, TRANS_DIR, train_filelist)
test_data = load_data(AUDIO_DIR, TRANS_DIR, test_filelist)
dev_data = load_data(AUDIO_DIR, TRANS_DIR, dev_filelist)

# Create HF datasets
def create_dataset(data):
    return Dataset.from_dict({
        "audio": data['audio'],
        "transcription": data['transcription'],
    })

train_dataset = create_dataset(train_data)
test_dataset = create_dataset(test_data)
dev_dataset = create_dataset(dev_data)

input_str = dev_dataset[0]["transcription"]
labels = tokenizer(input_str).input_ids
decoded_with_special = tokenizer.decode(labels, skip_special_tokens=False)
decoded_str = tokenizer.decode(labels, skip_special_tokens=True)

print(f"Input:                 {input_str}")
print(f"Decoded w/ special:    {decoded_with_special}")
print(f"Decoded w/out special: {decoded_str}")
print(f"Are equal:             {input_str == decoded_str}")


def preprocess_function(batch):
    # Process the audio with the Whisper processor
    inputs = feature_extractor(
        batch["audio"],
        sampling_rate=16000,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
    )

    # Process transcription text
    labels = tokenizer(batch["transcription"], return_tensors="pt", truncation=True).input_ids

    return {
        "input_features": inputs.input_features[0],
        "labels": labels[0]
    }

train_dataset = train_dataset.map(preprocess_function, remove_columns=["audio", "transcription"])#, batched=True, batch_size=8, num_proc=4)
dev_dataset = dev_dataset.map(preprocess_function, remove_columns=["audio", "transcription"])#, batched=True, batch_size=8, num_proc=4)
test_dataset = test_dataset.map(preprocess_function, remove_columns=["audio", "transcription"])#, batched=True, batch_size=8, num_proc=4)


dataset_dict = DatasetDict({
    "train": train_dataset,
    "test": test_dataset,
    "dev": dev_dataset
})

# Save the datasets to disk
output_dir = WORK_DIRECTORY + '/hsbcorpus_dataset_splits'

print(f"Saving Hugging Face dataset splits to {output_dir}")
dataset_dict.save_to_disk(output_dir)

