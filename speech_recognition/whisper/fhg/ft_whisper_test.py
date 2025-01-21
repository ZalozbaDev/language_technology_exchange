""" Created 2024

    Fine-Tune Open AI whisper, Evaluation

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

import torch
from datasets import DatasetDict, Dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor, GenerationConfig
from jiwer import wer, cer
import evaluate
import csv
import os
import string

os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Adjust acordingly
WORK_DIRECTORY = './'
TEST_SET = './hsbcorpus_dataset_splits/test'
#WORK_DIRECTORY = 'uasr-data/db-hsb-asr/ZalozbaDev/'

model_path = WORK_DIRECTORY + 'whisper-finetuned-model'

data_path = WORK_DIRECTORY + TEST_SET

dataset = Dataset.load_from_disk(data_path)

# Load the  model and processor
model = WhisperForConditionalGeneration.from_pretrained(model_path)
processor = WhisperProcessor.from_pretrained(model_path, task = "transcribe", language= "czech" )
generation_config = GenerationConfig.from_pretrained(model_path)

model.config.max_target_length = 1024

model.eval()

# Subset of data for testing
#data = dataset['dev'].select(range(50))
data = dataset#.select(range(50))

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Initialize WER and CER
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")


def clean_text(text):
    return text.translate(str.maketrans('', '', string.punctuation)).lower()


def evaluate_model(batch):
    audio_features = batch['input_features']
    with torch.no_grad():

        inputs = torch.tensor(audio_features, dtype=torch.float32).to(device)
        predicted_ids = model.generate(inputs, generation_config=generation_config)
        batch['predicted'] = [clean_text(pred) for pred in processor.batch_decode(predicted_ids, skip_special_tokens=True)]
        batch['correct'] = [clean_text(corr) for corr in processor.batch_decode(batch['labels'], skip_special_tokens=True)]
    return batch

results = data.map(evaluate_model, batched=True, batch_size=16)

# Compute WER and CER
wer_results = 100 * wer_metric.compute(predictions=results['predicted'], references=results['correct'])
cer_results = 100 * cer_metric.compute(predictions=results['predicted'], references=results['correct'])

# Print results
print(f'Average WER: {wer_results:.2f}')
print(f'Average CER: {cer_results:.2f}')

# Output results to CSV file
with open('whisper_results.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['transcription', 'correct_sentence', 'WER', 'CER'])

    # Write each result to the CSV file
    for idx, predicted in enumerate(results['predicted']):
        correct = results['correct'][idx]
        #file_name = data['file'][idx]  # Assuming 'file' column contains the file names
        wer_score = wer([correct], [predicted])
        cer_score = cer([correct], [predicted])
        writer.writerow([predicted, correct, wer_score, cer_score])
