#!/usr/bin/python3 

""" Created 2023

    @author: korla

Copyright (c) 2023 Korla Baier

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

EXPORT_AS = "hf"

from datasets import load_dataset

# check that data is in required format
# asr_data = load_dataset("csv", data_dir="./export/", data_files="metadata.csv", keep_in_memory=True)
asr_data = load_dataset("audiofolder", data_dir="./export/", keep_in_memory=True)

asr_data = asr_data["train"].train_test_split(test_size=0.2)

# print(asr_data["train"][0])

# define model to change
model_id = "openai/whisper-small"

from transformers import WhisperProcessor
processor = WhisperProcessor.from_pretrained(model_id, task="transcribe")

from transformers import WhisperFeatureExtractor
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_id)

from transformers import WhisperTokenizer
tokenizer = WhisperTokenizer.from_pretrained(model_id, task="transcribe")

def prepare_dataset(batch):
	audio = batch["audio"]
	batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

	batch["labels"] = tokenizer(batch["transcription"]).input_ids
	return batch

# check parameter num_proc (observe memory & swap usage?)
asr_data = asr_data.map(prepare_dataset, remove_columns=asr_data.column_names["train"], num_proc=1)

import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
	processor: Any

	def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
		input_features = [{"input_features": feature["input_features"]} for feature in features]
		batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
		
		label_features = [{"input_ids": feature["labels"]} for feature in features]
		labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

		labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
		
		if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
			labels = labels[:, 1:]

		batch["labels"] = labels
		return batch
		
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

import evaluate
metric = evaluate.load("wer")

def compute_metrics(pred):
	pred_ids = pred.predictions
	label_ids = pred.label_ids
	
	label_ids[label_ids == -100] = tokenizer.pad_token_id
	
	pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
	label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
	
	wer = 100 * metric.compute(predictions=pred_str, references=label_str)
	
	return {"wer": wer}

from transformers import WhisperForConditionalGeneration
model = WhisperForConditionalGeneration.from_pretrained(model_id)

model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

from transformers import Seq2SeqTrainingArguments
training_args = Seq2SeqTrainingArguments(
	output_dir="./training/results/",
	per_device_train_batch_size=8,
	gradient_accumulation_steps=2,
	learning_rate=1.25e-4,
	warmup_steps=1000,
	max_steps=100_000,
	gradient_checkpointing=True,
	fp16=True,
	evaluation_strategy="steps",
	per_device_eval_batch_size=16, # test if that works with restricted memory
	predict_with_generate=True,
	generation_max_length=225,
	save_steps=500,
	eval_steps=500,
	logging_steps=25,
	report_to=["tensorboard"],
	load_best_model_at_end=True,
	metric_for_best_model="wer",
	greater_is_better=False,
)

from transformers import Seq2SeqTrainer
trainer = Seq2SeqTrainer(
	args=training_args,
	model=model,
	train_dataset=asr_data["train"],
	eval_dataset=asr_data["test"],
	data_collator=data_collator,
	compute_metrics=compute_metrics,
	tokenizer=processor.feature_extractor,
)

# decide here whether to resume from checkpoint
trainer.train()
# trainer.train(resume_from_checkpoint=True)

model.save_pretrained(training_args.output_dir)
processor.save_pretrained(training_args.output_dir)

