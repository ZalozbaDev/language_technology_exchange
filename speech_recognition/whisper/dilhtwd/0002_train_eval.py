#!/usr/bin/python3 

""" Created 2023

    @author: korla, tamas

Copyright (c) 2024 Korla Baier, Tamás Janusko

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

from datasets import load_dataset, load_from_disk, DatasetDict

import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import evaluate

from transformers import WhisperProcessor
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperForConditionalGeneration
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer
from transformers import get_scheduler
from transformers import AdamW


freeze_layers = False
layers_number = 15

lr_dict = {
		"openai/whisper-tiny": 3.75e-5,
		"openai/whisper-small": 1.25e-5,
		"openai/whisper-base": 2.5e-5,
		"openai/whisper-medium": 6.25e-6,
		"openai/whisper-large-v3": 5e-6,
		"openai/whisper-large-v3-turbo": 5e-6,
	}


if torch.cuda.is_available():
	print("CUDA is available.")
	print(f"Torch Version: {torch.__version__}")
	print(f"Torch CUDA Version: {torch.version.cuda}")
	device = "cuda"
else:
	print("CUDA not available.")
	exit()
	device = "cpu"

with torch.cuda.device(device):

	asr_data = load_dataset("audiofolder", data_dir="../export2", keep_in_memory=True)

	train_test_valid = asr_data["train"].train_test_split(test_size=0.2, seed=1337)

	train_data = train_test_valid["train"]

	test_valid = train_test_valid["test"].train_test_split(test_size=0.5, seed=1337)

	#test_data = test_valid["train"]   # 10% of total data
	#validation_data = test_valid["test"] # 10% of total data

	asr_data = DatasetDict({
		"train": train_test_valid["train"],
		"test":  test_valid["train"],
		"valid": test_valid["test"]
		}
	)

	# define model to change
	model_id = "openai/whisper-base"

	processor = WhisperProcessor.from_pretrained(model_id, task="transcribe")

	feature_extractor = WhisperFeatureExtractor.from_pretrained(model_id)

	tokenizer = WhisperTokenizer.from_pretrained(model_id, task="transcribe")



	def prepare_dataset(batch):
		audio = batch["audio"]
		batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

		batch["labels"] = tokenizer(batch["transcription"]).input_ids
		return batch

	asr_data = asr_data.map(prepare_dataset, remove_columns=asr_data.column_names["train"], num_proc=1)


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


	metric = evaluate.load("wer")

	def compute_metrics(pred):
		pred_ids = pred.predictions
		label_ids = pred.label_ids

		label_ids[label_ids == -100] = tokenizer.pad_token_id

		pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
		label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

		wer = 100 * metric.compute(predictions=pred_str, references=label_str)

		return {"wer": wer}


	model = WhisperForConditionalGeneration.from_pretrained(model_id)

	# freeze first 15 layers as argued in https://asmp-eurasipjournals.springeropen.com/articles/10.1186/s13636-024-00349-3
	if freeze_layers:
		i = 0
		for layer in model.base_model.encoder.layers:
			if i < layers_number:
				layer.requires_grad = False
				i+=1
			else:
				break

	model.to(device)
	model.config.forced_decoder_ids = None
	model.config.suppress_tokens = []

	optimizer = AdamW(model.parameters(), lr=lr_dict[model_id])

	total_training_steps = 100_000
	warmup_steps = 1000

	scheduler = get_scheduler(
    		name="linear",
    		optimizer=optimizer,
    		num_warmup_steps=warmup_steps,
    		num_training_steps=total_training_steps
	)


	training_args = Seq2SeqTrainingArguments(
		output_dir="./training/results_base/",
		per_device_train_batch_size=64,
		gradient_accumulation_steps=1,
		learning_rate=lr_dict[model_id],
		warmup_steps=warmup_steps,
		max_steps=total_training_steps,
		gradient_checkpointing=True,
		gradient_checkpointing_kwargs={'use_reentrant':False},
		bf16=True,
		eval_strategy="steps",
		per_device_eval_batch_size=16,
		predict_with_generate=True,
		generation_max_length=225,
		save_steps=1000,
		eval_steps=1000,
		logging_steps=25,
		report_to=["tensorboard"],
		load_best_model_at_end=True,
		metric_for_best_model="wer",
		greater_is_better=False,
	)


	trainer = Seq2SeqTrainer(
		args=training_args,
		model=model,
		train_dataset=asr_data["train"],
		eval_dataset=asr_data["valid"],
		data_collator=data_collator,
		compute_metrics=compute_metrics,
		tokenizer=processor.feature_extractor,
		optimizers=(optimizer, scheduler),
	)

	# decide here whether to resume from checkpoint
	trainer.train()
	#trainer.train(resume_from_checkpoint=True)

	model.save_pretrained(training_args.output_dir)
	processor.save_pretrained(training_args.output_dir)

