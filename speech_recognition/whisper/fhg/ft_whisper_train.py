""" Created 2024

    Fine-Tune Open AI whisper, Training

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
from datasets import DatasetDict
from transformers import WhisperForConditionalGeneration, WhisperProcessor, Seq2SeqTrainingArguments, Seq2SeqTrainer
#from datasets import load_from_disk
import evaluate
from dataclasses import dataclass
from typing import Any, Dict, List, Union

os.environ["CUDA_VISIBLE_DEVICES"]="0"
#os.environ["CUDA_LAUNCH_BLOCKING"]="1"

# Adjust acordingly
WORK_DIRECTORY = './'
#WORK_DIRECTORY = 'uasr-data/db-hsb-asr/ZalozbaDev/'

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:

        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences and pad them
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    cer = 100 * metric_cer.compute(predictions=pred_str, references=label_str)

    return {"wer": wer, "cer": cer}


# Load the dataset
output_dir = WORK_DIRECTORY + '/hsbcorpus_dataset_splits'
dataset_dict = DatasetDict.load_from_disk(output_dir)

# Load model and processor
model_name = "openai/whisper-small"
processor = WhisperProcessor.from_pretrained(model_name, language ='czech', task='transcribe')
model = WhisperForConditionalGeneration.from_pretrained(model_name)
model.config.max_target_lenght = 1024


model.config.use_cache = False

# Prepare datasets
train_dataset = dataset_dict['train']
dev_dataset = dataset_dict['dev']
test_dataset = dataset_dict['test']

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

metric = evaluate.load("wer")
metric_cer = evaluate.load("cer")

collate_fn = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)


training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-finetuned",  # change to a repo name of your choice
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,  # increase by 2x for every 2x decrease in batch size
    learning_rate = 1.25e-4,
    warmup_steps=500,
    #max_steps=100,
    gradient_checkpointing=True,
    fp16=True,
    eval_strategy="steps",
    per_device_eval_batch_size=16,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=500,
    eval_steps=500,
    logging_steps=25,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
    num_train_epochs=5,
    lr_scheduler_type="linear",
    weight_decay=0.01,
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    tokenizer=processor,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
)

# Fine-tune the model
#trainer.train(resume_from_checkpoint="./whisper-finetuned/checkpoint-6510")
trainer.train()

model.config.use_cache = True
# Save the model
model.save_pretrained("./whisper-finetuned-model",safe_serialization=False)
processor.save_pretrained("./whisper-finetuned-model",safe_serialization=False)
generation_config = model.config
generation_config.save_pretrained("./whisper-finetuned-model")

# Evaluate on the dev dataset
dev_results = trainer.evaluate(dev_dataset)
print(f"Dev evaluation results: {dev_results}")

# Evaluate on the test dataset
test_results = trainer.evaluate(test_dataset)
print(f"Test evaluation results: {test_results}")


