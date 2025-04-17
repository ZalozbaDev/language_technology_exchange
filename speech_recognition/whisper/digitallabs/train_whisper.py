import os
from datasets import load_dataset
from transformers import (
    WhisperProcessor,
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import torch
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Union
import evaluate
import json
import argparse

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

@dataclass
class ModelArguments:
    model_id: str = field(
        default="openai/whisper-tiny",
        metadata={"help": "The model checkpoint for weights initialization. Can be a Hugging Face Hub ID or a local path."}
    )
    task: str = field(
        default="transcribe",
        metadata={"help": "The task to perform with the model."}
    )
    # Add other model-specific arguments here if needed

@dataclass
class DataArguments:
    training_data_dir: str = field(
        default="./TrainingData/export",
        metadata={"help": "The directory containing the training data."}
    )
    metadata_file: str = field(
        default="metadata.csv",
        metadata={"help": "The metadata file for the dataset."}
    )
    # Remove output_dir to avoid conflict

def main():
    parser = argparse.ArgumentParser(description="Train Whisper Model with specified training arguments.")
    parser.add_argument("args_path", type=str, help="Path to the training_args.json file.")
    args = parser.parse_args()

    # Load training arguments from JSON file
    with open(args.args_path, "r", encoding="utf-8") as f:
        training_args_dict = json.load(f)

    # Initialize data classes
    model_args = ModelArguments(
        model_id=training_args_dict.get("model_id", "openai/whisper-tiny"),
        task=training_args_dict.get("task", "transcribe")
    )
    data_args = DataArguments(
        training_data_dir=training_args_dict.get("training_data_dir", "./TrainingData/export"),
        metadata_file=training_args_dict.get("metadata_file", "metadata.csv")
    )

    # Extract Seq2SeqTrainingArguments
    training_args_fields = {k: v for k, v in training_args_dict.items() if k in Seq2SeqTrainingArguments.__dataclass_fields__}
    training_args = Seq2SeqTrainingArguments(**training_args_fields)

    # Configuration
    MODEL_ID = model_args.model_id
    TASK = model_args.task
    EXPORT_AS = "hf"  # Not used in this script, but kept for consistency
    TRAINING_DATA_DIR = data_args.training_data_dir
    OUTPUT_DIR = training_args.output_dir
    METADATA_FILE = data_args.metadata_file

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ------------------ Begin: Print Configuration Parameters ------------------

    # Convert dataclasses to dictionaries for pretty printing
    model_args_dict = asdict(model_args)
    data_args_dict = asdict(data_args)
    training_args_dict = training_args.to_dict()

    # Print the configurations in a formatted manner
    print("\n=== Configuration Parameters ===\n")
    print("Model Arguments:")
    print(json.dumps(model_args_dict, indent=2))
    print("\nData Arguments:")
    print(json.dumps(data_args_dict, indent=2))
    print("\nTraining Arguments:")
    print(json.dumps(training_args_dict, indent=2))
    print("\n=== End of Configuration Parameters ===\n")

    # ------------------ End: Print Configuration Parameters --------------------

    # Load Dataset
    asr_data = load_dataset("audiofolder", data_dir=TRAINING_DATA_DIR, keep_in_memory=False)

    # Split into train and test
    asr_data = asr_data["train"].train_test_split(test_size=0.2, seed=42)

    if not MODEL_ID.startswith(('openai/', 'facebook/')):
        local_model_path = os.path.join(os.getcwd(), "training", "models", MODEL_ID)
        if os.path.exists(local_model_path):
            print(f"Loading model from local path: {local_model_path}")
            MODEL_PATH = local_model_path
        else:
            raise ValueError(f"Local model not found at {local_model_path}")
    else:
        MODEL_PATH = MODEL_ID
    # Initialize Processor Components
    processor = WhisperProcessor.from_pretrained(MODEL_PATH, task=TASK)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_PATH)
    tokenizer = WhisperTokenizer.from_pretrained(MODEL_PATH, task=TASK)

    # Prepare Dataset
    def prepare_dataset(batch):
        audio = batch["audio"]
        batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        batch["labels"] = tokenizer(batch["transcription"]).input_ids
        return batch

    asr_data = asr_data.map(
        prepare_dataset,
        remove_columns=asr_data["train"].column_names,
        num_proc=1
    )

    # Initialize Data Collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # Evaluation Metric
    metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        label_ids[label_ids == -100] = tokenizer.pad_token_id

        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    # Initialize Model
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    # Initialize Trainer
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=asr_data["train"],
        eval_dataset=asr_data["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor,  # Updated from 'processor.feature_extractor'
    )

    # Save Processor
    processor.save_pretrained(OUTPUT_DIR)

    # Train the Model
    trainer.train()
    # To resume from checkpoint, uncomment the following line and specify the checkpoint path
    # trainer.train(resume_from_checkpoint="path_to_checkpoint")

    # Save the Model
    model.save_pretrained(OUTPUT_DIR)

    # Optional: Synchronize training results (e.g., to another directory or backup location)
    # Example using rsync (uncomment and modify the path as needed)
    # os.system(f"rsync -auv {OUTPUT_DIR} /path/to/backup/location")

if __name__ == "__main__":
    main()
