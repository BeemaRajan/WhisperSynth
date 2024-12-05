
import os
import torch
from datasets import load_dataset, Dataset
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)
import evaluate
import torchaudio
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from typing import Any, Dict, List, Union

# Define paths
data_path = '/content/drive/MyDrive/whisper_synth_files/data/dataset1/'

# Load the dataset
data_files = {'train': data_path + 'train.csv'}
dataset = load_dataset('csv', data_files=data_files)

# Split the dataset into training and evaluation sets
_, eval_data = train_test_split(dataset['train'].to_pandas(), test_size=0.2)

# Convert eval data back to Dataset object
eval_dataset = Dataset.from_pandas(pd.DataFrame(eval_data))

# Load the tokenizer and processor from the combined model directory
save_path = "/content/drive/MyDrive/whisper_synth_files/whisper_synth"

# Load the tokenizer
tokenizer = WhisperTokenizer.from_pretrained(save_path)

# Load processor and model for evaluation
processor = WhisperProcessor.from_pretrained(
    save_path,
    language="en",
    task="transcribe",
    tokenizer=tokenizer
)

# Load the combined model
model = WhisperForConditionalGeneration.from_pretrained(save_path)

# Force the model to generate in English
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
    language="en", task="transcribe"
)

def preprocess_function(examples):
    audio_path = examples['audio']
    audio_array, sampling_rate = torchaudio.load(audio_path)

    # Resample if needed
    if sampling_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)
        audio_array = resampler(audio_array)

    audio_array = audio_array.squeeze().numpy()

    # Extract input features (log-Mel spectrogram)
    input_features = processor.feature_extractor(
        audio_array, sampling_rate=16000
    ).input_features[0]

    # Tokenize target text to create decoder input IDs
    text = examples['text']
    labels = processor.tokenizer(
        text
    ).input_ids

    # Return a dictionary with the correct keys
    return {
        "input_features": input_features,
        "labels": labels
    }

# Preprocess eval dataset
eval_dataset = eval_dataset.map(preprocess_function, remove_columns=['audio', 'text'])

# Helper function to parse parameters from structured text
def parse_parameters(text):
    params = {}
    lines = text.strip().split('\n')
    for line in lines:
        if ':' in line:
            key, value = line.split(':', 1)
            params[key.strip()] = value.strip()
    return params

# Define custom compute_metrics function
def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    # Replace -100 with pad_token_id
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id

    # Decode predictions and references without special tokens
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    for p, l in zip(pred_str, label_str):
        print("Prediction:", p)
        print("Reference:", l)
        print("---")

    total = len(pred_str)
    exact_matches = sum([1 for p, l in zip(pred_str, label_str) if p.strip() == l.strip()])
    exact_match_accuracy = exact_matches / total

    # Parameter-level accuracy
    parameter_accuracy = {}
    for param in ["Waveform", "Voices", "Oscillator Detune", "Filter Type", "Filter Cutoff", "ADSR Envelope", "LFO Modulation"]:
        correct = 0
        for p, l in zip(pred_str, label_str):
            pred_params = parse_parameters(p)
            label_params = parse_parameters(l)
            if pred_params.get(param) == label_params.get(param):
                correct += 1
        parameter_accuracy[param] = correct / total

    # Combine metrics
    metrics = {"exact_match_accuracy": exact_match_accuracy}
    metrics.update(parameter_accuracy)
    return metrics

# Define the data collator
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Separate input_features and labels
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        labels = [{"input_ids": feature["labels"]} for feature in features]

        # Pad input_features using the feature extractor
        batch = self.processor.feature_extractor.pad(
            input_features,
            return_tensors="pt"
        )

        # Pad labels using the tokenizer
        labels_batch = self.processor.tokenizer.pad(
            labels,
            padding=True,
            return_tensors="pt"
        )

        # Replace padding token id's of the labels by -100
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Remove the decoder_start_token_id
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

# Initialize the data collator
data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

# Define Trainer for evaluation
training_args = Seq2SeqTrainingArguments(
    output_dir="/content/drive/MyDrive/whisper_synth_files/eval_logs",
    per_device_eval_batch_size=4,
    predict_with_generate=True,
    eval_strategy="no",  # Set to "no" as we're only evaluating here
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    eval_dataset=eval_dataset,
    tokenizer=processor.feature_extractor,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
)

# Perform evaluation
eval_results = trainer.evaluate()

# Print evaluation results
print(eval_results)
