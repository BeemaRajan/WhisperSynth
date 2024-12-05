
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
from peft import get_peft_model, LoraConfig
import torchaudio
from sklearn.model_selection import train_test_split
import pandas as pd
from dataclasses import dataclass
from typing import Any, Dict, List, Union

# Define paths
data_path = '/content/drive/MyDrive/whisper_synth_files/data/dataset1/'

# Load dataset
data_files = {'train': data_path + 'train.csv'}
dataset = load_dataset('csv', data_files=data_files)

# Split the dataset into training and evaluation sets
train_data, eval_data = train_test_split(dataset['train'].to_pandas(), test_size=0.2)

# Convert train and eval data back to Dataset objects
train_dataset = Dataset.from_pandas(pd.DataFrame(train_data))
eval_dataset = Dataset.from_pandas(pd.DataFrame(eval_data))

# Load the updated tokenizer
tokenizer = WhisperTokenizer.from_pretrained("/content/drive/MyDrive/whisper_synth_files/whisper_tokenizer_with_special_tokens")

# Load the processor with the updated tokenizer
processor = WhisperProcessor.from_pretrained(
    "openai/whisper-small",
    language="en",
    task="transcribe",
    tokenizer=tokenizer
)

# Load the model
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

# Resize the model's embeddings to accommodate new tokens
model.resize_token_embeddings(len(tokenizer))

def preprocess_function(examples):
    audio_path = examples['audio']
    audio_array, sampling_rate = torchaudio.load(audio_path)

    if sampling_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)
        audio_array = resampler(audio_array)

    audio_array = audio_array.squeeze().numpy()

    # Extract input features
    input_features = processor.feature_extractor(
        audio_array, sampling_rate=16000
    ).input_features[0]

    # Tokenize target text
    text = examples['text']
    labels = processor.tokenizer(
        text
    ).input_ids

    # Return a dictionary with the correct keys
    return {
        "input_features": input_features,
        "labels": labels
    }

# Preprocess datasets
train_dataset = train_dataset.map(preprocess_function, remove_columns=['audio', 'text'])
eval_dataset = eval_dataset.map(preprocess_function, remove_columns=['audio', 'text'])

# Apply LoRA configuration
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
    lora_dropout=0.1,
    bias="none"
)
model = get_peft_model(model, peft_config)

# Define custom data collator
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

# Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="/content/drive/MyDrive/whisper_synth_files/whisper_finetuned",
    per_device_train_batch_size=4,
    learning_rate=3e-5,
    num_train_epochs=10,
    logging_dir="/content/drive/MyDrive/whisper_synth_files/logs",
    logging_strategy="steps",
    logging_steps=10,
    save_total_limit=2,
    save_steps=500,
    eval_strategy="epoch",
    eval_steps=500,
    predict_with_generate=True,
    fp16=True,
)

# Define custom metric
def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    # Replace -100 with pad_token_id
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id

    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_str = processor.tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    # Calculate exact match accuracy
    exact_matches = [int(p.strip() == l.strip()) for p, l in zip(pred_str, labels_str)]
    accuracy = sum(exact_matches) / len(exact_matches)

    return {"accuracy": accuracy}

# Define Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=processor,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
)

# Start training
trainer.train()
