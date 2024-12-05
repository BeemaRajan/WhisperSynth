
import os
import torch
from transformers import WhisperForConditionalGeneration, WhisperTokenizer, WhisperProcessor
from peft import PeftModel

# Load the tokenizer used during fine-tuning
tokenizer = WhisperTokenizer.from_pretrained("/content/drive/MyDrive/whisper_synth_files/whisper_tokenizer_with_special_tokens")

# Load the base model and resize embeddings
base_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
base_model.resize_token_embeddings(len(tokenizer))

# Load the PeftModel with the adapters
adapter_checkpoint = "/content/drive/MyDrive/whisper_synth_files/whisper_finetuned/checkpoint-400"

model = PeftModel.from_pretrained(
   base_model,
   adapter_checkpoint
)

# Merge adapter weights with the base model
model = model.merge_and_unload()

# Save the complete model for future use
save_path = "/content/drive/MyDrive/whisper_synth_files/whisper_synth"
model.save_pretrained(save_path)

# Also save the processor with the correct tokenizer
processor = WhisperProcessor.from_pretrained(
   "openai/whisper-small",
   tokenizer=tokenizer
)
processor.save_pretrained(save_path)
