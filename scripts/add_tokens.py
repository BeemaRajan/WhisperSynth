
from transformers import WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration

# Load the base tokenizer and model
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

# Define special tokens
special_tokens = [
    # Tokens without leading spaces
    "Waveform:", "Voices:", "Oscillator Detune:", "Filter Type:", "Filter Cutoff:",
    "ADSR Envelope:", "Attack:", "Decay:", "Sustain:", "Release:",
    "LFO Modulation:", "Hz", "ms", "dB", "s", ",", "\n", "-", ".", "None",
    # Tokens with leading spaces
    " Waveform:", " Voices:", " Oscillator Detune:", " Filter Type:", " Filter Cutoff:",
    " ADSR Envelope:", " Attack:", " Decay:", " Sustain:", " Release:",
    " LFO Modulation:", " Hz", " ms", " dB", " s", " None", " -"
]

# Add special tokens to the tokenizer
num_added_toks = tokenizer.add_tokens(special_tokens)
print(f"Added {num_added_toks} tokens")

# Resize the model's embeddings to accommodate new tokens
model.resize_token_embeddings(len(tokenizer))

# Create a processor with the updated tokenizer
processor = WhisperProcessor.from_pretrained(
    "openai/whisper-small",
    language="en",
    task="transcribe",
    tokenizer=tokenizer
)

# Save the tokenizer and processor
tokenizer.save_pretrained("/content/drive/MyDrive/whisper_synth_files/whisper_tokenizer_with_special_tokens")
processor.save_pretrained("/content/drive/MyDrive/whisper_synth_files/whisper_processor_with_special_tokens")
