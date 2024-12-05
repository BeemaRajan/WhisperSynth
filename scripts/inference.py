
import torch
import torchaudio
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer
)

# Path to your combined model directory
model_dir = "/content/drive/MyDrive/whisper_synth_files/whisper_synth"

# Load the tokenizer
tokenizer = WhisperTokenizer.from_pretrained(model_dir)

# Load the processor with the tokenizer
processor = WhisperProcessor.from_pretrained(
    model_dir,
    language="en",
    task="transcribe",
    tokenizer=tokenizer
)

# Load the combined model
model = WhisperForConditionalGeneration.from_pretrained(model_dir)

# Force the model to generate in English
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
    language="en", task="transcribe"
)

# Set model to evaluation mode and move to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Load and process the audio file
audio_path = "/content/drive/MyDrive/whisper_synth_files/data/inference/inference1.wav"  # Replace with the path to your .wav file
audio_array, sampling_rate = torchaudio.load(audio_path)

# Resample to 16 kHz if necessary
if sampling_rate != 16000:
    resampler = torchaudio.transforms.Resample(
        orig_freq=sampling_rate, new_freq=16000
    )
    audio_array = resampler(audio_array)

# If the audio has multiple channels, convert it to mono
if audio_array.shape[0] > 1:
    audio_array = torch.mean(audio_array, dim=0, keepdim=True)

audio_array = audio_array.squeeze().numpy()

# Process audio with the feature extractor to get input features
input_features = processor.feature_extractor(
    audio_array, sampling_rate=16000, return_tensors="pt"
).input_features.to(device)

# Generate synth_patch
with torch.no_grad():
    generated_ids = model.generate(input_features)

# Decode the generated IDs to get the synth_patch
synth_patch = processor.tokenizer.batch_decode(
    generated_ids, skip_special_tokens=True
)[0]
print("Synth Patch:", synth_patch)
