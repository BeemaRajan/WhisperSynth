
from transformers import WhisperTokenizer

# Load the tokenizer with added special tokens
tokenizer = WhisperTokenizer.from_pretrained("/content/drive/MyDrive/whisper_synth_files/whisper_tokenizer_with_special_tokens")

# Test text
test_text = "Sustain: -12.3 dB"

# Tokenize the test text
tokens = tokenizer.tokenize(test_text)
print("Tokens:", tokens)

# Optionally, print token IDs
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print("Token IDs:", token_ids)
