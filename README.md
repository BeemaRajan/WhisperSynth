# WhisperSynth
_Fine-Tuning Whisper-Small for Synthesizer Sound Matching_

## Table of Contents: 

## Project Overview

### Problem Statement:
Synthesizer sound matching is a challenging task in music production that involves recreating a specific synthesizer sound by determining the precise settings used to generate it. This process is often time-consuming and requires a high level of expertise.

This project aims to automate the task by fine-tuning a pre-trained Whisper-Small model from HuggingFace to process audio input (e.g., a synthesizer sound in .wav format) and generate corresponding synthesizer settings in a structured .txt format. The ultimate goal is to make sound matching more accessible and efficient for producers, sound designers, and musicians.

While the model is not fully functional yet, this repository documents the progress made so far, the challenges encountered, and a roadmap for future work. By tackling this problem, the project seeks to demonstrate the versatility of Transformers in niche applications beyond traditional natural language processing tasks.

### Synthesizers and Their Role in Music:

_What are Synthesizers?_

Synthesizers are electronic musical instruments that generate sound by manipulating audio signals. Unlike traditional instruments, which produce sound through physical means (like strings or air columns), synthesizers create sound through oscillators, filters, envelopes, and modulators. This allows them to produce a virtually infinite range of sounds, from realistic emulations of acoustic instruments to entirely unique, otherworldly tones.

_How Are Synthesizers Used in Music?_

Synthesizers are foundational tools in modern music production, appearing across almost every genre, from electronic dance music (EDM) and hip-hop to rock and pop. They are used to:

* Create Unique Sounds: Producers craft custom tones, such as leads, pads, or basslines, to give their music a distinct sonic identity.
* Recreate Classic Sounds: Synthesizers can emulate iconic sounds from classic records, adding nostalgia or familiarity to a track.
* Enhance Composition: Synthesizers provide endless possibilities for experimentation, helping artists push creative boundaries.

Whether creating lush, ambient textures or aggressive, cutting-edge soundscapes, synthesizers are essential for shaping the sound of modern music.

### Context

Sound design is a critical aspect of music production, where producers and sound designers use synthesizers to craft unique audio textures and tones. Despite advancements in technology, the process of replicating or matching a synthesizer sound remains highly manual and requires significant expertise. Producers often rely on trial and error to recreate a sound they hear, adjusting parameters such as oscillators, filters, and envelopes until the desired tone is achieved. This process can be time-consuming and is not always precise.

With the increasing complexity of modern synthesizers and the demand for custom sound creation, automating sound matching has the potential to save valuable time and lower the barrier to entry for less experienced musicians. This project explores how Transformer-based models, specifically Whisper-Small, can be fine-tuned to bridge this gap by converting audio input into detailed synthesizer settings.

### Project Impact
By automating the process of synthesizer sound matching, this project could provide the following benefits:

1. Streamlined Workflow:
Producers and sound designers can quickly match and replicate sounds without the need for extensive manual adjustments, allowing them to focus on creative aspects of music production.

2. Accessibility:
Aspiring musicians and beginners without deep technical knowledge of synthesizers can easily recreate sounds, democratizing sound design.

3. Educational Tool:
The project could serve as a learning aid, helping users understand how specific settings impact the sound of a synthesizer.

4. Creative Applications:
Automated sound matching could open up new possibilities for generative music production, where AI assists artists in discovering and experimenting with novel soundscapes.

This project represents a step toward integrating AI and music production tools, potentially transforming how producers interact with synthesizers and enhancing the creative process in the music industry.

## Approach

This project leverages the pre-trained Whisper-Small model from HuggingFace and fine-tunes it for the task of synthesizer sound matching. The goal is to take a .wav file as input and output a structured .txt file containing the corresponding synthesizer settings. Below is an outline of the approach taken:

### Model Selection
_Why Whisper-Small?_
The Whisper-Small model, originally designed for speech-to-text tasks, has demonstrated exceptional performance in understanding audio input and generating meaningful text outputs. Its ability to process complex audio data makes it a suitable candidate for adapting to a niche domain like sound matching.

_Whisper-Small Architechure:_
```python
WhisperForConditionalGeneration(
  (model): WhisperModel(
    (encoder): WhisperEncoder(
      (conv1): Conv1d(80, 768, kernel_size=(3,), stride=(1,), padding=(1,))
      (conv2): Conv1d(768, 768, kernel_size=(3,), stride=(2,), padding=(1,))
      (embed_positions): Embedding(1500, 768)
      (layers): ModuleList(
        (0-11): 12 x WhisperEncoderLayer(
          (self_attn): WhisperSdpaAttention(
            (k_proj): Linear(in_features=768, out_features=768, bias=False)
            (v_proj): Linear(in_features=768, out_features=768, bias=True)
            (q_proj): Linear(in_features=768, out_features=768, bias=True)
            (out_proj): Linear(in_features=768, out_features=768, bias=True)
          )
          (self_attn_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (activation_fn): GELUActivation()
          (fc1): Linear(in_features=768, out_features=3072, bias=True)
          (fc2): Linear(in_features=3072, out_features=768, bias=True)
          (final_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        )
      )
      (layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
    )
    (decoder): WhisperDecoder(
      (embed_tokens): Embedding(51865, 768, padding_idx=50257)
      (embed_positions): WhisperPositionalEmbedding(448, 768)
      (layers): ModuleList(
        (0-11): 12 x WhisperDecoderLayer(
          (self_attn): WhisperSdpaAttention(
            (k_proj): Linear(in_features=768, out_features=768, bias=False)
            (v_proj): Linear(in_features=768, out_features=768, bias=True)
            (q_proj): Linear(in_features=768, out_features=768, bias=True)
            (out_proj): Linear(in_features=768, out_features=768, bias=True)
          )
          (activation_fn): GELUActivation()
          (self_attn_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (encoder_attn): WhisperSdpaAttention(
            (k_proj): Linear(in_features=768, out_features=768, bias=False)
            (v_proj): Linear(in_features=768, out_features=768, bias=True)
            (q_proj): Linear(in_features=768, out_features=768, bias=True)
            (out_proj): Linear(in_features=768, out_features=768, bias=True)
          )
          (encoder_attn_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (fc1): Linear(in_features=768, out_features=3072, bias=True)
          (fc2): Linear(in_features=3072, out_features=768, bias=True)
          (final_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        )
      )
      (layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
    )
  )
  (proj_out): Linear(in_features=768, out_features=51865, bias=False)
)
```

**Architechture in Pseudocode:** 
```pseudo
Algorithm: P ← WhisperModel(x, z | θ)
/* Encoder-Decoder Transformer forward pass for audio-to-text */
Input: 
  x: Input audio features (sequence of token IDs)
  z: Target sequence of token IDs (for training)
Output: 
  P: Conditional probabilities for the target sequence tokens
Hyperparameters:
  L_enc, L_dec: Number of layers in encoder and decoder
  d_model: Model embedding dimension
  d_ff: Feedforward hidden dimension

Parameters:
  θ includes:
    W_embed, W_pos: Token and positional embedding matrices
    Encoder:
      W_enc, LayerNorm params, Attention params, MLP params
    Decoder:
      W_dec, Cross-Attention params, Attention params, MLP params

---

/* Encoder: Process the input audio sequence */
1. Encode input sequence:
   z_enc ← length(x)
   e_enc ← x[t] × W_embed + W_pos[t]
   for l = 1 to L_enc:
       /* Multi-head self-attention */
       Z ← SelfAttention(e_enc, W_enc, Mask=1)
       /* Feedforward network with LayerNorm */
       Z ← LayerNorm(GELU(Z × W_ff1 + b_ff1) × W_ff2 + b_ff2)
       /* Update e_enc */
       e_enc ← LayerNorm(Z)

/* Decoder: Process the target sequence and incorporate context */
2. Decode target sequence:
   z_dec ← length(z)
   e_dec ← z[t] × W_embed + W_pos[t]
   for l = 1 to L_dec:
       /* Multi-head self-attention for decoder */
       X ← SelfAttention(e_dec, W_dec, Mask=1)
       /* Cross-attention with encoder output */
       X ← CrossAttention(X, Z, W_cross, Mask=1)
       /* Feedforward network with LayerNorm */
       X ← LayerNorm(GELU(X × W_ff1 + b_ff1) × W_ff2 + b_ff2)
       /* Update e_dec */
       e_dec ← LayerNorm(X)

/* Output probabilities */
3. Compute final probabilities:
   P ← softmax(e_dec × W_proj_out)
```

_Data Preparation_
* Input Data:
A dataset of .wav files was created, representing various synthesizer sounds. Each sound corresponds to a unique set of synthesizer settings.
* Output Data:
Each .wav file is paired with a .txt file that contains structured information about the synthesizer settings (e.g., oscillator type, frequency, envelope settings). The .txt files were automatically generated using a Jupyter notebook (data_gen.ipynb). My original goal was to have the model output in a .fxp file format (used by Serum); however, due to challenges in reverse-engineering the .fxp format, this has not been feasible (yet).
* Audio Analysis with Whisper:
Whisper processes audio files by converting them into log-mel spectrograms, a feature representation that captures the frequency and energy distribution of audio signals over time. This transformation serves as the input for Whisper's encoder and allows the model to analyze audio data effectively.

[Picture of log-mel spectrogram]

Challenges:

* Limited availability of pre-labeled data in this domain.
* Need for consistent formatting of .txt files for effective training.
* Lack of automated data generation.

_Fine-Tuning the Model_
* Dataset Formatting:
The dataset was preprocessed to align with Whisper-Small's input requirements, ensuring audio and text pairings were appropriately structured.
* Training Setup:
Leveraged HuggingFace's Transformers library for fine-tuning.
Used a hybrid approach to gradually introduce additional training data.
* Evaluation Metrics:
Initial evaluation focuses on the syntactic correctness of the output .txt files.
Future evaluation will incorporate metrics for accuracy in matching the intended sound.
* Current Status
Fine-tuning is in progress, with challenges encountered in aligning the output format to the desired .txt structure.
Debugging and optimization are ongoing to improve model performance and output quality.

Below are pictures of Ableton's interface, Serum's interface, and an example of the structured .txt data used for fine-tuning:

Ableton:

Serum:

Sample .txt data:

_Tools and Frameworks_
* HuggingFace Transformers Library: For model handling and fine-tuning.
* PyTorch: Backend framework for training and customization.
* Ableton Live and Serum: Used to generate and organize synthesizer sounds.
* Jupyter Notebooks: For data generation, analysis, and experimentation.

### Current Progress
This section outlines the progress made so far in the project, detailing the completed steps, challenges encountered, and insights gained.

_Steps Completed_

Dataset Creation:

Generated a small dataset (~200) .wav files representing synthesizer sounds, paired with .txt files containing the corresponding synthesizer settings.
Organized the dataset into a structured format for incremental training and testing.

_Model Preparation:_

* Selected the Whisper-Small model from HuggingFace as the base model.
* Set up the training environment using HuggingFace Transformers, PyTorch, and Colab.
* Adapted the model to process audio inputs and generate text outputs in the desired .txt format.

_Initial Fine-Tuning Attempts:_

* Began fine-tuning the model using the prepared dataset using LoRA.
* Addressed initial preprocessing challenges to align audio-text pairs with the model's input requirements.

_Preliminary Evaluation:_

* Ran the model on test .wav files to observe output structure and identify formatting issues.
* Loss demonstrates successful learning for the model, yet outputs prove challenges with structure.

_Challenges Faced_
* Dataset Size and Quality:
Limited data availability for such a niche task.
* Synthesizer settings need more diverse and representative samples to improve model generalization.

* Serum file formats are proprietary and difficult to replicate.

* Model Adaptation:

Whisper-Small was not initially designed for this domain, requiring adjustments to outputs.
Output .txt files lack the structured formatting necessary for practical use.

**_Training Performance:_**

```python
{'loss': 2.5115, 'grad_norm': 15.277137756347656, 'learning_rate': 2.9475e-05, 'epoch': 0.25}
{'loss': 1.3586, 'grad_norm': 1.9794156551361084, 'learning_rate': 2.8725e-05, 'epoch': 0.5}
{'loss': 1.1712, 'grad_norm': 1.3977088928222656, 'learning_rate': 2.7975e-05, 'epoch': 0.75}
{'loss': 1.0025, 'grad_norm': 1.029678463935852, 'learning_rate': 2.7225e-05, 'epoch': 1.0}
{'loss': 0.8528, 'grad_norm': 1.0061405897140503, 'learning_rate': 2.6475e-05, 'epoch': 1.25}
{'loss': 0.7281, 'grad_norm': 0.9967676401138306, 'learning_rate': 2.5725000000000002e-05, 'epoch': 1.5}
{'loss': 0.6174, 'grad_norm': 1.0779485702514648, 'learning_rate': 2.4975e-05, 'epoch': 1.75}
{'loss': 0.5238, 'grad_norm': 1.2355623245239258, 'learning_rate': 2.4225e-05, 'epoch': 2.0}
{'loss': 0.4368, 'grad_norm': 0.7758647799491882, 'learning_rate': 2.3475e-05, 'epoch': 2.25}
{'loss': 0.3542, 'grad_norm': 0.7342891693115234, 'learning_rate': 2.2725e-05, 'epoch': 2.5}
{'loss': 0.3161, 'grad_norm': 0.7141870856285095, 'learning_rate': 2.1975000000000002e-05, 'epoch': 2.75}
{'loss': 0.2823, 'grad_norm': 0.7133969664573669, 'learning_rate': 2.1225e-05, 'epoch': 3.0}
{'loss': 0.2659, 'grad_norm': 0.6530483961105347, 'learning_rate': 2.0475e-05, 'epoch': 3.25}
{'loss': 0.2491, 'grad_norm': 0.7656163573265076, 'learning_rate': 1.9725e-05, 'epoch': 3.5}
{'loss': 0.2278, 'grad_norm': 0.5166074633598328, 'learning_rate': 1.8975e-05, 'epoch': 3.75}
{'loss': 0.2255, 'grad_norm': 0.6514007449150085, 'learning_rate': 1.8225000000000003e-05, 'epoch': 4.0}
{'loss': 0.2175, 'grad_norm': 0.4755321443080902, 'learning_rate': 1.7475e-05, 'epoch': 4.25}
{'loss': 0.2159, 'grad_norm': 0.5834320783615112, 'learning_rate': 1.6725e-05, 'epoch': 4.5}
{'loss': 0.2082, 'grad_norm': 0.7452165484428406, 'learning_rate': 1.5975e-05, 'epoch': 4.75}
{'loss': 0.2097, 'grad_norm': 0.7459815740585327, 'learning_rate': 1.5224999999999999e-05, 'epoch': 5.0}
{'loss': 0.2029, 'grad_norm': 0.6532655358314514, 'learning_rate': 1.4475e-05, 'epoch': 5.25}
{'loss': 0.1958, 'grad_norm': 0.6958538889884949, 'learning_rate': 1.3725000000000002e-05, 'epoch': 5.5}
{'loss': 0.2072, 'grad_norm': 0.5873009562492371, 'learning_rate': 1.2975e-05, 'epoch': 5.75}
{'loss': 0.19, 'grad_norm': 0.639061689376831, 'learning_rate': 1.2224999999999999e-05, 'epoch': 6.0}
{'loss': 0.1967, 'grad_norm': 0.7394694089889526, 'learning_rate': 1.1475000000000001e-05, 'epoch': 6.25}
{'loss': 0.1982, 'grad_norm': 1.0083016157150269, 'learning_rate': 1.0725e-05, 'epoch': 6.5}
{'loss': 0.1879, 'grad_norm': 0.6625001430511475, 'learning_rate': 9.975e-06, 'epoch': 6.75}
{'loss': 0.1832, 'grad_norm': 0.7143978476524353, 'learning_rate': 9.225e-06, 'epoch': 7.0}
{'loss': 0.1864, 'grad_norm': 0.6730558276176453, 'learning_rate': 8.475e-06, 'epoch': 7.25}
{'loss': 0.189, 'grad_norm': 0.914880633354187, 'learning_rate': 7.725e-06, 'epoch': 7.5}
{'loss': 0.1832, 'grad_norm': 0.6366786360740662, 'learning_rate': 6.975e-06, 'epoch': 7.75}
{'loss': 0.1879, 'grad_norm': 0.6175647974014282, 'learning_rate': 6.225e-06, 'epoch': 8.0}
{'loss': 0.1833, 'grad_norm': 0.7860107421875, 'learning_rate': 5.475e-06, 'epoch': 8.25}
{'loss': 0.1875, 'grad_norm': 0.6213966012001038, 'learning_rate': 4.7250000000000005e-06, 'epoch': 8.5}
{'loss': 0.176, 'grad_norm': 0.5397642254829407, 'learning_rate': 3.975e-06, 'epoch': 8.75}
{'loss': 0.1867, 'grad_norm': 0.6178697347640991, 'learning_rate': 3.225e-06, 'epoch': 9.0}
{'loss': 0.1801, 'grad_norm': 0.5428252816200256, 'learning_rate': 2.475e-06, 'epoch': 9.25}
{'loss': 0.1841, 'grad_norm': 0.5276110768318176, 'learning_rate': 1.7250000000000002e-06, 'epoch': 9.5}
{'loss': 0.1784, 'grad_norm': 0.8070985078811646, 'learning_rate': 9.75e-07, 'epoch': 9.75}
{'loss': 0.1846, 'grad_norm': 0.6955456733703613, 'learning_rate': 2.25e-07, 'epoch': 10.0}
{'train_runtime': 326.3445, 'train_samples_per_second': 4.903, 'train_steps_per_second': 1.226, 'train_loss': 0.3935928952693939, 'epoch': 10.0}
```

Results from Evaluation script (eval.py):

```python
Prediction:  you
Reference: Waveform: Sine
Voices: 1
Oscillator Detune: None
Filter Type: Highpass
Filter Cutoff: 3000Hz
ADSR Envelope: Attack: 400ms, Decay: 900ms, Sustain: -12 dB, Release: 600ms
LFO Modulation: None
---
Prediction:  you
Reference: Waveform: Saw
Voices: 3
Oscillator Detune: 0.10
Filter Type: N/A
Filter Cutoff: N/A
ADSR Envelope: Attack: 300ms, Decay: 1.00s, Sustain: -12 dB, Release: 500ms
LFO Modulation: None
---
Prediction:  you
Reference: Waveform: Square
Voices: 1
Oscillator Detune: None
Filter Type: N/A
Filter Cutoff: N/A
ADSR Envelope: Attack: 150ms, Decay: 1.50s, Sustain: -3 dB, Release: 200ms
LFO Modulation: None
---
Prediction:  you
Reference: Waveform: Square
Voices: 1
Oscillator Detune: None
Filter Type: Bandpass
Filter Cutoff: 368Hz
ADSR Envelope: Attack: 244ms, Decay: 583ms, Sustain: -0.6 dB, Release: 894ms
LFO Modulation: None
---
```

## Critical Analysis

**_Challenges Identified:_**

Mismatch Between Model Architecture and Task Requirements:
* Whisper-Small was designed for speech-to-text tasks, where the output space is a natural language. Synthesizer sound matching, however, requires highly structured, domain-specific outputs, which Whisper-Small may not inherently support.
* The model struggles to generalize from audio features to structured .txt representations, indicating a need for task-specific architecture modifications or fine-tuning strategies.

Dataset Constraints:

* The limited dataset size is likely contributing to poor model performance. A larger dataset with numerous examples of .wav to .txt mappings is critical for improving generalization.
* Synthesizer settings often have a complex relationship with the audio output, making it challenging for the model to infer precise settings without extensive training data.

Evaluation Challenges:

* Current evaluation metrics may not effectively capture the accuracy of structured outputs, especially when slight deviations (e.g., an incorrect filter type) render the result invalid for practical use.
* A new task-specific evaluation metric could be developed to assess alignment between predictions and references more accurately.

**_Potential Improvements:_**

Custom Loss Function:

* Implement a loss function tailored to this task, such as one that penalizes deviations in structured fields (e.g., waveform type, ADSR envelope values) more heavily than general mismatches.

Domain-Specific Preprocessing:

* Augment the input features with task-specific embeddings or descriptors (e.g., pre-calculated audio features such as spectral centroid, harmonic content) to help the model better correlate audio characteristics with structured settings.

Alternative Architectures:

* Explore architectures better suited for structured output generation, such as sequence-to-sequence models with constrained decoding mechanisms or multimodal models that explicitly link audio features with textual representations.

Transfer Learning and Pretraining:

* Pretrain the model on a larger corpus of audio-to-structured-text mappings, such as MIDI files or pre-labeled synthesizer datasets, to give it a foundational understanding before fine-tuning on the specific .wav to .txt task.

_Lessons Learned_

Model Adaptation for Niche Tasks:

*Fine-tuning large, general-purpose models for highly specific tasks can require significant modifications, including task-specific loss functions, evaluation metrics, and data preprocessing pipelines.

The Importance of Data Quality and Size:

* For niche tasks, creating a high-quality, domain-specific dataset is just as important as model architecture. The lack of sufficient data can be a bottleneck for performance.

Iterative Experimentation is Key:

* Each fine-tuning attempt has provided valuable insights into how the model interprets audio inputs and structured outputs, underscoring the need for iterative testing and debugging to refine the approach.

## Future Scope

In the long term, this project opens up several opportunities for further exploration and improvement. Below are the key areas I would like to focus on:

Understanding and Automating .fxp File Generation

* A deeper understanding of the .fxp file format used in Serum is critical. By reverse-engineering these files, I aim to automate the data generation process, enabling the creation of structured .fxp outputs directly from the model.
* This would streamline the workflow and bring the project closer to practical use in music production.

Exploration of Custom Tokens for Structured Outputs

* I plan to invest more time in designing custom tokens or formatting strategies to better align the model's outputs with the required structured format.
* This could involve defining specific tokenization rules or using post-processing steps to enhance the output consistency.

Experimenting with Different Models

* While Whisper-Small serves as a foundation, evaluating alternative models (e.g., T5, GPT-based models, or audio-specific architectures) may yield better results for this task.
* Comparing performance across different architectures will help identify the best approach for generating structured synthesizer settings.

Optimization of the Training Pipeline

* Improving the training process through enhanced loss functions, task-specific evaluation metrics, or more robust preprocessing pipelines will be a focus for future iterations.

Scaling the Dataset

* Expanding the dataset to include a larger and more diverse set of .wav files with corresponding settings will significantly improve model performance and generalization.

Interactive and Real-Time Applications

* Exploring Alternative Audio Analysis Methods

While Whisper relies on log-mel spectrograms for audio analysis, exploring alternative approaches such as wavelet transforms, learned audio embeddings, or raw waveform analysis may improve the model's ability to extract relevant features for this specific task.

## References & Other Resources
[Synthesizer Sound Matching Using Audio Spectrogram Transformers](https://arxiv.org/pdf/2407.16643)
[Formal Algorithms for Transformers](https://arxiv.org/pdf/2207.09238)
[Serum Product Manual]
[Generating Musical Synthesizer Patches with Machine Learning](https://jakespracher.medium.com/generating-musical-synthesizer-patches-with-machine-learning-c52f66dfe751)

## Acknowledgements
