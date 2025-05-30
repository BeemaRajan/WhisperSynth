# WhisperSynth
**_Fine-Tuning Whisper-Small for Synthesizer Sound Matching: Is it possible?_**

My final fine-tuned model is included in the "releases" section, which can be accessed via tags.

## Table of Contents: 

1. [Project Overview](#project-overview)

2. [Approach](#approach)

3. [Progress](#progress)

4. [Critical Analysis](#critical-analysis)

5. [Future Scope](#future-scope)

6. [References & Other Resources](#references--other-resources)

7. [Acknowledgements](#acknowledgements)

## Project Overview

### Problem Statement:
Synthesizer sound matching is a challenging task in music production that involves recreating a specific synthesizer sound by determining the precise settings used to generate it. This process is often time-consuming and requires a high level of expertise.

This project aims to automate the task by fine-tuning a pre-trained Whisper-Small model from HuggingFace to process audio input (e.g., a synthesizer sound in .wav format) and generate corresponding synthesizer settings in a structured .txt format. The ultimate goal is to make sound matching more accessible and efficient for producers, sound designers, and musicians.

While the model is not functional, this repository documents the progress made, the challenges encountered, and a roadmap for future work. By tackling this problem, the project seeks to demonstrate the versatility of Transformers in niche applications beyond traditional natural language processing tasks.

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

With the increasing complexity of modern synthesizers and the demand for custom sound creation, automating sound matching has the potential to save valuable time and lower the barrier to entry for less experienced musicians. This project explores how Transformer-based models, specifically Whisper-Small, might be fine-tuned to bridge this gap by converting audio input into detailed synthesizer settings.

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

This project attempts to leverage the pre-trained Whisper-Small model from HuggingFace and fine-tune it for the task of synthesizer sound matching. The goal is to take a .wav file as input and output a structured .txt file containing the corresponding synthesizer settings. Below is an outline of the approach taken:

### Model Selection
_Why Whisper-Small?_
The Whisper-Small model, originally designed for speech-to-text tasks, has demonstrated exceptional performance in understanding audio input and generating meaningful text outputs. Its ability to process complex audio data makes it a suitable candidate for adapting to a niche domain like synthesizer sound matching, which is a complex classification task.

_Whisper-Small Architechure:_

In examining the Architechture, we can see that the model itself isn't complex or unlike previous Encoder-Decoder models.

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

Log-Mel Spectrogram:

![Picture of Log-Mel Spectrogram](images/spectrogram_image.png)

Challenges:

* Limited availability of pre-labeled data in this domain.
* Need for consistent formatting of .txt files for effective training.
* Lack of automated data generation.

Below are pictures of Ableton's interface, Serum's interface, and an example of the structured .txt data used for fine-tuning:

Ableton:

![Abelton](images/ableton_image.png)

Serum:

![Serum](images/serum_image.png)

Sample .txt data:

![.txt data](images/sound100_image.png)

_Fine-Tuning the Model_
* Dataset Formatting:
The dataset was preprocessed to align with Whisper-Small's input requirements, ensuring audio and text pairings were appropriately structured.
* Training Setup:
Leveraged HuggingFace's Transformers library and utilized LoRA for fine-tuning.
* Evaluation Metrics:
Initial evaluation focuses on the syntactic correctness of the output .txt files.
Further evaluation incorporates metrics for accuracy in matching the intended sound.
* Status
Fine-tuning has presented challenges encountered in aligning the output format to the desired .txt structure with speculation that Whisper-small might not be suitable for this task.
While defining domain-specific tokens yielded greater training loss, it has not been significant enough for the model to accurately output the structured data.

## Progress
This section outlines the progress made in the project, detailing the completed steps, challenges encountered, and insights gained.

_Steps Completed_

Dataset Creation:

Generated a small dataset (200) .wav files representing synthesizer sounds, paired with .txt files containing the corresponding synthesizer settings.
Organized the dataset into a structured format for incremental training and testing.

_Model Preparation:_

* Selected the Whisper-Small model from HuggingFace as the base model.
* Set up the training environment using HuggingFace Transformers, PyTorch, and Colab.
* Adapted the model to process audio inputs, added specialized tokens for the task, and generate text outputs in the desired .txt format.

_Fine-Tuning Attempts:_

* Began fine-tuning the model using the prepared dataset using LoRA.
* Addressed initial preprocessing challenges to align audio-text pairs with the model's input requirements.

_Preliminary Evaluation:_

* Ran the model on test .wav files to observe output structure and identify formatting issues.
* Loss demonstrates unsuccessful learning for the model as outputs prove challenges with structure.

_Challenges Faced_
* Dataset Size and Quality:
Limited data availability for such a niche task.

* Serum file formats are proprietary and difficult to replicate.

* Model Adaptation: Whisper-Small was not initially designed for this domain, requiring adjustments to outputs.
Output .txt files lack the structured formatting necessary for practical use.

**_Training Performance:_**

Results from finetune_whisper.py

| Epoch | Loss   | Grad Norm       | Learning Rate      |
|-------|--------|-----------------|--------------------|
| 0.25  | 10.2973 | 11.1445         | 2.9325e-05         |
| 0.5   | 9.275   | 7.9859          | 2.8575e-05         |
| 0.75  | 8.6888  | 9.8403          | 2.805e-05          |
| 1.0   | 8.2177  | 4.9978          | 2.730e-05          |
| 1.25  | 7.9385  | 5.3561          | 2.655e-05          |
| 1.5   | 7.5249  | 4.8783          | 2.58e-05           |
| 1.75  | 7.4684  | 4.9382          | 2.505e-05          |
| 2.0   | 7.1981  | 4.6030          | 2.43e-05           |
| 2.25  | 7.0305  | 5.1984          | 2.355e-05          |
| 2.5   | 6.8372  | 5.2830          | 2.28e-05           |
| 2.75  | 6.6826  | 8.1163          | 2.205e-05          |
| 3.0   | 6.3609  | 8.0886          | 2.13e-05           |
| 3.25  | 6.296   | 5.7906          | 2.055e-05          |
| 3.5   | 6.2891  | 11.0990         | 1.98e-05           |
| 3.75  | 6.209   | 11.7580         | 1.905e-05          |
| 4.0   | 6.1502  | 19.7905         | 1.83e-05           |
| 4.25  | 6.0496  | 15.9363         | 1.755e-05          |
| 4.5   | 6.092   | 7.5143          | 1.68e-05           |
| 4.75  | 6.0616  | 15.5779         | 1.605e-05          |
| 5.0   | 5.9329  | 10.7182         | 1.53e-05           |
| 5.25  | 5.9926  | 14.4724         | 1.455e-05          |
| 5.5   | 6.0027  | 13.2221         | 1.38e-05           |
| 5.75  | 5.855   | 8.9715          | 1.305e-05          |
| 6.0   | 5.8312  | 7.1103          | 1.23e-05           |
| 6.25  | 5.8531  | 7.5813          | 1.155e-05          |
| 6.5   | 5.8525  | 7.6279          | 1.08e-05           |
| 6.75  | 5.8466  | 9.1781          | 1.005e-05          |
| 7.0   | 5.8164  | 7.6931          | 9.3e-06            |
| 7.25  | 5.7546  | 10.8081         | 8.55e-06           |
| 7.5   | 5.8458  | 10.6714         | 7.8e-06            |
| 7.75  | 5.7898  | 8.0403          | 7.05e-06           |
| 8.0   | 5.7788  | 7.8717          | 6.3e-06            |
| 8.25  | 5.753   | 8.7981          | 5.55e-06           |
| 8.5   | 5.7186  | 10.0870         | 4.8e-06            |
| 8.75  | 5.8041  | 9.0158          | 4.05e-06           |
| 9.0   | 5.7663  | 7.9293          | 3.3e-06            |
| 9.25  | 5.7493  | 7.0522          | 2.55e-06           |
| 9.5   | 5.6538  | 8.3509          | 1.8e-06            |
| 9.75  | 5.7924  | 8.3206          | 1.05e-06           |
| 10.0  | 5.7762  | 8.3964          | 3.0e-07            |
| **Final** | **Train Loss:** 6.4708 |

While changing the learning rate (from 5e-5 to 3e-5) and number of epochs (from 3 to 10) yielded slightly more training loss, the fluctuation in the training loss indicates that the model may not be learning the structured data correctly.

Results from Evaluation script (eval.py):

```python
---
Prediction: んんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんんん
Reference: Waveform: Triangle
Voices: 3
Oscillator Detune: 0.30
Filter Type: N/A
Filter Cutoff: N/A
ADSR Envelope: Attack: 85ms, Decay: 243ms, Sustain: -6.3 dB, Release: 156ms
LFO Modulation: None
---
Prediction:  ḍᵗᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦ
Reference: Waveform: Square
Voices: 1
Oscillator Detune: None
Filter Type: Lowpass
Filter Cutoff: 10000Hz
ADSR Envelope: Attack: 500ms, Decay: 1.00s, Sustain: -6 dB, Release: 300ms
LFO Modulation: None
---

{'eval_loss': 7.573992729187012, 'eval_model_preparation_time': 0.006, 'eval_exact_match_accuracy': 0.0, 'eval_Waveform': 0.0, 'eval_Voices': 0.0, 'eval_Oscillator Detune': 0.0, 'eval_Filter Type': 0.0, 'eval_Filter Cutoff': 0.0, 'eval_ADSR Envelope': 0.0, 'eval_LFO Modulation': 0.0, 'eval_runtime': 70.4598, 'eval_samples_per_second': 0.568, 'eval_steps_per_second': 0.142}
```

Clearly, the output is not what we would want to expect. However, this output is different than the model's output prior to adding task-specific tokens:

```python
---
Prediction:  you
Reference: Waveform: Square
Voices: 1
Oscillator Detune: None
Filter Type: Lowpass
Filter Cutoff: 10000Hz
ADSR Envelope: Attack: 500ms, Decay: 1.00s, Sustain: -6 dB, Release: 300ms
LFO Modulation: None
---
```

A potential positive outcome is that the model may be identifying the repetition of the same sound throughout the entire audio clip. For example, the output:

```python
ḍᵗᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦᶦ
```

could indicate that a transient (beginning of sound) was detected followed by the remainder of the synthesizer sound. This may be an optimistic take, but regardless, the outputs lack any similarity to the .txt files, indicating that this model is likely unsuitable for the task.

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

**_Lessons Learned_**

Model Adaptation for Niche Tasks:

* Fine-tuning large, general-purpose models for highly specific tasks can require significant modifications, including task-specific loss functions, evaluation metrics, and data preprocessing pipelines.

The Importance of Data Quality and Size:

* For niche tasks, creating a high-quality, domain-specific dataset is just as important as model architecture. The lack of sufficient data can be a bottleneck for performance.

Iterative Experimentation is Key:

* Each fine-tuning attempt has provided valuable insights into how the model interprets audio inputs and structured outputs, underscoring the need for iterative testing and debugging to refine the approach.

## Future Scope

In the long term, this project opens up several opportunities for further exploration and improvement. Below are the key areas I would like to focus on:

Understanding and Automating .fxp File Generation

* A deeper understanding of the .fxp file format used in Serum is critical. By reverse-engineering these files, I aim to automate the data generation process, enabling the creation of structured .fxp outputs directly from the model.
* This would streamline the workflow and bring the project closer to practical use in music production.

Exploration of Custom Tokens for Structured Outputs in Encoder-Decoder Models

* I plan to invest more time in designing custom tokens or formatting strategies to better align the model's outputs with the required structured format.
* This could involve defining specific tokenization rules or using post-processing steps to enhance the output consistency.

Experimenting with Different Models

* While Whisper-Small serves as a foundation, evaluating alternative models (e.g., T5, GPT-based models, or audio-specific architectures) may yield better results for this task.
* Comparing performance across different architectures will help identify the best approach for generating structured synthesizer settings.

Establishing new ways to Analyze Audio

* While Whisper relies on log-mel spectrograms for audio analysis, exploring alternative approaches such as wavelet transforms, learned audio embeddings, or raw waveform analysis may improve the model's ability to extract relevant features for this specific task.
* This could be a pivotal step in addressing the limitations of log-mel spectrograms by providing new and potentially more informative ways to analyze audio.

## References & Other Resources
[Whisper-Small (HuggingFace)](https://huggingface.co/openai/whisper-small)

[Synthesizer Sound Matching Using Audio Spectrogram Transformers](https://arxiv.org/pdf/2407.16643)

[Formal Algorithms for Transformers](https://arxiv.org/pdf/2207.09238)

[Generating Musical Synthesizer Patches with Machine Learning](https://jakespracher.medium.com/generating-musical-synthesizer-patches-with-machine-learning-c52f66dfe751)

[Understanding Log-Mel Spectrogram](https://medium.com/analytics-vidhya/understanding-the-mel-spectrogram-fca2afa2ce53)

[Serum Website](https://xferrecords.com/products/serum/)

## Acknowledgements
_Tools and Frameworks_
* HuggingFace Transformers Library: For model handling and fine-tuning.
* PyTorch: Backend framework for training and customization.
* Ableton Live & Serum: Used to generate and organize synthesizer sounds.
* Jupyter Notebooks: For data generation, analysis, and experimentation.
