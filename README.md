# WhisperSynth
Fine-Tuning Whisper-Small for Synthesizer Sound Matching

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

_Data Preparation_
* Input Data:
A dataset of .wav files was created, representing various synthesizer sounds. Each sound corresponds to a unique set of synthesizer settings.
* Output Data:
Each .wav file is paired with a .txt file that contains structured information about the synthesizer settings (e.g., oscillator type, frequency, envelope settings). The .txt files were automatically generated used a jupyter notebook (data_gen.ipynb).

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

Below are pictures of Ableton's interface, Serum's interface, and an example of the structured .txt data used for training:

Ableton:

Serum:

Sample .txt data:

_Tools and Frameworks_
* HuggingFace Transformers Library: For model handling and fine-tuning.
* PyTorch: Backend framework for training and customization.
* Ableton Live and Serum: Used to generate and organize synthesizer sounds.
* Jupyter Notebooks: For data generation, analysis, and experimentation.

Current Progress
This section outlines the progress made so far in the project, detailing the completed steps, challenges encountered, and insights gained.

Steps Completed
Dataset Creation:

Generated a small dataset of .wav files representing synthesizer sounds, paired with .txt files containing the corresponding synthesizer settings.
Organized the dataset into a structured format for incremental training and testing.
Model Preparation:

Selected the Whisper-Small model from HuggingFace as the base model.
Set up the training environment using HuggingFace Transformers and PyTorch.
Adapted the model to process audio inputs and generate text outputs in the desired .txt format.
Initial Fine-Tuning Attempts:

Began fine-tuning the model using the prepared dataset.
Addressed initial preprocessing challenges to align audio-text pairs with the model's input requirements.
Preliminary Evaluation:

Ran the model on test .wav files to observe output structure and identify formatting issues.
Identified areas where the model output diverges from the expected .txt structure.
Challenges Faced
Dataset Size and Quality:

Limited data availability for such a niche task.
Synthesizer settings need more diverse and representative samples to improve model generalization.
Model Adaptation:

Whisper-Small was not initially designed for this domain, requiring adjustments to both inputs and outputs.
Output .txt files occasionally lack the structured formatting necessary for practical use.
Training Performance:

Model fine-tuning has been computationally intensive, slowing iteration.
Early results show the need for better alignment between the input sound and output text.
Insights Gained
Fine-tuning a pre-trained model like Whisper-Small for a highly specific task highlights the need for a carefully curated dataset and precise output formatting.
Debugging and iterative testing are essential for adapting models to non-standard tasks.
Although the model is not fully functional yet, the process has provided valuable learnings about the adaptation of Transformers to audio-to-text tasks.
