import os
import random

def create_synth_txt_files(directory, num_files):
    # Create directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Find the highest existing sound{i}.txt file
    existing_files = [f for f in os.listdir(directory) if f.startswith("sound") and f.endswith(".txt")]
    highest_index = 0
    for file in existing_files:
        try:
            index = int(file[5:-4])
            if index > highest_index:
                highest_index = index
        except ValueError:
            continue
    
    # Define parameter options and ranges
    waveforms = ["Sine", "Saw", "Square", "Triangle", "Pulse"]
    voices_range = (1, 8)  # Number of voices between 1 and 8
    detune_range = (0.0, .5)
    filter_types = ["Lowpass", "Highpass", "Bandpass", "N/A"]
    cutoff_range = (20, 10000)  # Hz
    adsr_ranges = {
        "Attack": (0.0, 1000.0),     # ms
        "Decay": (0.0, 1000.0),      # ms
        "Sustain": (-12.0, 0.0),     # dB
        "Release": (0.0, 1000.0)     # ms
    }
    lfo_modulations = ["None"]

    # Start creating new files from the highest index + 1
    start_index = highest_index + 1
    for i in range(start_index, start_index + num_files):
        ## Generate random values for each parameter

        # Waveform
        waveform = random.choice(waveforms)

        # Voices
        if random.random() < 0.2: # 20% chance of 1
            voices = 1
        else:
            voices = random.randint(*voices_range)

        # Oscillator Detune
        if voices == 1: 
            oscillator_detune = "None"
        else:
            detune_value = random.uniform(*detune_range)
            oscillator_detune = f"{detune_value:.2f}"  # Always show two decimal places

        # Filter Type
        if random.random() < 0.2: 
            filter_type = "N/A"
        else:
            filter_type = random.choice(filter_types)

        # Filter Cutoff
        if filter_type == "N/A":
            filter_cutoff = "N/A"
        else:
            filter_cutoff_value = random.randint(*cutoff_range)
            filter_cutoff = f"{filter_cutoff_value}Hz"

        # ADSR Envelope

        # Attack
        attack_value = random.uniform(*adsr_ranges['Attack'])
        if attack_value >= 1000:
            attack_sec = attack_value / 1000
            attack_formatted = f"{round(attack_sec, 2)}s"
        else:
            attack_formatted = f"{int(round(attack_value))}ms"

        # Decay
        decay_value = random.uniform(*adsr_ranges['Decay'])
        if decay_value >= 1000:
            decay_sec = decay_value / 1000
            decay_formatted = f"{round(decay_sec, 2)}s"
        else:
            decay_formatted = f"{int(round(decay_value))}ms"

        # Sustain
        sustain_value = random.uniform(*adsr_ranges['Sustain'])
        sustain_formatted = f"{round(sustain_value, 1)} dB"

        # Release
        release_value = random.uniform(*adsr_ranges['Release'])
        if release_value >= 1000:
            release_sec = release_value / 1000
            release_formatted = f"{round(release_sec, 2)}s"
        else:
            release_formatted = f"{int(round(release_value))}ms"

        # LFO Modulation
        lfo_modulation = random.choice(lfo_modulations)

        # Prepare the content to write to the file
        content = f"""Waveform: {waveform}
Voices: {voices}
Oscillator Detune: {oscillator_detune}
Filter Type: {filter_type}
Filter Cutoff: {filter_cutoff}
ADSR Envelope: Attack: {attack_formatted}, Decay: {decay_formatted}, Sustain: {sustain_formatted}, Release: {release_formatted}
LFO Modulation: {lfo_modulation}
"""
        # Write the content to the file
        file_path = os.path.join(directory, f"sound{i}.txt")
        with open(file_path, 'w') as file:
            file.write(content)

# Usage - Change number to how many .txt files to generate
create_synth_txt_files('./data/dataset1', 1)