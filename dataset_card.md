# **Dataset Card: WhisperSynth Dataset**

### **Dataset Name:** 
WhisperSynth Training Dataset

### **Number of Examples:** 
~200

### **Data Modality:** 
Audio (`.wav`) and Text (`.txt`)

---

## **1. Dataset Description**
### **Context:**  
This dataset contains `.wav` files representing synthesizer sounds and corresponding `.txt` files with synthesizer settings.  

### **Data Sources:**  
- Audio was generated using **Serum** and **Ableton Live** synthesizers.  
- Settings were manually recorded and formatted into `.txt` files.  

### **Example Pair:**  
**Audio File (`example.wav`):** Synthesizer sound.  
**Text File (`example.txt`):**  

Waveform: Sine

Voices: 1

Oscillator Detune: None

Filter Type: Highpass

Filter Cutoff: 3000Hz

ADSR Envelope: Attack: 400ms, Decay: 900ms, Sustain: -12 dB, Release: 600ms

LFO Modulation: None


---

## **2. Intended Use**
### **Purpose:**  
Training models for audio-to-structured-text tasks, particularly synthesizer sound matching.  

### **Not Suitable For:**  
- General audio-to-text transcription tasks.  
- Real-time applications (due to dataset size and diversity).  

---

## **3. Dataset Construction**
### **Generation Process:**  
1. **Audio Generation:**  
   - `.wav` files were synthesized using Serum and Ableton Live.  
   - Different combinations of oscillator types, filters, ADSR envelopes, and modulation were used.  

2. **Text Annotations:**  
   - Corresponding `.txt` files were manually created, containing structured representations of synthesizer settings.  

3. **Preprocessing:**  
   - Data was structured to align with Whisper-Small’s input/output requirements.  
   - `.wav` files were converted into log-mel spectrograms for model ingestion.  

---

## **4. Limitations**
- Dataset size (200 examples) is relatively small, which limits model generalization.  
- Proprietary `.fxp` format for Serum presets is not included due to reverse-engineering challenges.  

---

## **5. License**
- **Dataset License:** Proprietary (not publicly available).  

---

## **6. Ethical Considerations**
### **Potential Biases:**  
- The dataset is focused on Serum synthesizer sounds, which may bias the model toward specific sound design tools.  

### **Limitations of Use:**  
- This dataset is not suitable for broader audio processing tasks beyond synthesizer sound matching.  

---

## **7. Future Improvements**
- Expand the dataset to include more `.wav` files and corresponding `.txt` settings.  
- Incorporate `.fxp` files into the dataset for direct preset generation.  

---
