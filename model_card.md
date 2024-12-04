# **Model Card: WhisperSynth**

### **Model Name:** 
WhisperSynth (Fine-Tuned Whisper-Small for Synthesizer Sound Matching)

### **Model Type:** 
Encoder-Decoder Transformer

### **Version:** 
v1.0

---

## **1. Overview**
### **Purpose:**  
Automates synthesizer sound matching by converting `.wav` files into structured string representations of synthesizer settings.  

### **Intended Use:**  
- Assisting music producers in recreating synthesizer sounds.  
- Educational purposes in sound design and machine learning.  

### **Not Intended For:**  
- General audio-to-text transcription (e.g., speech recognition).  
- Real-time, low-latency applications (current version).  

---

## **2. Model Details**
### **Architecture:** 
Whisper-Small (pre-trained HuggingFace model).  

### **Input:** 
Log-mel spectrograms derived from `.wav` files.  

### **Output:** 
Synthesizer settings in a structured string format.  

### **Fine-Tuning Dataset:**  
- 200 `.wav` files and their corresponding `.txt` files generated with Serum and Ableton.  
- Dataset is not publicly available (proprietary sound design).  

### **Training Configuration:**  
- Framework: HuggingFace Transformers and PyTorch.  
- Learning Rate: `5e-5`.  
- Epochs: `10`.  

---

## **3. Performance**
### **Metrics:**  
- **Training Loss:** `0.39` after 10 epochs.  
- **Evaluation Accuracy:** Poor alignment between predictions and structured `.txt` targets.  

### **Current Limitations:**  
- Model struggles to generalize outputs into structured formats.  
- Dataset size and diversity are insufficient for strong performance.  

---

## **4. Ethical Considerations**
### **Risks and Limitations:**  
- Outputs may not reflect exact synthesizer settings, leading to inaccuracies.  
- Model performance is highly dependent on the quality of input data.  

### **Bias and Fairness:**  
- Dataset is highly niche (e.g., focused on Serum synthesizer). This may bias the model toward specific use cases.  

---

## **5. Usage**
### **How to Use:**  
1. Preprocess your `.wav` files via the model's built-in capabilities.  
2. Pass the files through the model to generate a string.  

### **Limitations of Current Version:**  
- Outputs are strings, not `.fxp`. Further development is required for `.fxp` integration.  

---
