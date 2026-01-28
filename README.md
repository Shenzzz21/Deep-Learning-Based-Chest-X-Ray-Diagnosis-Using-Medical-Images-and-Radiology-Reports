# Multimodal Chest X-Ray Diagnostic Assistant

## Project Overview
This project presents a **multimodal deep learning system** for automated chest X-ray abnormality detection by jointly analyzing **chest X-ray images** and **radiology reports**. Traditional diagnostic systems often rely on a single modality, which can limit diagnostic robustness. To address this, the proposed approach integrates visual and textual information to better reflect real-world clinical workflows.

The system is designed as a **clinical decision-support tool**, providing probabilistic predictions for *Normal* and *Abnormal* cases rather than replacing medical professionals.

---

## Objectives
- Develop a supervised deep learning pipeline for chest X-ray abnormality detection  
- Integrate image and text modalities using pretrained deep learning models  
- Evaluate model performance using clinically relevant metrics  
- Demonstrate the benefits and limitations of multimodal learning in medical imaging  

---

## Dataset
- **Indiana University Chest X-Ray (IU-Xray) Dataset**
- Frontal chest X-ray images paired with radiology reports
- Binary classification task: **Normal (0)** vs **Abnormal (1)**
- Dataset preprocessing includes class balancing strategies and label-leakage prevention

---

## Methodology

### Image Processing
- Chest X-ray images resized to **224 × 224**
- Normalized using **ImageNet mean and standard deviation**
- Compatible with pretrained CNN architectures

### Text Processing
- Radiology reports cleaned and lowercased
- Explicit label-related words removed to prevent leakage
- Tokenized using **BioBERT tokenizer**
- Fixed maximum sequence length of **256 tokens**

---

## Model Architecture

### Image Branch
- **ResNet-18** pretrained on ImageNet
- Final classification layer removed
- Outputs a **512-dimensional image embedding**

### Text Branch
- **BioBERT (dmis-lab/biobert-v1.1)** pretrained on biomedical corpora
- Outputs a **768-dimensional text embedding**

### Multimodal Fusion
- Feature-level concatenation of image and text embeddings
- Fully connected classifier with dropout
- Binary classification: *Normal* vs *Abnormal*

---

## Training Setup
- **Loss Function:** Class-weighted Cross-Entropy Loss
- **Optimizer:** AdamW
- **Batch Size:** 16
- **Epochs:** Up to 20 with early stopping
- **Regularization:** Dropout, gradient clipping, learning rate scheduling
- **Hardware:** GPU-enabled Google Colab environment

---

## Major Technical Outcomes

### Performance Metrics (Test Set)
- **Accuracy:** 82%
- **ROC–AUC:** 0.80
- **Precision (Normal):** 0.93
- **Recall (Normal):** 0.86
- **Recall (Abnormal):** 0.52

### Key Observations
- Strong performance on normal cases
- Moderate sensitivity for abnormal cases due to dataset imbalance
- Multimodal learning improves robustness compared to image-only approaches
- Early stopping effectively mitigates overfitting

---

## Results Visualization
- Class distribution analysis
- Radiology report word count and term frequency analysis
- Confusion matrix and ROC curve
- Interactive inference using a **Gradio web interface**

---

## Limitations
- Relatively small dataset size limits generalization
- Class imbalance impacts abnormal-class precision
- High computational cost due to transformer-based text encoder

---

## Future Work
- Training on larger datasets such as **MIMIC-CXR** or **CheXpert**
- Exploring advanced multimodal fusion techniques (e.g., cross-attention)
- Improving abnormal-class sensitivity
- Deployment as a real-time clinical decision-support system

---

## Disclaimer
This project is intended **for research and educational purposes only** and should not be used as a standalone diagnostic tool in clinical practice.

---

## Author
**Sheneeza Chaudhary**  
Neural Networks & Deep Learning Coursework
