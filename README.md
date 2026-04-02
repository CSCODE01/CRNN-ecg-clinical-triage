# 🫀 CRNN (ResNet + BiGRU) for ECG Classification
**A hybrid Deep Learning architecture for highly vigilant 12-lead ECG clinical triage and pathology detection using the PTB-XL Dataset.**

## 📌 Project Overview
This repository implements a Convolutional Recurrent Neural Network (CRNN) to classify 12-lead electrocardiogram (ECG) signals. The architecture combines the spatial feature extraction power of **ResNet** (to capture morphological changes like Hypertrophy) with the temporal context modeling of **BiGRU** (to track rhythm across the entire signal).

The objective is to evaluate the model as a **Hierarchical Clinical Triage System**, prioritizing patient safety and ensuring a near-zero rate of missed critical cases (High Recall).

## 📊 Clinical Evaluation Framework
The model is evaluated based on real-world clinical priorities across three dimensions:
1. **Level 1: Rapid Triage (Sick vs. Healthy)** - ER screening to route patients.
2. **Level 2: Specific Pathology Detection** - Classifying CD, HYP, MI, NORM, and STTC.
3. **Level 3: Co-morbidities (Complex Cases)** - Identifying high-risk patients with simultaneous, overlapping diseases.

## 🚀 Key Clinical Results
The model was evaluated using a strict **80% Train | 10% Validation | 10% Test** data split.

* **Overall Triage Accuracy:** 89.31%
* **Triage Recall (Catching Sick):** 94.53% (Standout metric: Exceptional sensitivity to prioritize patient safety)
* **Triage Precision:** 87.22%
* **Overall Co-morbidity Accuracy:** 87.06%
* **Co-morbidity Recall:** 69.15% (Highly vigilant, actively flags complex multi-disease scenarios)

## ⚙️ Dataset & Preprocessing
* **Dataset:** PTB-XL (approx. 21,800 patients, cleaned version).
* **Signal Specifications:** 10 seconds of 12-lead ECG signals sampled at 100Hz (1000 samples).

## 🛠️ Requirements & Execution
This code is designed to be executed in a GPU-accelerated environment (e.g., Kaggle GPU P100).

### Dependencies:
```bash
numpy
tensorflow >= 2.x
matplotlib
seaborn
scikit-learn
