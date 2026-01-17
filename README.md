# 🫀🩺🔊 PCG Heart Sound Classification Project

This repository contains the complete implementation of an end-to-end deep learning pipeline for classifying Phonocardiogram (PCG) heart sound signals into Normal and Abnormal classes using time–frequency representations and a ResNet-18 model.

The project focuses on comparing multiple time–frequency features under a strict, leakage-free evaluation protocol.

---

## 📊 Dataset

This project uses the **PhysioNet 2016 Heart Sound Dataset**.

Download it from:
[https://physionet.org/content/challenge-2016/1.0.0/](https://physionet.org/content/challenge-2016/1.0.0/)

---

## 🧩 Features Compared

Five time–frequency representations are evaluated:

* STFT Spectrogram
* Mel-Spectrogram
* MFCC
* Continuous Wavelet Transform (CWT)
* Wavelet Scattering Transform (WST)

All features are converted to image-like inputs and classified using ResNet-18.

---

### 📁 Folder Setup

After downloading, copy the training a,b,c,d,e and f folders from the Physionet 2016 dataset to your cloned repository like below:

```text
PCG-Heart-Sound-Classification-Thesis/
├── data/
│   ├── raw/
│   │   ├── training-a/
│   │   ├── training-b/
│   │   ├── ...
│   │   └── training-f/
├── src/
├── stage1_main.py
├── stage2_main.py
├── stage3_classification.py
├── .
├── .
└── README.md
```

## 🛠️ Environment Setup

### 1. Clone the repository

```
git clone https://github.com/milanpatel09/PCG-Heart-Sound-Classification-Thesis.git
cd PCG-Heart-Sound-Classification-Thesis
```

### 2. Create virtual environment

```
python -m venv env
```

Activate:

* Windows:

```
.\env\Scripts\activate
```

* Linux/Mac:

```
source env/bin/activate
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

---

## ⚙️ Running the Pipeline (Stage-wise)

The project is divided into three main stages:

### Stage 1: Preprocessing

This stage:

* Resamples audio from 2000 Hz → 1000 Hz
* Applies Schmidt spike removal
* Applies Butterworth band-pass filter (25–450 Hz)
* Performs Z-score normalization
* Segments audio into 5s clips with 2.5s overlap
* Saves output as NumPy arrays

Run:

```
python stage1_main.py
```

Output: 3240 noisy audio recordings preprocessed into 24450 segments. (might change)

```
data/processed/X_data.npy   # shape: (24450, 5000)
data/processed/Y_data.npy
```

-- xx --

### Stage 2: Feature Extraction

This stage converts segments into five time–frequency features:

* STFT Spectrogram
* Mel-Spectrogram
* MFCC
* CWT
* Wavelet Scattering Transform

Run:

```
python stage2_main.py
```

Output files:

```
data/features/spectrogram.npy
data/features/mel-spec.npy
data/features/mfcc.npy
data/features/cwt.npy
data/features/scattering.npy
```

-- xx --

### Optional: To Visualze the featuers

Run:

```
python visualize_feautes_individual.py
```

check the visualization in the visualizations folder.

-- xx --

### Stage 3: Classification (No Leakage Setting)

before stage 3, Run: to generate groups.npy for group wise splitting to prevent leakage.
```
python generate_groups.py
```

Classification stage:

* Uses ResNet-18
* Converts features to 3-channel images
* Applies stratified, recording-wise split (80/20)
* Trains model and test model and saves all .pth files

Run:

```
python stage3_classification.py --feature "feature_name" --arch "architecture_name"
```

replace "feature_name" with: melspec, mfcc, cwt, spectrogram, scattering

replace "architecture_name" with: resnet18, resnet34, resnet50

One at a time. Example: "To run mfcc on resnet18: "python stage3_classification.py --feature mfcc --arch resnet18"

Models saved in:

```
models_checkpoints/
```

### Optional: Leakage Experiment

To see optimism bias when leakage is allowed:

```
python stage3_leaky.py --feature "feature_name" --arch "architecture_name"
```

This performs random splitting, allowing data leakage.

---

## 📈 Evaluation Metrics

Models are evaluated using:

* Validation Accuracy
* Sensitivity
* Specificity
* F1-Score
* Mean Accuracy = (Sensitivity + Specificity) / 2

---

## 📝 Key Findings

* Best feature for 2D CNN (ResNet-18):
  Mel-Spectrogram (Mean accuracy of 91.90%) > MFCC > CWT > STFT > WST

* Data leakage inflates performance by 5–10%.

* Feature–model compatibility is more important than model depth.

--- 

## 🧾 License
This project is released under the MIT License.

---

## 🙋 Author
milanpatel09 — contributions and feedback welcome.
