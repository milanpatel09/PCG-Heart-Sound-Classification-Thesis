# PCG-Heart-Sound-Classification-Project

## 📌 Project Overview
Cardiovascular diseases are a leading cause of mortality. This project develops an automated, Subject-Independent deep learning framework to classify Phonocardiogram (PCG) signals (Heart Sounds) into **Normal** and **Abnormal** categories.

We systematically compare five time-frequency feature representations using a **ResNet-18** backbone:
1. **Mel-Spectrogram** (Best Performer)
2. **MFCC** (Mel-Frequency Cepstral Coefficients)
3. **CWT** (Continuous Wavelet Transform)
4. **STFT** (Short-Time Fourier Transform)
5. **Wavelet Scattering Transform** (WST)

---

## 🔑 Key Features
* **Leakage-Free Evaluation:** Implements **Stratified Group-Wise Splitting** to ensure patients in the test set are completely unseen during training.
* **End-to-End Pipeline:** Includes preprocessing (spike removal, filtering), segmentation, feature extraction, and classification.
* **Optimism Bias Analysis:** demonstrably proves that improper data splitting (Random Split) inflates accuracy by **5-10%** compared to the realistic Group-Wise split.
* **Reproducible Results:** Fixed seeds and unified pipeline for fair comparison.

---

## 📂 Dataset Setup (Crucial Step)
This project uses the **PhysioNet 2016** dataset. Due to licensing and size, the raw audio files are **not** included in this repository.

### **Step 1: Download Data**
1.  Go to the [PhysioNet 2016 Challenge Page](https://physionet.org/content/challenge-2016/1.0.0/).
2.  Download the training set (approximately 1 GB).

### **Step 2: Organize Folders**
Inside your cloned repository, create a folder structure like this:
`data/raw/`

Copy the **unzipped** folders (`training-a` through `training-f`) into `data/raw`. Your directory should look like this:

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
├── stage3_leaky.py
├── requirements.txt
└── README.md