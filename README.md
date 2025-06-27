# Medical-Image-Analysis-
Pneumonia Medical Image Analysis
# Pneumonia Detection Using GANs and CNNs

This project leverages **Generative Adversarial Networks (GANs)** and **Convolutional Neural Networks (CNNs)** to generate synthetic chest X-ray images and classify them as **Normal** or **Pneumonia**. It addresses challenges like limited labeled medical data, class imbalance, and patient data privacy—especially relevant in healthcare systems with limited resources.

---

## 🔍 Project Overview

- ✅ **GANs** (CycleGAN, Pix2Pix) generate synthetic pneumonia X-rays
- ✅ **ResNet-18 CNN** for binary classification (Normal vs Pneumonia)
- ✅ **Tkinter GUI** (`app.py`) for real-time interaction
- ✅ **Metrics**: Accuracy, F1-score, Precision, Recall, FID Score
- ✅ **Privacy**: No real patient data required for model training

---

## 📁 Repository Structure

```bash
.
├── main.py         # Model training / testing script
├── app.py          # GUI for image upload and diagnosis
├── 0.5             # (Unknown — remove or rename if unused)
├── README.md       # Project documentation (this file)
└── dataset/        # (NOT included; download manually)
```

---

## 📥 Dataset (Manual Download Required)

Due to file size constraints, the datasets are not included in this repository. Download them manually from these official sources:

### 🧪 NIH Chest X-ray Dataset (112,000+ images)
- 🔗 https://nihcc.app.box.com/v/ChestXray-NIHCC

### 🏥 RSNA Pneumonia Detection Challenge
- 🔗 https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data

After downloading, place the dataset in the following directory:

```bash
project-root/
└── dataset/
    ├── images/
    └── labels.csv or similar
```

---

## ⚙️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/medical-image-gan.git
cd medical-image-gan
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
Or install manually:
```bash
pip install tensorflow keras opencv-python scikit-learn pillow matplotlib
```

---

## 🚀 How to Run

### For Training / Testing Models
```bash
python main.py
```

### For GUI Interface
```bash
python app.py
```

The Tkinter window will open, allowing image upload and prediction.

---

## 📊 Evaluation Metrics

- ✅ **Accuracy**
- ✅ **Precision**
- ✅ **Recall**
- ✅ **F1-Score**
- ✅ **Fréchet Inception Distance (FID)** – for synthetic image quality
- ✅ **Dice Coefficient** – for segmentation quality (if used)

---

## 👨‍⚕️ Use Cases

- Augment small medical datasets
- Classify pneumonia from chest X-rays
- Protect patient data via synthetic augmentation
- Prototype healthcare AI tools in low-data settings

---

## 👨‍💻 Authors

Developed by:
- Satram Prashanti Sai
- Kashetty Jahnavi Chakravarthy
- Nareddy Sushank Reddy
- Vippakayala Rahul

Guided by **Mrs. B. Spandana**  
Dept. of CSE (AI & ML), Sreyas Institute of Engineering and Technology

---

## 📄 License

This project is for **academic and research purposes only**. Contact the authors for permission if using beyond educational or demo contexts.
