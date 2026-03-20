# 🔬 DermaDetect — AI-Powered Skin Cancer Detection

<p align="center">
  <img src="https://img.shields.io/badge/Accuracy-81%25-brightgreen?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Macro_AUC-0.9703-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Melanoma_Recall-75.95%25-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Python-3.11-yellow?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/TensorFlow-2.18-FF6F00?style=for-the-badge&logo=tensorflow" />
  <img src="https://img.shields.io/badge/Flask-Web_App-000000?style=for-the-badge&logo=flask" />
</p>

<p align="center">
  A deep learning web application that classifies skin lesions into <b>7 categories</b> using <b>EfficientNetB0</b> transfer learning, with Grad-CAM heatmaps, PDF reports, and MongoDB patient records.
</p>

---

## 📊 Model Performance

EfficientNetB0 was benchmarked against 4 other architectures on a combined **HAM10000 + ISIC 2019** test set of **7,070 samples**:

| Model | Accuracy | Macro AUC | Macro F1 | Melanoma Recall | Melanoma Spec. |
|:---|:---:|:---:|:---:|:---:|:---:|
| ✅ **EfficientNetB0** | **81.00%** | **0.9703** | **0.7942** | **75.95%** | **90.76%** |
| ResNet50 | 64.96% | 0.8866 | 0.4707 | 36.73% | 93.45% |
| VGG16 | 59.00% | 0.7827 | 0.2920 | 36.65% | 86.29% |
| InceptionV3 | 54.20% | 0.7722 | 0.2121 | 55.46% | 73.65% |
| MobileNetV2 | 52.05% | 0.7450 | 0.2299 | 66.64% | 67.05% |

> EfficientNetB0 is the **only model** achieving melanoma recall >70% with specificity >90% simultaneously — the clinically required operating point.

---

## 🧬 Skin Lesion Classes

| Label | Class | Abbrev. |
|:---:|:---|:---:|
| 0 | Actinic Keratosis | AK |
| 1 | Basal Cell Carcinoma | BCC |
| 2 | Benign Keratosis | BKL |
| 3 | Dermatofibroma | DF |
| 4 | Melanocytic Nevi | NV |
| 5 | Pyogenic Granuloma | PG |
| 6 | **Melanoma** | **MEL** |

---

## ✨ Features

- 🧠 **EfficientNetB0** transfer learning with two-phase fine-tuning
- 🔥 **Focal Loss** (γ=2.0) + asymmetric class weights for imbalanced data
- 🗺️ **Grad-CAM heatmaps** — visualise exactly what the model looks at
- 📄 **PDF report generation** per patient scan
- 🗄️ **MongoDB** patient record storage with timestamps
- 🔐 **Login/Register** system with secure password hashing
- 🐳 **Docker** support for one-command deployment

---

## 🏗️ Architecture

```
Upload Image (224×224 RGB)
        ↓
EfficientNetB0 (ImageNet pretrained)
        ↓
GlobalAveragePooling2D
        ↓
Dense(256) → BatchNorm → Dropout(0.4)
        ↓
Dense(128) → Dropout(0.3)
        ↓
Dense(7, softmax)  ← 7-class output
        ↓
Grad-CAM Heatmap + PDF Report + MongoDB
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- MongoDB running locally

### Install & Run

```bash
# Clone
git clone https://github.com/harshaldonarkar/DermaDetect.git
cd DermaDetect

# Setup
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run
python app.py
```

Open **http://localhost:5002**

### Docker (Recommended)

```bash
docker compose up --build
```

---

## 🏋️ Training

Download datasets first:
- [HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
- [ISIC 2019](https://challenge.isic-archive.com/data/)

```bash
# Train EfficientNetB0 (best model)
python train_model.py

# Train comparison models
python train_model.py --model resnet50
python train_model.py --model vgg16
python train_model.py --model mobilenetv2
python train_model.py --model inceptionv3

# Evaluate all models
python evaluate_model.py
```

Results and charts saved to `evaluation_results/`.

---

## 📁 Project Structure

```
DermaDetect/
├── app.py                  # Flask routes, auth, prediction pipeline
├── skin_cancer_detection.py # EfficientNetB0 model definition
├── gradcam.py              # Grad-CAM heatmap generation
├── report_generator.py     # PDF report generation
├── train_model.py          # Training script (5 models)
├── evaluate_model.py       # Evaluation + comparison charts
├── wsgi.py                 # Gunicorn entry point
├── templates/              # HTML templates
├── evaluation_results/     # Saved metrics, charts, confusion matrices
├── docker-compose.yml
└── requirements.txt
```

---

## 🗃️ Dataset

| Dataset | Samples | Source |
|:---|:---:|:---|
| HAM10000 | 10,015 | Medical Univ. Vienna + Queensland |
| ISIC 2019 | 25,331 | International clinical sites |
| **Combined** | **35,346** | After merging |
| **After oversampling** | **51,758** | Minority classes 2× |

---

## 🛠️ Tech Stack

| Layer | Technology |
|:---|:---|
| Model | EfficientNetB0, TensorFlow 2.18, Keras |
| Loss | Categorical Focal Loss (γ=2.0) |
| Backend | Flask, Flask-Login, Gunicorn |
| Database | MongoDB, pymongo |
| Reports | ReportLab |
| Explainability | Grad-CAM |
| Deployment | Docker, Docker Compose |

---

## 📰 Research

> Kuthe A, Donarkar H. *Advanced Skin Lesion Classification Using EfficientNetB0 Transfer Learning: A Five-Model Comparative Study on HAM10000 and ISIC 2019 Combined Dataset.* 2025.

---

## ⚠️ Disclaimer

This tool is for **research and educational purposes only**. It is **not a substitute** for professional medical diagnosis. Always consult a qualified dermatologist.
