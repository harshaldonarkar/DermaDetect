# DermaDetect — AI-Powered Skin Cancer Detection

A Flask web application for AI-assisted skin cancer detection using **EfficientNetB0** transfer learning. Classifies dermoscopic images into 7 categories with **81% accuracy**, Grad-CAM heatmaps, PDF reports, and MongoDB patient records.

---

## Results

| Model | Accuracy | Macro AUC | Melanoma Recall |
|---|---|---|---|
| **EfficientNetB0** | **81.00%** | **0.9703** | **75.95%** |
| ResNet50 | 64.96% | 0.8866 | 36.73% |
| VGG16 | 59.00% | 0.7827 | 36.65% |
| InceptionV3 | 54.20% | 0.7722 | 55.46% |
| MobileNetV2 | 52.05% | 0.7450 | 66.64% |

Trained on **HAM10000 + ISIC 2019** combined dataset (~35,000 images, 7 classes).

---

## Features

- 7-class skin lesion classification (Melanoma, BCC, Melanocytic Nevi, and more)
- Grad-CAM heatmaps for visual explainability
- PDF report generation per patient
- MongoDB patient record storage
- Login/register system with secure authentication
- Docker support for easy deployment

---

## Classes

| Label | Class |
|---|---|
| 0 | Actinic Keratosis |
| 1 | Basal Cell Carcinoma |
| 2 | Benign Keratosis |
| 3 | Dermatofibroma |
| 4 | Melanocytic Nevi |
| 5 | Pyogenic Granuloma |
| 6 | Melanoma |

---

## Setup

### Prerequisites
- Python 3.11+
- MongoDB running locally
- (Optional) Docker

### Install & Run

```bash
# Clone the repo
git clone https://github.com/harshaldonarkar/DermaDetect.git
cd DermaDetect

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

App runs at: `http://localhost:5002`

### Docker

```bash
docker compose up --build
```

---

## Training

Download [HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) and [ISIC 2019](https://challenge.isic-archive.com/data/) datasets, place them in the project root, then:

```bash
# Train EfficientNetB0 (default)
python train_model.py

# Train other models for comparison
python train_model.py --model resnet50
python train_model.py --model vgg16
python train_model.py --model mobilenetv2
python train_model.py --model inceptionv3

# Evaluate all trained models
python evaluate_model.py
```

Results saved to `evaluation_results/`.

---

## Tech Stack

- **Model**: EfficientNetB0 (TensorFlow/Keras), Focal Loss, two-phase fine-tuning
- **Backend**: Flask, Flask-Login, Gunicorn
- **Database**: MongoDB (pymongo)
- **Reports**: ReportLab PDF
- **Explainability**: Grad-CAM
- **Deployment**: Docker + Docker Compose

---

## Research

This project is part of a research paper on AI-based skin cancer detection submitted for publication.

> Kuthe A, Donarkar H. *Advanced Skin Lesion Classification Using EfficientNetB0 Transfer Learning: A Five-Model Comparative Study on HAM10000 and ISIC 2019 Combined Dataset.* 2025.

---

## Disclaimer

This tool is intended for **research and educational purposes only**. It is not a substitute for professional medical diagnosis. Always consult a qualified dermatologist.
