# Face Verification Server 🔐🧠

A Flask-based API for face verification and ID card information extraction, combining computer vision and OCR technologies.

## Features

- **Face Verification**:
  - Face detection and alignment
  - Face embedding extraction
  - Cosine similarity comparison

- **ID Card Processing**:
  - Arabic text extraction using EasyOCR
  - Field detection using YOLO object detection
  - ID number recognition with Tesseract OCR
  - Structured data extraction from ID cards

## 🛠️ Tech Stack

- Python 3.9
- Flask
- MTCNN (for face detection)
- FaceNet (for face embeddings)
- NumPy
- OpenCV
- Scikit-learn (for distance computation)
- keras
- EasyOCR
- Tesseract OCR
- YOLO for Object Detection

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/MuhammedMaklad/Face-Verification-Api.git cd face-verification-server
