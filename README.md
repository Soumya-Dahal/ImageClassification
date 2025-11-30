# 🖼️ Image Classification using CNNs with TensorFlow & Keras
## 📋 Project Overview
This project demonstrates how to build a powerful Convolutional Neural Network (CNN) for image classification using TensorFlow and Keras. The model achieves 84% accuracy in classifying various scene categories including buildings, forests, mountains etc.

## 🚀 Quick Start Guide
### 1. 📥 Download the Dataset
Get the dataset from Kaggle:
🔗 Intel Image Classification Dataset: https://www.kaggle.com/datasets/puneet6060/intel-image-classification

### 2. 📁 Setup Folder Structure
Organize your dataset as follows:

dataset/
├── train/
│   ├── buildings/
│   ├── forest/
│   ├── glacier/
│   ├── mountain/
│   ├── sea/
│   └── street/
├── test/
│   ├── buildings/
│   ├── forest/
│   ├── glacier/
│   ├── mountain/
│   ├── sea/
│   └── street/

(NOTE: I have not used the validatoin data provided in the dataset, I just used 20% of training data as validation data.)

## ⚙️ Installation
bash
pip install tensorflow

## 🎯 Run the Model
bash
python3 main.py

## 🏗️ Model Architecture
📊 CNN Architecture Diagram
<img width="3664" height="426" alt="diagram-export-11-30-2025-10_27_06-PM" src="https://github.com/user-attachments/assets/c2338c09-a68e-4650-9fdc-6e9661ab28a9" />

🧠 Network Layers
The model features a sophisticated CNN architecture with:

Multiple Convolutional Layers with ReLU activation

MaxPooling Layers for dimensionality reduction

Batch Normalization for stable training

Dropout Layers to prevent overfitting

Dense Layers for final classification

## 📊 Performance Metrics
✅ 84% Test Accuracy

📈 Comprehensive training/validation graphs


Check the notebook for detailed performance visualizations!

## 💾 Model Output
After training, you'll get:

ImgClassification.keras - Your trained model file

## 🌐 Web Integration
if you wish to deploy use your .keras model with:

### 🛠️ Framework Options
Flask - Lightweight and flexible

Django - Full-featured and scalable

FastAPI - Modern and high-performance

💡 Quick Start with Flask
python
from tensorflow import keras
from flask import Flask, request, jsonify

model = keras.models.load_model('ImgClassification.keras')
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Your prediction logic here
    return jsonify({'prediction': 'your_class'})
## ✨ Key Features
🎯 High Accuracy - 84% on test data

🔧 Easy Customization - Modify hyperparameters easily

📱 Deployment Ready - Direct integration with web frameworks

📊 Comprehensive Analysis - Full training insights

🎨 Clean Architecture - Well-structured and documented

🎨 Image Classes
The model classifies images into 6 categories:

Category	Examples
🏢 Buildings:  Skyscrapers, houses, structures
🌳 Forest:	Woodlands, trees, natural vegetation
🧊 Glacier:	Ice formations, snowy landscapes
⛰️ Mountain:	Peaks, hills, rocky terrain
🌊 Sea:	Oceans, beaches, marine views
🛣️ Street:	Roads, urban scenes, city streets

## Feel free to experiment with:

### 1. Learning rates

### 2. Number of layers

### 3. Dropout rates

### 4. Batch sizes

### 5. Optimizer choices
