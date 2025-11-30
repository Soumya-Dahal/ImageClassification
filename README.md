🖼️ Image Classification using CNNs with TensorFlow & Keras
📋 Project Overview
This project demonstrates how to build a powerful Convolutional Neural Network (CNN) for image classification using TensorFlow and Keras. The model achieves 84% accuracy in classifying various scene categories including buildings, forests, mountains etc.

🚀 Quick Start Guide
1. 📥 Download the Dataset
Get the dataset from Kaggle:
🔗 Intel Image Classification Dataset

2. 📁 Setup Folder Structure
Organize your dataset as follows:

text
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
3. ⚙️ Installation
bash
pip install tensorflow
4. 🎯 Run the Model
bash
python3 main.py
🏗️ Model Architecture
📊 CNN Architecture Diagram
https://github.com/user-attachments/assets/f44dc34a-28ef-4857-b029-7da3891197bc

🧠 Network Layers
The model features a sophisticated CNN architecture with:

Multiple Convolutional Layers with ReLU activation

MaxPooling Layers for dimensionality reduction

Batch Normalization for stable training

Dropout Layers to prevent overfitting

Dense Layers for final classification

📊 Performance Metrics
✅ 84% Test Accuracy

📈 Comprehensive training/validation graphs

📉 Loss convergence analysis

🎯 Precision-recall metrics

Check the notebook for detailed performance visualizations!

💾 Model Output
After training, you'll get:

ImgClassification.keras - Your trained model file

🌐 Web Integration
Ready to deploy? Use your .keras model with:

🛠️ Framework Options
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
✨ Key Features
🎯 High Accuracy - 84% on test data

🔧 Easy Customization - Modify hyperparameters easily

📱 Deployment Ready - Direct integration with web frameworks

📊 Comprehensive Analysis - Full training insights

🎨 Clean Architecture - Well-structured and documented

🎨 Image Classes
The model classifies images into 6 categories:

Category	Examples
🏢 Buildings	Skyscrapers, houses, structures
🌳 Forest	Woodlands, trees, natural vegetation
🧊 Glacier	Ice formations, snowy landscapes
⛰️ Mountain	Peaks, hills, rocky terrain
🌊 Sea	Oceans, beaches, marine views
🛣️ Street	Roads, urban scenes, city streets
🔄 Customization

Feel free to experiment with:

Learning rates

Number of layers

Dropout rates

Batch sizes

Optimizer choices
