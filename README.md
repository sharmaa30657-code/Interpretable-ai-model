# Interpretable-ai-model
second year AI/ML project on interpretable deep learning models.

# Interpretable AI Image Classification System

## Project Overview
This project implements an interpretable AI-based image classification system. 
The model uses transfer learning with ResNet-18 to classify images uploaded by users. 
The web application allows users to upload an image and receive the predicted class 
along with the confidence score.

## Features
• Image upload through web interface  
• Image classification using deep learning  
• Display of prediction confidence  
• Custom dataset training using transfer learning  
• Flask-based web application

## Dataset
The model was trained on a custom dataset containing the following classes:

- Cat
- Tiger
- Audi

Each class contains approximately 50 training images.

## Technologies Used
- Python
- PyTorch
- Flask
- HTML / CSS
- Transfer Learning (ResNet18)

## How to Run

1. Clone the repository
2. Install dependencies

pip install -r requirements.txt
3. Run the application
python app.py
4. Open the web interface and upload an image to get predictions.

## Future Improvements
• Add Grad-CAM for model interpretability  
• Improve dataset size  
• Add better error handling

## live demo 
.live App:
https://interpretable-ai-model.onrender.com/

## GitHub Repository
https://github.com/sharmaa30657-code/Interpretable-ai-model

## CI/CD Pipeline

- Source code managed using GitHub
- Automatic deployment configured using Render
- Whenever new code is pushed to GitHub, Render automatically rebuilds and redeploys the application
- This creates a simple CI/CD workflow

---
title: Interpretable AI Model
emoji: 🚀
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
---

# Interpretable AI Model

This project uses PyTorch and Grad-CAM for image classification and explainable AI visualization.