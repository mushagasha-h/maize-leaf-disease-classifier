
# Maize Leaf Disease Classification with MobileNet

## 📋 Project Overview
This project implements a deep learning-based image classification system for identifying maize (corn) plant diseases. The model uses **MobileNet**, a lightweight convolutional neural network architecture, to classify maize leaf images into **4 disease categories** (0=**Corn Blight**, 1=**Corn Common Rust**, 2=**Corn Gray leaf Spot**, 3=**Healthy**).

## 🎯 Purpose
To create an accurate and efficient computer vision model that can detect and classify maize diseases from leaf images, helping farmers with early disease detection and treatment planning.

## 📁 Dataset Structure
The dataset contains images organized into 4 classes (representing different disease conditions or healthy plants). The data is automatically split into:

- **Training set:** 3,562 images  
- **Validation set:** 600 images  
- **Test set:** 600 images  

## 🏗️ Model Architecture
- **Base Model:** MobileNet v1 (224×224), pre-trained on ImageNet (transfer learning)  
- **Input shape:** 224×224×3 (RGB images)  
- **Total parameters:** 4,253,864  
- **Trainable parameters:** 4,231,976  
- **Non-trainable parameters:** 21,888  

**Custom Classification Head:**
- Global Average Pooling layer  
- Reshape layer  
- Final Dense layer with 4 units (one per disease class)  
- Softmax activation for multi-class classification  

## 🔧 Key Features
- **Automatic Data Organization:** Script organizes raw dataset into train/validation/test splits  
- **Data Augmentation:** Uses TensorFlow's `ImageDataGenerator` for preprocessing  
- **Transfer Learning:** Leverages pre-trained MobileNet weights  
- **Efficient Architecture:** Optimized for mobile/embedded devices  
- **Proper Validation:** 20% of data reserved for validation during training  

## 📊 Data Processing Pipeline
1. **Image Preprocessing**
   - Resize to 224×224 pixels  
   - MobileNet-specific preprocessing (normalization)  
   - Batch processing with size 10  

2. **Data Splitting**
   - Random sampling for validation and test sets (150 images per class)  
   - Maintains class balance across splits  

3. **Generators**
   - Training batches with shuffling  
   - Validation batches for model evaluation  
   - Test batches without shuffling for consistent evaluation  

## 🛠️ Dependencies
```bash
python
tensorflow>=2.0
keras
numpy
scikit-learn
matplotlib
pillow
