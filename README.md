# 🧥 fashion-classifier-ai

A minimal PyTorch project that trains a TinyVGG-style Convolutional Neural Network (CNN) on the FashionMNIST dataset and visualizes feature maps from its first convolutional block.

---

## 🧠 Overview

This project trains a lightweight convolutional model, **TinyVGG**, to classify grayscale 28x28 images of clothing into 10 categories such as shirts, shoes, and trousers. After training, it also visualizes the **feature maps** learned in the first convolutional block — offering insights into how CNNs "see" image data.

---

## 📂 Features

- ✅ Trains a CNN from scratch on FashionMNIST  
- ✅ Splits data into training, validation, and test sets  
- ✅ Plots loss and accuracy over epochs  
- ✅ Computes and plots a confusion matrix  
- ✅ Visualizes feature maps (activation maps) from the first conv layer

---

## 🧰 Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
