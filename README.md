# 🌌 Exoplanet Detection using CNN

This project simulates star light curves and uses a Convolutional Neural Network (CNN) to detect exoplanet transits.

---

## 🔭 Overview

The model analyzes brightness variations in stars and detects dips caused by planets passing in front of them (transits).

---

## ⚙️ Features

- Synthetic light curve generation  
- Multiple dips (planets)  
- Fake dips (false positives)  
- Stellar noise & variability  
- CNN-based detection  

---

## 🤖 Model

- Conv1D neural network (PyTorch)
- Detects patterns in noisy time-series data

---

## 📊 Results

- Accuracy: ~0.67 
- Handles noisy and complex signals  

---

## 📸 Output
![alt text](Screenshot_2026-03-21_17-23-37.png)


---

## 🔧 Tech Stack

- Python  
- NumPy  
- PyTorch  
- Matplotlib  

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
python main.py
