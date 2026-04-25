# Hybrid Phishing Detection System

## 🚨 Problem
Phishing websites bypass traditional detection by mimicking trusted domains and UI designs.

## 💡 Solution
This project implements a hybrid phishing detection system combining:
- Rule-based domain verification
- BERT-based URL classification
- EfficientNet-B3 image analysis

## ⚙️ Tech Stack
- Python, Flask  
- PyTorch  
- HuggingFace Transformers  
- EfficientNet-B3 (CNN)  
- BERT (NLP)

## 🧠 Architecture
1. **Domain Verification**
   - Detects trusted domains
   - Identifies impersonation attempts

2. **URL Analysis (BERT)**
   - Classifies URLs using NLP

3. **Image Analysis (EfficientNet-B3)**
   - Detects phishing via webpage screenshots

4. **Hybrid Decision Engine**
   - Combines outputs using confidence scores
   - Applies rule-based overrides

## 🔍 Key Features
- Multi-layer phishing detection  
- Combines NLP + Computer Vision  
- Confidence-based decision system  
- Real-time analysis via Flask API  

## HOW TO RUN!
- pip install -r requirements.txt
- python app.py

## Future Improvements
- Larger training dataset
- Improved image model accuracy
