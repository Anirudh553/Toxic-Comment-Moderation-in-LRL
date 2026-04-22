# Toxic Comment Moderation for Low-Resource Languages

## Disclaimer

Important Notice

This project deals with toxic, offensive, and potentially harmful language.

- The dataset and outputs may include:
  - Hate speech  
  - Abusive language  
  - Offensive expressions  

These are used strictly for research and educational purposes in the context of content moderation.

The creators do not endorse or promote any harmful language or viewpoints present in the data.

---

## Overview
This project focuses on building a toxic comment detection system specifically for low-resource languages, where:

- Data is scarce  
- Text is noisy (slang, emojis, code-mixing)  
- Standard NLP tools are limited  

Unlike traditional moderation systems built for English-heavy datasets, this system is optimized for real-world social media text.

---

## Objectives
- Detect toxic, abusive, hateful, and offensive comments  
- Handle Hinglish / code-mixed language  
- Work effectively with limited training data  
- Improve robustness against:
  - Slang  
  - Misspellings  
  - Emojis  
  - Informal grammar  

---

## Approach

### Pipeline:

### Techniques Used:
- Multilingual Transformers (mBERT, XLM-R)  
- Transfer Learning  
- Class Imbalance Handling (Focal Loss, weighting)  
- Data Augmentation  
- Noise Handling (slang normalization, emoji processing)  

---

## Features
- Handles noisy social media text  
- Works in low-resource settings  
- Supports code-mixed languages (Hinglish, etc.)  
- Provides interpretable predictions (attention/token importance)  
- Designed for real-world moderation scenarios  

---

## Models Explored
- Logistic Regression  
- SVM  
- LSTM / BiLSTM  
- Transformer-based models (mBERT, XLM-R)  

---

## Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- F1 Score  
- Macro F1 (for imbalance handling)  

---

## Project Structure

---

## Why This Matters
Most moderation systems ignore low-resource languages.

This project aims to:
- Make AI moderation more inclusive  
- Support regional and underrepresented languages  
- Improve safety in diverse online communities  

---

## References
- Multilingual Toxic Comment Dataset  
- Hate Speech Detection Research  
- Transformer Models (mBERT, XLM-R)  
- Transfer Learning in NLP  

---

## Author
Anirudh Anand Krishnan

---

## Future Work
- Better Hinglish dataset collection  
- Context-aware moderation (image + text)  
- Real-time moderation API  
- Explainable AI improvements  

## Demo Video
Watch here: https://drive.google.com/file/d/1GHhG_AK85db5R2p3Vlk26II8ODSV4ukn/view?usp=sharing
