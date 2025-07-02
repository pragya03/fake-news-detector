<h1 align="center">📰 Fake News Detector</h1>

<p align="center">
  <b>A Streamlit App with Machine Learning + Real-Time Source Verification</b><br>
  <i>Detect fake news, verify sources, and browse similar articles — instantly.</i>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/streamlit-app-red?style=for-the-badge" />
  <img src="https://img.shields.io/badge/python-3.9+-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/machine%20learning-sklearn-yellow?style=for-the-badge" />
</p>

---

## 🔍 Overview

This project is a powerful news analysis tool built with **Streamlit** and **machine learning**. It:
- Classifies news content as **Fake** or **Real**
- Checks the **trustworthiness of the source**
- Searches **similar articles live using Google Search API**
- Logs all predictions for **admin review**
- Comes with a sleek, dark-themed UI ✨

---

## 🧠 Features

✅ Machine Learning model trained on `Fake.csv` and `True.csv`  
✅ TF-IDF vectorization with `ngram_range=(1,2)`  
✅ Trusted and untrusted domain detection  
✅ Real-time Google Programmable Search integration  
✅ Final verdict combines ML + search results + confidence  
✅ Clean UI with background gradients or images  
✅ Exported logs with timestamp and results

---

## Project Structure 
```bash
fake-news-detector/
├── app.py                  # Streamlit app for fake news detection
├── fake_news_detection.py  # ML training script
├── model.pkl               # Trained ML model
├── vectorizer.pkl          # TF-IDF vectorizer
├── known_sources.py        # Lists of trusted/untrusted domains
├── source_checker.py       # Functions to check source reliability and fetch titles
├── logs.csv                # Prediction logs (auto-generated)
├── style.css               # Custom CSS for beautiful UI
├── .gitignore              # Ignore config and system files
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── .streamlit/
    └── secrets.toml        # 🔐 API keys (not committed to Git)


