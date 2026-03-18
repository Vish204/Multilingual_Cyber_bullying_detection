# 🚀 Cyberbullying Detection API

This module provides a FastAPI-based backend service for real-time cyberbullying detection using a fusion of multiple AI models.

---

## 📌 Features

- 🔍 Predict cyberbullying from text
- 📊 Fusion-based scoring (CB + sarcasm + emotion)
- ⚠️ Severity classification (None, Mild, Moderate, Severe)
- 🧠 Emotion detection (Top 2 emotions)
- 📝 Logging of predictions
- 📜 History endpoint for past predictions

---

## 📂 Structure

api/
│── main.py              # FastAPI server
│── logger.py            # Logging utility
│── logs/
│    └── predictions.json

---

## ▶️ Run the API

```bash
uvicorn cyberbullying.api.main:app --reload