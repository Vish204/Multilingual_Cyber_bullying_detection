# Emotion Detection Module — Cyberbullying Project

This module implements the **emotion detection layer** for our Multi-Task Knowledge Distillation (MTKD) framework. It is based on the **XLM-RoBERTa** model fine-tuned on a code-mixed adaptation of the **GoEmotions dataset**, targeting emotions relevant to cyber-abuse detection.

---

## 🗂 Folder Contents


src/cyberbullying/emotion/
│
├─ preprocess.py # Data cleaning and text preparation
├─ split.py # Stratified train/val/test split
├─ model.py # Load tokenizer & model functions
├─ train.py # Training script using HuggingFace Trainer
├─ inference.py # Test/evaluation & save per-post probabilities
└─ pycache/


---

## 📝 Dataset

- **Source:** Adapted from GoEmotions (English) + code-mixed Indian content.
- **Classes:** Filtered to 3 core labels:
  - `neutral`
  - `aggression`
  - `distress`
- **Splits:** Stored in `data/emotion/splits/`:
  - `train.csv`
  - `val.csv`
  - `test.csv`
- **Columns:**  
  - `text` — input sentence/post  
  - `label` — integer label (0,1,2)

---

## ⚙️ Preprocessing (`preprocess.py`)

- Cleans raw text (removes special characters, extra spaces).
- Converts emotion labels from original 27 categories to 3 target classes.
- Outputs processed CSVs in `data/emotion/processed/`.

---

## 📊 Train/Val/Test Split (`split.py`)

- Performs **stratified splitting** to maintain class distribution.
- Saves CSVs in `data/emotion/splits/`.
- Ensures proper train, validation, and test sets for modeling.

---

## 🤖 Model (`model.py`)

- **Tokenizer:** `XLMRobertaTokenizer`
- **Model:** `XLMRobertaForSequenceClassification`
- Initialized from `xlm-roberta-base`.
- Configured for 3-class classification (`neutral`, `aggression`, `distress`).

---

## 🏋️ Training (`train.py`)

- Uses **HuggingFace Trainer API** with PyTorch backend.
- Features:
  - Cross-entropy loss
  - AdamW optimizer
  - Early stopping (`patience=2` on validation F1)
  - Mixed precision (`fp16`) if GPU available
- Training outputs saved in:  
  `models/emotion/` (model checkpoint, tokenizer, config)
- Metrics logged per epoch: `loss`, `accuracy`, `weighted F1`, `macro F1`.

---

## 🔍 Inference & Evaluation (`inference.py`)

- Loads trained model and tokenizer from `models/emotion/`.
- Evaluates on `test.csv`.
- Computes metrics:
  - Accuracy, weighted F1, macro F1
  - Per-class F1
  - Precision & Recall
  - Confusion matrix (`confusion_matrix.png`)
  - Multi-class ROC-AUC (`roc_curves.png`)
- Saves results in `notebooks/analysis_results/emotion/`:
  - `metrics.json` — all computed metrics
  - `emotion_test_probabilities.json` — per-post class probabilities (for Fusion)
  - `emotion_test_predictions.csv` — predicted labels + probabilities
- Supports batch inference for large test sets.

---

## 🧪 Sanity Testing

- Test on **real Indian-style posts** to check emotion activation:
  - `"Wah kya baat hai genius ho tum"` → Neutral/Sarcastic
  - `"Mujhe bohot bura lag raha hai"` → Distress
- Noted limitation: model is **English-heavy**; Hindi detection may be imperfect.

---

## ⚡ Phase 3 Fusion Preparation

- The **per-post probabilities JSON** serves as input to MTKD fusion.
- No need to load the model in Phase 3 unless retraining is required.

---

## 🗂 Outputs Directory Structure


models/emotion/final
│ pytorch_model.bin
│ config.json
│ tokenizer files...

notebooks/analysis_results/emotion/
│ metrics.json
│ emotion_test_probabilities.json
│ emotion_test_predictions.csv
│ confusion_matrix.png
│ roc_curves.png


---

## 🔖 Notes

- Early stopping is implemented (patience=2 epochs).
- All metrics and probabilities are saved in JSON format for consistency.
- Ready for integration with sarcasm detection for the final bullying detection MTKD framework.