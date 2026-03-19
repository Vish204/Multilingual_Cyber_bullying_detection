# Explainability Module (SHAP-based)

---

# Overview

This module provides interpretability for the cyberbullying detection system using **SHAP (SHapley Additive Explanations)**.

It explains why a post is classified as cyberbullying by highlighting important words and supporting signals.

---

# Folder Structure

```
explainability/
│
├── shap_explainer.py   # Core SHAP logic
├── test_shap.py        # Testing script for explanations
└── README.md           # Documentation
```

---

# File Descriptions

## 1. shap_explainer.py

Main file that:

* Loads trained artifacts (XGBoost, TF-IDF, scaler, keywords)
* Converts input text into a feature vector
* Applies SHAP (TreeExplainer)
* Extracts important features
* Formats explanation output

### Key Functions

* `get_shap_values()` → Computes SHAP values
* `explain_text()` → Returns final structured explanation

---

## 2. test_shap.py

Used for:

* Local testing without API
* Debugging explanation quality
* Verifying SHAP outputs before integration

---

# Explanation Components

Each prediction includes:

## Trigger Words

Words that increase cyberbullying probability.

### Example

```json
{"word": "idiot", "impact": 0.031}
```

---

## Counter Words

Words that reduce the cyberbullying probability.

---

## Supporting Signals

Non-text features such as:

* keyword_ratio
* keyword_count
* sentiment
* code_mix

---

# Important Design Decisions

## 1. Active Feature Filtering

Only features present in the input are considered.

---

## 2. Stopword Removal

Removes common words such as "the", "is", etc.

---

## 3. Keyword Integration

Keywords from the multilingual dataset are used to strengthen explanations, especially for words not present in the TF-IDF vocabulary.

---

## 4. Fallback Mechanism

If no TF-IDF words are found, the system highlights meaningful input words.

---

# Multilingual Support

* Keyword-based detection helps capture Hinglish, Hindi, and slang
* TF-IDF may miss rare or unseen words
* Hybrid approach ensures robustness

---

# Benefits

* Transparent predictions
* Human-interpretable outputs
* Supports moderation decisions
* Enables auditing and trust
