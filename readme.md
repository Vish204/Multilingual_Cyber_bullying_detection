# Real-Time Multilingual Emotion-Aware Cyberbullying Detection System

This repository implements a **multilingual cyberbullying detection system** using:

* Multi-Teacher Knowledge Distillation
* Emotion-Aware Analysis
* Sarcasm Detection
* Fusion-Based Prediction
* Explainable AI (Phase 4)

The system processes **social media posts and comments** and predicts whether the content contains **cyberbullying**, along with **emotion and sarcasm signals** that improve detection accuracy.

The project follows a **modular ML pipeline**, where each phase can be executed independently.

---

# Project Pipeline Overview

```
Data Collection
      ↓
Keyword Processing
      ↓
Data Preprocessing
      ↓
Teacher Model Training
      ↓
Knowledge Distillation
      ↓
Student Model Training
      ↓
Emotion Detection
      ↓
Sarcasm Detection
      ↓
Fusion Prediction
      ↓
Evaluation & Error Analysis
      ↓
Deployment (Phase 4)
```

---

# Project Structure

```
project_root/

data/
    raw/
    processed/
    probs/
    predictions/

models/
    teacher/
    student/

resources/
    processed_keywords.json

src/
    cyberbullying/

        data_collection/
            reddit_collector.py
            twitter_collector.py
            youtube_collector.py

        preprocessing/
            preprocess.py

        training/
            teacher_trainer.py
            student_trainer.py

        distillation/
            distill.py

        emotion/
            emotion_predictor.py

        sarcasm/
            sarcasm_predictor.py

        fusion/
            fusion_predictor.py
            inference_fusion.py

        evaluation/
            evaluate_fusion.py
            run_error_analysis.py
            plot_error_analysis.py

notebooks/
    experiments/

outputs/
    plots/
    error_analysis/

README.md
requirements.txt
```

---

# Execution Order (IMPORTANT)

The pipeline **must be executed in the following order.**

---
# Phase 1 – Data collection and preprocessing

# 1. Data Collection

### Location

```
src/cyberbullying/data_collection/
```

### Scripts

* `reddit_collector.py`
* `twitter_collector.py`
* `youtube_collector.py`

### Output

```
data/raw/
```

These scripts collect **multilingual social media posts and comments**.

### Platforms Supported

* Reddit
* Twitter / X
* YouTube

This step is usually executed **once to gather raw data**.

---

# 2. Keyword Processing

### Location

```
src/data/
```

### Scripts

* `01_build_keywords.py`
* `03_load_keywords.py`
* `04_validate_keywords.py`

### Output

```
resources/processed_keywords.json
```

This stage builds and validates a **keyword dictionary used for filtering and preprocessing**.

---

# 3. Data Preprocessing

### Location

```
src/cyberbullying/preprocessing/
```

### Script

```
preprocess.py
```

### Input

```
data/raw/
resources/processed_keywords.json
```

### Output

```
data/processed/
```

### Operations Performed

* Text cleaning
* Emoji handling
* Noise removal
* Language normalization
* Dataset splitting

### Generated Datasets

```
train.csv
validation.csv
test.csv
```

---
# Phase 2 – Model Training

# 4. Teacher Model Training

### Location

```
src/cyberbullying/training/
```

### Script

```
teacher_trainer.py
```

### Input

```
data/processed/train.csv
```

### Output

```
models/teacher/
```

Multiple **Transformer teacher models** are trained for high accuracy.

### Example Models

* MuRIL
* mBERT
* XLM-R

Training experiments are available in:

```
notebooks/experiments/
```

---

# 5. Knowledge Distillation

### Location

```
src/cyberbullying/distillation/
```

### Script

```
distill.py
```

### Input

```
models/teacher/
data/processed/train.csv
```

### Output

```
data/probs/teacher_probs.npy
```

Teacher models generate **soft probability labels** which are used to train a **lightweight student model**.

---

# 6. Student Model Training

### Location

```
src/cyberbullying/training/
```

### Script

```
student_trainer.py
```

### Input

```
data/processed/train.csv
data/probs/teacher_probs.npy
```

### Output

```
models/student/
```

The **student model** is optimized for:

* Faster inference
* Reduced compute
* Easier deployment

---

# Phase 3 – Multimodal Signal Integration

Phase 3 introduces **emotion and sarcasm signals** to improve cyberbullying detection.

---

# 7. Emotion Detection

### Location

```
src/cyberbullying/emotion/
```

### Script

```
emotion_predictor.py
```

### Output

Emotion probability scores:

```
p_emotion
```

Emotion signals help identify **anger, hate, sadness, or hostility** often present in bullying content.

---

# 8. Sarcasm Detection

### Location

```
src/cyberbullying/sarcasm/
```

### Script

```
sarcasm_predictor.py
```

### Output

Sarcasm probability:

```
p_sarcasm
```

Sarcasm detection helps capture **implicit bullying or ironic insults**.

---

# 9. Fusion-Based Prediction

### Location

```
src/cyberbullying/fusion/
```

### Scripts

* `fusion_predictor.py`
* `inference_fusion.py`

### Input Signals

* Cyberbullying probability
* Emotion probability
* Sarcasm probability

### Output

```
data/predictions/fusion_predictions.csv
```

### Example Columns

```
text
p_cb
p_sarcasm
p_emotion
fusion_score
prediction
severity
```

The **fusion model combines multiple signals** to produce the final cyberbullying decision.

---

# 10. Evaluation

### Location

```
src/cyberbullying/evaluation/
```

### Script

```
evaluate_fusion.py
```

### Metrics Computed

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

---

# 11. Error Analysis

### Location

```
src/cyberbullying/evaluation/
```

### Scripts

* `run_error_analysis.py`
* `plot_error_analysis.py`

### Generated Files

```
outputs/error_analysis/

false_positives.csv
false_negatives.csv
true_positives.csv
true_negatives.csv
```

### Visualizations

```
outputs/plots/

tp_tn_fp_fn_chart.png
fusion_score_distribution.png
probability_distribution.png
confusion_matrix.png
```

These plots help analyze **model weaknesses and misclassifications**.

---

# Phase 3 Status

Phase 3 includes:

* Emotion detection
* Sarcasm detection
* Fusion prediction
* Evaluation
* Error analysis

✅ **Phase 3 is fully completed.**

---

# Phase 4 (Planned System Deployment)

Phase 4 will convert the ML pipeline into a **real-time cyberbullying detection system**.

### Planned Components

* Inference Service
* FastAPI Backend
* Platform Collectors
* Database Storage
* React Dashboard
* Explainable AI (SHAP)
* Cloud Deployment

The system will automatically **collect posts/comments from online platforms and detect cyberbullying in real time**.

---

# Reproducibility

Raw data stored in:

```
data/raw/
```

Intermediate outputs can be regenerated.

The pipeline supports **full retraining**.

---

# Model Storage

Trained models are **not included** in the repository due to large size (**>1GB**).

To reproduce teacher models run:

```
notebooks/experiments/

02_muril_teacher_train.py
03_mbert_teacher_train.py
04_xlmr_teacher_train.py
```

Models will be saved in:

```
models/teacher/
```

---

# Install Dependencies

```
pip install -r requirements.txt
```

---

# License

This project is intended for **academic research and educational use**.
