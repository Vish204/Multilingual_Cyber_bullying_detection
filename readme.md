# Source Code – Cyberbullying Detection Pipeline

This directory contains the complete implementation of the cyberbullying detection system.
The pipeline is modular and must be executed in a defined order.

---

## Execution Order (IMPORTANT)

### 1. Data Collection
Path:


src/cyberbullying/data collection/


Scripts:
- reddit_collector.py
- twitter_collector.py
- youtube_collector.py

Output:


data/raw/


These scripts collect raw multilingual social media data.
They are run only once.

---

### 2. Keyword Processing
Path:


src/data/


Scripts:
- 01_build_keywords.py
- 03_load_keywords.py
- 04_validate_keywords.py

Output:


resources/processed_keywords.json


Keywords are used during preprocessing to assist labeling and cleaning.

---

### 3. Data Preprocessing
Path:


src/cyberbullying/preprocessing/


Script:
- preprocess.py

Input:
- data/raw/
- resources/processed_keywords.json

Output:


data/processed/


This step cleans text, removes noise, and creates train/validation/test splits.

---

### 4. Teacher Model Training
Path:


src/cyberbullying/training/


Script:
- teacher_trainer.py

Input:
- data/processed/train.csv

Output:


models/Teacher/


A transformer-based teacher model is trained for high accuracy.

---

### 5. Knowledge Distillation
Path:


src/cyberbullying/distillation/


Script:
- distill.py

Input:
- models/Teacher/
- data/processed/train.csv

Output:


data/probs/teacher_probs.npy


Soft-label probabilities are generated from the teacher model.

---

### 6. Student Model Training
Path:


src/cyberbullying/training/


Script:
- student_trainer.py

Input:
- data/processed/train.csv
- data/probs/teacher_probs.npy

Output:


models/student/


A lightweight student model is trained using knowledge distillation.

---

### 7. Evaluation
Path:


src/cyberbullying/evaluation/


Input:
- models/student/
- data/processed/test.csv

Output:
- Performance metrics and evaluation logs

---

## Reproducibility

- Raw data is preserved in `data/raw/`
- All intermediate outputs can be regenerated
- The pipeline supports complete retraining if needed

## Trained models are not stored in the repository due to size (>1GB).

To reproduce:
1. Train using:
   notebooks/experiment/02_muril_teacher_train.py
   notebooks/experiment/03_mbert_teacher_train.py
   notebooks/experiment/04_xlmr_teacher_train.py
2. Models will be saved locally under:
   models/teacher/