# Experiments

This folder contains experimental Python scripts used during model
development and research. These scripts are **not part of the final
production pipeline**.

Each file represents a specific experiment, baseline, or debugging step.

## Files Overview

### 01_baseline_xgboost.py
Trains a baseline XGBoost model using extracted features.
Used to establish a non-deep-learning performance benchmark.

### 02_muril_teacher_train.py
Trains a MURIL-based teacher model for multilingual cyberbullying detection.
This model serves as the teacher in knowledge distillation.

### 03_muril_kd_xgboost.py
Implements knowledge distillation where predictions or embeddings
from the MURIL teacher model are used to train an XGBoost student model.



## Notes
- Scripts may rely on intermediate datasets or resources.
- Execution order is not guaranteed unless explicitly stated.
- Results from these experiments informed the final pipeline design.
