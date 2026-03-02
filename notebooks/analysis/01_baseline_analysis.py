import joblib
import pandas as pd
import numpy as np
import json
import re
from pathlib import Path

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)

import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
# PATH SETUP
# ===============================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS_ROOT = Path(__file__).resolve().parents[1]

MODEL_NAME = "baseline_xgboost"
MODEL_PATH = PROJECT_ROOT / "models" / "baseline_xgboost" / "baseline_xgboost.pkl"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "test_data.csv"

RESULTS_DIR = NOTEBOOKS_ROOT / "analysis_results" / MODEL_NAME
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ===============================
# FEATURE EXTRACTION
# (Must match training exactly)
# ===============================

def extract_features(texts):
    features = []
    for text in texts:
        text = str(text)
        features.append([
            len(text),
            len(text.split()),
            sum(1 for c in text if c.isupper()),
            text.count('!'),
            text.count('?'),
        ])
    return features


# ===============================
# LOAD MODEL
# ===============================

print("Loading model:", MODEL_PATH)
model = joblib.load(MODEL_PATH)

# ===============================
# BASELINE MODEL SIZE
# ===============================

model_size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)

print("Baseline Model Size (MB):", round(model_size_mb, 2))

# ===============================
# LOAD TEST DATA
# ===============================

test_df = pd.read_csv(DATA_PATH)

texts = test_df["text"].tolist()
y_true = test_df["label"].values

print("Test samples:", len(texts))

# ===============================
# FEATURE EXTRACTION
# ===============================

X_test = extract_features(texts)

# ===============================
# PREDICT
# ===============================

y_pred = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1]

# ===============================
# CONFUSION MATRIX
# ===============================

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=["non_toxic", "toxic"],
    yticklabels=["non_toxic", "toxic"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"{MODEL_NAME.upper()} Confusion Matrix")

plt.savefig(RESULTS_DIR / "confusion_matrix.png")
plt.close()

# ===============================
# CLASSIFICATION REPORT
# ===============================

report = classification_report(
    y_true,
    y_pred,
    target_names=["non_toxic", "toxic"],
    output_dict=True
)

accuracy = report["accuracy"]
precision = report["toxic"]["precision"]
recall = report["toxic"]["recall"]
f1 = report["toxic"]["f1-score"]

# ===============================
# FALSE POSITIVES / NEGATIVES
# ===============================

false_positives = []
false_negatives = []

for text, true, pred in zip(texts, y_true, y_pred):
    if true == 0 and pred == 1:
        false_positives.append(text)
    if true == 1 and pred == 0:
        false_negatives.append(text)

pd.DataFrame(false_positives, columns=["text"]).to_csv(
    RESULTS_DIR / "false_positives.csv", index=False
)

pd.DataFrame(false_negatives, columns=["text"]).to_csv(
    RESULTS_DIR / "false_negatives.csv", index=False
)

# ===============================
# HINDI ERROR ANALYSIS
# ===============================

def contains_hindi(text):
    return bool(re.search(r'[\u0900-\u097F]', text))

hindi_errors = []

for text, true, pred in zip(texts, y_true, y_pred):
    if contains_hindi(text) and true != pred:
        hindi_errors.append(text)

pd.DataFrame(hindi_errors, columns=["text"]).to_csv(
    RESULTS_DIR / "hindi_errors.csv", index=False
)

# ===============================
# ROC-AUC (SAFE)
# ===============================

try:
    auc = roc_auc_score(y_true, probs)
except:
    auc = 0

# ===============================
# ROC CURVE PLOT (NEW)
# ===============================

try:
    fpr, tpr, _ = roc_curve(y_true, probs)

    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr)
    plt.plot([0,1], [0,1], linestyle="--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{MODEL_NAME.upper()} ROC Curve")

    plt.savefig(RESULTS_DIR / "roc_curve.png")
    plt.close()

except:
    print("ROC curve could not be generated.")

# ===============================
# PROBABILITY DISTRIBUTION (NEW)
# Helps understand baseline behaviour
# ===============================

plt.figure(figsize=(6,5))
sns.histplot(probs, bins=50)

plt.title("Prediction Probability Distribution")
plt.xlabel("Toxic Probability")
plt.ylabel("Count")

plt.savefig(RESULTS_DIR / "probability_distribution.png")
plt.close()

# ===============================
# SAVE METRICS JSON
# ===============================

results = {
    "model_name": MODEL_NAME,
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "roc_auc": float(auc),
    "parameters": None,
    "model_size_mb": float(model_size_mb),
    "false_positives": len(false_positives),
    "false_negatives": len(false_negatives),
    "hindi_errors": len(hindi_errors)
}

with open(RESULTS_DIR / "metrics.json", "w") as f:
    json.dump(results, f, indent=4)

print("\nEvaluation Saved in:", RESULTS_DIR)
print(json.dumps(results, indent=4))