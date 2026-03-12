import pandas as pd
import numpy as np
import json
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

import matplotlib.pyplot as plt
import seaborn as sns


# ------------------------
# Paths
# ------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

FUSION_DIR = PROJECT_ROOT / "notebooks" / "analysis_results" / "fusion"
DATA_DIR = PROJECT_ROOT / "data" / "processed"

FUSION_PRED_PATH = FUSION_DIR / "fusion_predictions.csv"
TEST_DATA_PATH = DATA_DIR / "test_data.csv"

CONF_MATRIX_PATH = FUSION_DIR / "fusion_confusion_matrix.png"
ROC_CURVE_PATH = FUSION_DIR / "fusion_roc_curve.png"
METRICS_PATH = FUSION_DIR / "fusion_metrics.json"


# ------------------------
# Load data
# ------------------------

fusion_df = pd.read_csv(FUSION_PRED_PATH)
test_df = pd.read_csv(TEST_DATA_PATH)

y_true = test_df["label"].values
y_pred = fusion_df["prediction"].values
y_prob = fusion_df["fusion_score"].values

assert len(y_true) == len(y_pred), "Mismatch between labels and predictions!"


# ------------------------
# Metrics
# ------------------------

acc = accuracy_score(y_true, y_pred)

precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)

f1 = f1_score(y_true, y_pred, zero_division=0)

roc_auc = roc_auc_score(y_true, y_prob)


# ------------------------
# Error counts
# ------------------------

false_pos = int(((y_pred == 1) & (y_true == 0)).sum())
false_neg = int(((y_pred == 0) & (y_true == 1)).sum())


metrics_dict = {

    "accuracy": acc,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "roc_auc": roc_auc,

    "false_positives": false_pos,
    "false_negatives": false_neg
}


print("\nFusion Evaluation Metrics:")

for k, v in metrics_dict.items():

    if isinstance(v, float):
        print(f"{k}: {v:.4f}")
    else:
        print(f"{k}: {v}")


# ------------------------
# Confusion Matrix
# ------------------------

cm = confusion_matrix(y_true, y_pred)

np.save(FUSION_DIR / "fusion_confusion_matrix.npy", cm)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Non-CB","CB"],
    yticklabels=["Non-CB","CB"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.title("Fusion Confusion Matrix")

plt.tight_layout()

plt.savefig(CONF_MATRIX_PATH)

plt.close()

print(f"Confusion matrix saved at: {CONF_MATRIX_PATH}")


# ------------------------
# ROC Curve
# ------------------------

fpr, tpr, _ = roc_curve(y_true, y_prob)

plt.figure(figsize=(6,5))

plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.4f}")

plt.plot([0,1],[0,1],'--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("Fusion ROC Curve")

plt.legend(loc="lower right")

plt.tight_layout()

plt.savefig(ROC_CURVE_PATH)

plt.close()

print(f"ROC curve saved at: {ROC_CURVE_PATH}")


# ------------------------
# Save metrics
# ------------------------

with open(METRICS_PATH, "w") as f:

    json.dump(metrics_dict, f, indent=4)

print(f"Metrics saved at: {METRICS_PATH}")