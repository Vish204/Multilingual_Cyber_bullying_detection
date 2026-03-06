import pandas as pd
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
FUSION_DIR = PROJECT_ROOT / "notebooks" /"analysis_results" / "fusion"
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
assert len(y_true) == len(y_pred), "Mismatch between test labels and predictions!"

# ------------------------
# Metrics
# ------------------------
acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
roc_auc = roc_auc_score(y_true, y_prob)

metrics_dict = {
    "accuracy": acc,
    "f1_score": f1,
    "precision": precision,
    "recall": recall,
    "roc_auc": roc_auc
}

print("\nFusion Evaluation Metrics:")
for k, v in metrics_dict.items():
    print(f"{k}: {v:.4f}")

# ------------------------
# Confusion Matrix
# ------------------------
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
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
fpr, tpr, thresholds = roc_curve(y_true, y_prob)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
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
import json
with open(METRICS_PATH, "w") as f:
    json.dump(metrics_dict, f, indent=4)
print(f"Metrics saved at: {METRICS_PATH}")