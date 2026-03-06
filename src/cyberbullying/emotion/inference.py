import torch
import pandas as pd
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, auc,
    classification_report
)
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification

# ======================
# PATHS
# ======================

BASE_DIR = Path(__file__).resolve().parents[3]
SPLIT_DIR = BASE_DIR / "data" / "emotion" / "splits"
TEST_PATH = SPLIT_DIR / "test.csv"

MODEL_DIR = BASE_DIR / "models" / "emotion" / "final"
RESULTS_DIR = BASE_DIR / "notebooks" / "analysis_results" / "emotion"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PROBS_PATH = RESULTS_DIR / "emotion_test_probabilities.json"
PREDICTIONS_PATH = RESULTS_DIR / "emotion_test_predictions.csv"
METRICS_PATH = RESULTS_DIR / "metrics.json"

# ======================
# LOAD TEST DATA
# ======================

test_df = pd.read_csv(TEST_PATH)
texts = test_df["text"].tolist()
labels = test_df["label"].tolist()

num_classes = len(set(labels))
class_names = ["neutral", "aggression", "distress"]  # adjust if needed

print(f"Test samples: {len(texts)}")

# ======================
# LOAD MODEL & TOKENIZER (IMPORTANT FIX)
# ======================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_DIR)
model = XLMRobertaForSequenceClassification.from_pretrained(MODEL_DIR)

model.to(device)
model.eval()

print("Model loaded from trained checkpoint and set to eval mode.")

# ======================
# PREDICTIONS
# ======================

batch_size = 16
all_preds = []
all_probs = []

with torch.no_grad():
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]

        encodings = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)

        all_preds.extend(preds.tolist())
        all_probs.extend(probs.tolist())

y_true = np.array(labels)
y_pred = np.array(all_preds)
y_probs = np.array(all_probs)

# ======================
# METRICS
# ======================

metrics_dict = {
    "accuracy": float(accuracy_score(y_true, y_pred)),
    "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
    "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    "precision": float(precision_score(y_true, y_pred, average="weighted")),
    "recall": float(recall_score(y_true, y_pred, average="weighted")),
    "per_class_f1": f1_score(y_true, y_pred, average=None).tolist(),
    "classification_report": classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True
    )
}

# ======================
# CONFUSION MATRIX
# ======================

cm = confusion_matrix(y_true, y_pred)
metrics_dict["confusion_matrix"] = cm.tolist()

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Emotion Confusion Matrix")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "confusion_matrix.png")
plt.close()

# ======================
# ROC-AUC (Multi-class)
# ======================

y_true_onehot = np.zeros((len(y_true), num_classes))
for i, val in enumerate(y_true):
    y_true_onehot[i, val] = 1

try:
    roc_auc = roc_auc_score(
        y_true_onehot,
        y_probs,
        average="macro",
        multi_class="ovr"
    )
    metrics_dict["roc_auc"] = float(roc_auc)

    plt.figure(figsize=(7, 6))
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_probs[:, i])
        auc_i = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC={auc_i:.2f})")

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Emotion ROC Curves")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "roc_curves.png")
    plt.close()

except Exception as e:
    metrics_dict["roc_auc"] = None
    print("ROC computation failed:", e)

# ======================
# SAVE METRICS
# ======================

with open(METRICS_PATH, "w") as f:
    json.dump(metrics_dict, f, indent=4)

print(f"Metrics saved at {METRICS_PATH}")

# ======================
# SAVE PER-POST PROBABILITIES (FOR FUSION)
# ======================

probability_output = []

for idx, text in enumerate(texts):
    entry = {
        "text": text,
        "true_label": int(y_true[idx]),
        "predicted_label": int(y_pred[idx]),
        "probabilities": {
            class_names[i]: float(y_probs[idx][i])
            for i in range(num_classes)
        }
    }
    probability_output.append(entry)

with open(PROBS_PATH, "w") as f:
    json.dump(probability_output, f, indent=4)

print(f"Per-post probabilities saved at {PROBS_PATH}")

# ======================
# SAVE CSV PREDICTIONS
# ======================

output_df = test_df.copy()
for i, class_name in enumerate(class_names):
    output_df[f"prob_{class_name}"] = y_probs[:, i]

output_df["predicted_label"] = y_pred
output_df.to_csv(PREDICTIONS_PATH, index=False)

print(f"Predictions CSV saved at {PREDICTIONS_PATH}")

print("✅ Emotion Test Evaluation Complete.")