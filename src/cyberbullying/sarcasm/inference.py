import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    classification_report
)

from model import SarcasmModel

# =========================
# PATHS
# =========================

BASE_DIR = Path(__file__).resolve().parents[3]

SPLIT_DIR = BASE_DIR / "data" / "sarcasm" / "splits"
TEST_PATH = SPLIT_DIR / "test.csv"

MODEL_DIR = BASE_DIR / "models" / "sarcasm"
MODEL_PATH = MODEL_DIR / "best_model.pt"
VOCAB_PATH = MODEL_DIR / "vocab.json"

RESULTS_DIR = BASE_DIR / "notebooks" / "analysis_results" / "sarcasm"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PREDICTIONS_PATH = RESULTS_DIR / "sarcasm_predictions.csv"
PROBS_PATH = RESULTS_DIR / "sarcasm_test_probabilities.json"
METRICS_PATH = RESULTS_DIR / "metrics.json"

CM_PATH = RESULTS_DIR / "confusion_matrix.png"
ROC_PATH = RESULTS_DIR / "roc_curve.png"

# =========================
# CONFIG
# =========================

MAX_LEN = 50
EMBED_DIM = 128
HIDDEN_DIM = 128
BATCH_SIZE = 64

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", DEVICE)

# =========================
# LOAD TEST DATA
# =========================

test_df = pd.read_csv(TEST_PATH)

texts = test_df["text"].tolist()
labels = test_df["label"].tolist()

print("Test samples:", len(texts))

# =========================
# LOAD VOCAB
# =========================

with open(VOCAB_PATH) as f:
    vocab = json.load(f)

VOCAB_SIZE = len(vocab)

print("Loaded vocab size:", VOCAB_SIZE)

# =========================
# TOKENIZER
# =========================

def tokenize(text):

    tokens = [vocab.get(word, 0) for word in text.strip().split()]

    if len(tokens) < MAX_LEN:
        tokens += [0] * (MAX_LEN - len(tokens))
    else:
        tokens = tokens[:MAX_LEN]

    return tokens


# =========================
# CREATE DATASET
# =========================

class SarcasmDataset(torch.utils.data.Dataset):

    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):

        tokens = tokenize(self.texts[idx])

        return (
            torch.tensor(tokens, dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.float)
        )


dataset = SarcasmDataset(texts, labels)

loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# =========================
# LOAD MODEL
# =========================

model = torch.load(MODEL_PATH, map_location=DEVICE)

model.to(DEVICE)
model.eval()

print("Model loaded.")

# =========================
# INFERENCE
# =========================

all_probs = []
all_preds = []
all_labels = []

with torch.no_grad():

    for x_batch, y_batch in loader:

        x_batch = x_batch.to(DEVICE)

        outputs = model(x_batch).squeeze(1)

        probs = outputs.cpu().numpy()

        preds = (probs >= 0.5).astype(int)

        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_labels.extend(y_batch.numpy().tolist())


y_true = np.array(all_labels)
y_pred = np.array(all_preds)
y_prob = np.array(all_probs)

# =========================
# METRICS
# =========================

metrics = {
    "accuracy": float(accuracy_score(y_true, y_pred)),
    "precision": float(precision_score(y_true, y_pred)),
    "recall": float(recall_score(y_true, y_pred)),
    "f1_score": float(f1_score(y_true, y_pred)),
    "roc_auc": float(roc_auc_score(y_true, y_prob)),
    "classification_report": classification_report(
        y_true,
        y_pred,
        output_dict=True
    )
}

# =========================
# CONFUSION MATRIX
# =========================

cm = confusion_matrix(y_true, y_pred)

metrics["confusion_matrix"] = cm.tolist()

plt.figure(figsize=(5,4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Not Sarcastic","Sarcastic"],
    yticklabels=["Not Sarcastic","Sarcastic"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Sarcasm Confusion Matrix")

plt.tight_layout()
plt.savefig(CM_PATH)
plt.close()

print("Confusion matrix saved.")

# =========================
# ROC CURVE
# =========================

fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc = roc_auc_score(y_true, y_prob)

plt.figure(figsize=(6,5))

plt.plot(
    fpr,
    tpr,
    label=f"ROC curve (AUC = {roc_auc:.3f})"
)

plt.plot([0,1],[0,1],'k--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Sarcasm ROC Curve")

plt.legend(loc="lower right")

plt.tight_layout()
plt.savefig(ROC_PATH)
plt.close()

print("ROC curve saved.")

# =========================
# SAVE METRICS
# =========================

with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=4)

print("Metrics saved.")

# =========================
# SAVE CSV PREDICTIONS
# =========================

output_df = test_df.copy()

output_df["predicted_label"] = y_pred
output_df["prob_sarcasm"] = y_prob

output_df.to_csv(PREDICTIONS_PATH, index=False)

print("Predictions CSV saved.")

# =========================
# SAVE PROBABILITIES (FOR FUSION)
# =========================

probability_output = []

for i in range(len(texts)):

    probability_output.append({
        "text": texts[i],
        "true_label": int(y_true[i]),
        "predicted_label": int(y_pred[i]),
        "probabilities": {
            "not_sarcasm": float(1 - y_prob[i]),
            "sarcasm": float(y_prob[i])
        }
    })


with open(PROBS_PATH, "w") as f:
    json.dump(probability_output, f, indent=4)

print("Per-post probabilities saved (for fusion).")

print("✅ Sarcasm inference complete.")