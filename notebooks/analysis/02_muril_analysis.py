import torch
import numpy as np
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
import json
import os

from pathlib import Path
from datasets import Dataset
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer

# ===============================
# 1. PATH SETUP
# ===============================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS_ROOT = Path(__file__).resolve().parents[1]

MODEL_NAME = "muril"
MODEL_PATH = PROJECT_ROOT / "models" / "teacher" / MODEL_NAME / "final_model"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "test_data.csv"

RESULTS_DIR = NOTEBOOKS_ROOT / "analysis_results" / MODEL_NAME
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============================
# 2. LOAD MODEL + TOKENIZER
# ===============================

print("Loading model from:", MODEL_PATH)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()


trainer = Trainer(model=model)

# ===============================
# 3. LOAD TEST DATA
# ===============================

test_df = pd.read_csv(DATA_PATH)
test_df["label"] = test_df["label"].astype(int)

texts = test_df["text"].tolist()
labels = test_df["label"].tolist()

# ===============================
# 4. TOKENIZE
# ===============================

encodings = tokenizer(
    texts,
    truncation=True,
    padding=True,
    max_length=256
)

test_dataset = Dataset.from_dict({
    "input_ids": encodings["input_ids"],
    "attention_mask": encodings["attention_mask"],
    "labels": labels
})

# ===============================
# 5. PREDICTIONS
# ===============================

predictions = trainer.predict(test_dataset)

y_true = predictions.label_ids
y_pred = np.argmax(predictions.predictions, axis=1)

probs = torch.nn.functional.softmax(
    torch.tensor(predictions.predictions), dim=1
).numpy()

# ===============================
# 6. CONFUSION MATRIX
# ===============================

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=["non_toxic", "toxic"],
            yticklabels=["non_toxic", "toxic"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"{MODEL_NAME.upper()} Confusion Matrix")
plt.savefig(RESULTS_DIR / "confusion_matrix.png")
plt.close()

# ===============================
# 7. CLASSIFICATION REPORT
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
# 8. FALSE POSITIVES / NEGATIVES
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
# 9. HINDI ERROR ANALYSIS
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
# 10. ROC-AUC
# ===============================

auc = roc_auc_score(y_true, probs[:,1])

# ===============================
# 11. SAVE METRICS JSON
# ===============================

results = {
    "model_name": MODEL_NAME,
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "roc_auc": float(auc),
    "false_positives": len(false_positives),
    "false_negatives": len(false_negatives),
    "hindi_errors": len(hindi_errors)
}

with open(RESULTS_DIR / "metrics.json", "w") as f:
    json.dump(results, f, indent=4)

print("\nEvaluation Saved in:", RESULTS_DIR)
print(json.dumps(results, indent=4))
