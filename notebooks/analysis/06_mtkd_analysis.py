import pandas as pd
import numpy as np
import joblib
import json
import os
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
from textblob import TextBlob
import re

# ==========================
# Configuration
# ==========================

BATCH_SIZE = 64
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "test_data.csv"
MODEL_DIR = PROJECT_ROOT / "models" / "student"
ANALYSIS_DIR = PROJECT_ROOT / "notebooks" / "analysis_results" / "mtkd_xgboost"

ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "mtkd_student_xgb.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
TFIDF_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"

# ==========================
# Helper Functions
# ==========================

def stylometric_features(text):
    return [
        sum(1 for c in text if c.isupper()) / max(1, len(text)),
        sum(1 for c in text if c in "!?") / max(1, len(text)),
        sum(1 for c in text if c.isdigit()) / max(1, len(text)),
        len(text.split()),
        len(text)
    ]

def sentiment_polarity(text):
    return TextBlob(text).sentiment.polarity

def code_mixing_index(text):
    eng_words = sum(1 for w in text.split() if re.match(r'[a-zA-Z]+', w))
    total_words = max(1, len(text.split()))
    return eng_words / total_words

def extract_handcrafted_features(texts):
    features = []
    for text in texts:
        feats = [
            sentiment_polarity(text),
            code_mixing_index(text)
        ]
        feats.extend(stylometric_features(text))
        features.append(feats)
    return np.array(features)

def get_teacher_embeddings(model, tokenizer, texts):
    all_embeddings = []
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Encoding"):
            batch = texts[i:i+BATCH_SIZE]
            encodings = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).to(DEVICE)

            outputs = model(**encodings)
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            all_embeddings.append(embeddings)

    return np.vstack(all_embeddings)

# ==========================
# Load Data
# ==========================

print(f"Loading test data from: {DATA_PATH}")
test_df = pd.read_csv(DATA_PATH)

texts = test_df["text"].astype(str).tolist()
y_test = test_df["label"].values

print(f"Test samples: {len(test_df)}")

# ==========================
# Load Saved Components
# ==========================

print("Loading student model...")
model = joblib.load(MODEL_PATH)

print("Loading scaler...")
scaler = joblib.load(SCALER_PATH)

print("Loading TF-IDF vectorizer...")
tfidf_vectorizer = joblib.load(TFIDF_PATH)

# ==========================
# TF-IDF Features
# ==========================

X_test_tfidf = tfidf_vectorizer.transform(texts).toarray()

# ==========================
# Teacher Embeddings
# ==========================

teacher_models_info = [
    {"name": "mbert", "path": PROJECT_ROOT / "models" / "teacher" / "mbert" / "final_model"},
    {"name": "xlmr", "path": PROJECT_ROOT / "models" / "teacher" / "xlmr" / "final_model"},
    {"name": "muril", "path": PROJECT_ROOT / "models" / "teacher" / "muril" / "final_model"},
]

test_teacher_embeddings = []

for teacher in teacher_models_info:
    print(f"\nLoading teacher: {teacher['name']}")
    tokenizer = AutoTokenizer.from_pretrained(teacher["path"])
    model_teacher = AutoModel.from_pretrained(teacher["path"])

    emb = get_teacher_embeddings(model_teacher, tokenizer, texts)
    test_teacher_embeddings.append(emb)

X_test_teacher = np.hstack(test_teacher_embeddings)

# ==========================
# Handcrafted Features
# ==========================

X_test_hand = extract_handcrafted_features(texts)

# ==========================
# Combine All Features
# ==========================

X_test = np.hstack([X_test_teacher, X_test_hand, X_test_tfidf])

# Apply same scaler
X_test = scaler.transform(X_test)

# ==========================
# Predictions
# ==========================

y_pred = model.predict(X_test)
y_probs = model.predict_proba(X_test)[:, 1]

# ==========================
# Metrics
# ==========================

accuracy = accuracy_score(y_test, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(
    y_test, y_pred, average="weighted"
)
roc_auc = roc_auc_score(y_test, y_probs)

cm = confusion_matrix(y_test, y_pred)

false_positives = int(((y_pred == 1) & (y_test == 0)).sum())
false_negatives = int(((y_pred == 0) & (y_test == 1)).sum())

model_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)

# ==========================
# Save Confusion Matrix
# ==========================

plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix - MTKD XGBoost")
plt.colorbar()
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig(ANALYSIS_DIR / "confusion_matrix.png")
plt.close()

# ==========================
# Save ROC Curve
# ==========================

fpr, tpr, _ = roc_curve(y_test, y_probs)

plt.figure()
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - MTKD XGBoost")
plt.savefig(ANALYSIS_DIR / "roc_curve.png")
plt.close()

# ==========================
# Save Metrics JSON
# ==========================

metrics = {
    "model_name": "mtkd_xgboost",
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "roc_auc": float(roc_auc),
    "model_size_mb": float(model_size_mb),
    "false_positives": false_positives,
    "false_negatives": false_negatives
}

with open(ANALYSIS_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("\nEvaluation Saved in:", ANALYSIS_DIR)
print(json.dumps(metrics, indent=4))
