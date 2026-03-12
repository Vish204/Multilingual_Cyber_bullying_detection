import pandas as pd
import numpy as np
import joblib
import json
import re
from pathlib import Path
import os

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve
)

from scipy.sparse import hstack, csr_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from textblob import TextBlob


# =========================================================
# CONFIG
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEST_PATH = PROJECT_ROOT / "data/processed/test_data.csv"

MODEL_DIR = PROJECT_ROOT / "models/student_v2"

KEYWORDS_DIR = PROJECT_ROOT / "resources/keywords/multilingual_keywords"

RESULT_DIR = PROJECT_ROOT / "notebooks/analysis_results/student_v2"
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# LOAD DATA
# =========================================================

print("Loading test dataset...")

test_df = pd.read_csv(TEST_PATH)

texts = test_df["text"].astype(str).tolist()
# y_true = test_df["label"].values
y_true = test_df["label"].astype(int).values


# =========================================================
# LOAD MODEL ARTIFACTS
# =========================================================

print("Loading model artifacts...")

student = joblib.load(MODEL_DIR / "student_xgb_model.pkl")
word_vectorizer = joblib.load(MODEL_DIR / "word_tfidf.pkl")
char_vectorizer = joblib.load(MODEL_DIR / "char_tfidf.pkl")
scaler = joblib.load(MODEL_DIR / "scaler.pkl")


# =========================================================
# LOAD KEYWORDS
# =========================================================

keywords = set()

for file in KEYWORDS_DIR.glob("*.json"):
    with open(file, encoding="utf-8") as f:
        data = json.load(f)

    for kw in data["keywords"]:
        keywords.add(kw.lower())


# =========================================================
# FEATURE FUNCTIONS
# =========================================================

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

    eng = sum(1 for w in text.split() if re.match(r"[a-zA-Z]+", w))
    total = max(1, len(text.split()))

    return eng / total


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


# =========================================================
# KEYWORD FEATURES
# =========================================================

def keyword_features(texts):

    feats = []

    for text in texts:

        words = text.lower().split()

        count = sum(1 for w in words if w in keywords)

        present = 1 if count > 0 else 0

        ratio = count / max(1, len(words))

        feats.append([present, count, ratio])

    return np.array(feats)


# =========================================================
# TFIDF FEATURES
# =========================================================

print("Transforming TF-IDF...")

X_word = word_vectorizer.transform(texts)
X_char = char_vectorizer.transform(texts)


# =========================================================
# NUMERIC FEATURES
# =========================================================

print("Extracting numeric features...")

X_hand = extract_handcrafted_features(texts)
X_key = keyword_features(texts)

X_numeric = np.hstack([X_hand, X_key])

X_numeric = scaler.transform(X_numeric)

X_numeric = csr_matrix(X_numeric)


# =========================================================
# COMBINE FEATURES
# =========================================================

X_test = hstack([

    X_word,
    X_char,
    X_numeric

])


# =========================================================
# PREDICTIONS
# =========================================================

print("Running predictions...")

y_prob = student.predict(X_test)

y_pred = (y_prob >= 0.5).astype(int)

print("Unique predictions:", np.unique(y_pred))
# =========================================================
# METRICS
# =========================================================

report = classification_report(
    y_true,
    y_pred,
    output_dict=True,
    zero_division=0
)

auc = roc_auc_score(y_true, y_prob)

cm = confusion_matrix(y_true, y_pred)

false_pos = int(((y_pred == 1) & (y_true == 0)).sum())
false_neg = int(((y_pred == 0) & (y_true == 1)).sum())

model_size_mb = os.path.getsize(MODEL_DIR / "student_xgb_model.pkl") / (1024*1024)

# =========================================================
# SAVE METRICS
# =========================================================

metrics = {

    "model_name": "student_v2",

    "accuracy": report["accuracy"],
    "precision": report["1"]["precision"],
    "recall": report["1"]["recall"],
    "f1_score": report["1"]["f1-score"],

    "roc_auc": float(auc),

    "model_size_mb": float(model_size_mb),
    "false_positives": false_pos,
    "false_negatives": false_neg
}

with open(RESULT_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)


# =========================================================
# CONFUSION MATRIX
# =========================================================

np.save(RESULT_DIR / "confusion_matrix.npy", cm)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Non-CB","CB"],
    yticklabels=["Non-CB","CB"]
)

plt.title("Confusion Matrix - Student V2")

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(RESULT_DIR / "confusion_matrix.png")

plt.close()


# =========================================================
# ROC CURVE
# =========================================================

fpr, tpr, _ = roc_curve(y_true, y_prob)

plt.figure()

plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")

plt.plot([0,1],[0,1],'--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve - Student V2")

plt.legend()

plt.tight_layout()

plt.savefig(RESULT_DIR / "roc_curve.png")

plt.close()


# =========================================================
# PRECISION RECALL CURVE
# =========================================================

precision, recall, _ = precision_recall_curve(y_true, y_prob)

plt.figure()

plt.plot(recall, precision)

plt.xlabel("Recall")
plt.ylabel("Precision")

plt.title("Precision-Recall Curve")

plt.tight_layout()

plt.savefig(RESULT_DIR / "precision_recall_curve.png")

plt.close()


# =========================================================
# PROBABILITY DISTRIBUTION
# =========================================================

plt.figure()

sns.histplot(y_prob, bins=40)

plt.title("Prediction Probability Distribution")

plt.xlabel("Predicted Probability")

plt.tight_layout()

plt.savefig(RESULT_DIR / "probability_distribution.png")

plt.close()


print("Analysis completed.")
print("Results saved to:", RESULT_DIR)