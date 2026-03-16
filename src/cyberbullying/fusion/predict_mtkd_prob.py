import joblib
import numpy as np
import pandas as pd
import re
import json
import unicodedata

from pathlib import Path
from scipy.sparse import hstack, csr_matrix
from textblob import TextBlob


# =========================================================
# CONFIG
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "test_data.csv"

MODEL_DIR = PROJECT_ROOT / "models" / "student_v2"

KEYWORDS_DIR = PROJECT_ROOT / "resources" / "keywords" / "multilingual_keywords"

OUTPUT_DIR = PROJECT_ROOT / "notebooks" / "analysis_results" / "fusion"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# TEXT NORMALIZATION (Google multilingual trick)
# =========================================================

def normalize_text(text):

    text = unicodedata.normalize("NFKC", text)
    text = text.lower()

    return text


# =========================================================
# LOAD MODEL ARTIFACTS
# =========================================================

print("Loading Student V2 artifacts...")

student_model = joblib.load(MODEL_DIR / "student_xgb_model.pkl")

word_vectorizer = joblib.load(MODEL_DIR / "word_tfidf.pkl")
char_vectorizer = joblib.load(MODEL_DIR / "char_tfidf.pkl")

scaler = joblib.load(MODEL_DIR / "scaler.pkl")


# =========================================================
# LOAD KEYWORDS
# =========================================================

print("Loading keyword dictionaries...")

keywords = set()

for file in KEYWORDS_DIR.glob("*.json"):

    with open(file, encoding="utf-8") as f:

        data = json.load(f)

    for kw in data["keywords"]:

        keywords.add(kw.lower())

# regex keyword matcher
keyword_pattern = re.compile(
    r"\b(" + "|".join(map(re.escape, keywords)) + r")\b",
    re.IGNORECASE
)


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

    feats = []

    for text in texts:

        f = [
            sentiment_polarity(text),
            code_mixing_index(text)
        ]

        f.extend(stylometric_features(text))

        feats.append(f)

    return np.array(feats)


# =========================================================
# KEYWORD FEATURES (regex based)
# =========================================================

def keyword_features(texts):

    feats = []

    for text in texts:

        matches = keyword_pattern.findall(text)

        count = len(matches)

        present = 1 if count > 0 else 0

        ratio = count / max(1, len(text.split()))

        feats.append([present, count, ratio])

    return np.array(feats)


# =========================================================
# LOAD DATASET
# =========================================================

print("Loading test dataset...")

df = pd.read_csv(DATA_PATH)

texts = df["text"].astype(str).tolist()
labels = df["label"].values

# apply normalization
texts = [normalize_text(t) for t in texts]

print("Samples:", len(texts))


# =========================================================
# TF-IDF FEATURES
# =========================================================

print("Generating TF-IDF features...")

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

print("Combining features...")

X = hstack([

    X_word,
    X_char,
    X_numeric

])


# =========================================================
# PREDICT PROBABILITIES
# =========================================================

print("Generating Student V2 probabilities...")

probs = student_model.predict(X)

# safety clip
probs = np.clip(probs, 0, 1)


# =========================================================
# SAVE OUTPUT
# =========================================================

output_df = pd.DataFrame({

    "text": texts,
    "label": labels,
    "P_cb": probs

})

output_path = OUTPUT_DIR / "mtkd_v2_probs.csv"

output_df.to_csv(output_path, index=False)

print("\nSaved probabilities to:", output_path)