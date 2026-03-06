import os
import joblib
import numpy as np
import pandas as pd
import torch
import re

from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from textblob import TextBlob

# ----------------------------
# Configuration
# ----------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "test_data.csv"

MODEL_DIR = PROJECT_ROOT / "models" / "student"
TEACHER_DIR = PROJECT_ROOT / "models" / "teacher"

OUTPUT_DIR = PROJECT_ROOT / "notebooks" /"analysis_results" / "fusion"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Load Models
# ----------------------------

print("Loading student model...")

student_model = joblib.load(MODEL_DIR / "mtkd_student_xgb.pkl")
scaler = joblib.load(MODEL_DIR / "scaler.pkl")
tfidf_vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.pkl")

# ----------------------------
# Teacher Models
# ----------------------------

teacher_models = [
    {
        "name": "mbert",
        "path": TEACHER_DIR / "mbert" / "final_model"
    },
    {
        "name": "xlmr",
        "path": TEACHER_DIR / "xlmr" / "final_model"
    },
    {
        "name": "muril",
        "path": TEACHER_DIR / "muril" / "final_model"
    }
]

loaded_teachers = []

for teacher in teacher_models:

    print(f"Loading teacher: {teacher['name']}")

    tokenizer = AutoTokenizer.from_pretrained(teacher["path"])
    model = AutoModel.from_pretrained(teacher["path"]).to(DEVICE)
    model.eval()

    loaded_teachers.append((tokenizer, model))

# ----------------------------
# Feature Functions
# ----------------------------

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


# ----------------------------
# Teacher Embeddings
# ----------------------------

def get_teacher_embeddings(tokenizer, model, texts):

    all_embeddings = []

    with torch.no_grad():

        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Encoding"):

            batch = texts[i:i+BATCH_SIZE]

            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).to(DEVICE)

            outputs = model(**enc)

            emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

            all_embeddings.append(emb)

    return np.vstack(all_embeddings)

# ----------------------------
# Load Dataset
# ----------------------------

print("\nLoading test dataset...")

df = pd.read_csv(DATA_PATH)

texts = df["text"].astype(str).tolist()
labels = df["label"].values

print(f"Test samples: {len(texts)}")

# ----------------------------
# TF-IDF Features
# ----------------------------

print("\nGenerating TF-IDF features...")

X_tfidf = tfidf_vectorizer.transform(texts).toarray()

# ----------------------------
# Handcrafted Features
# ----------------------------

print("Generating handcrafted features...")

X_hand = extract_handcrafted_features(texts)

# ----------------------------
# Teacher Embeddings
# ----------------------------

teacher_embeddings = []

for tokenizer, model in loaded_teachers:

    emb = get_teacher_embeddings(tokenizer, model, texts)

    teacher_embeddings.append(emb)

X_teacher = np.hstack(teacher_embeddings)

# ----------------------------
# Combine Features
# ----------------------------

X = np.hstack([X_teacher, X_hand, X_tfidf])

# ----------------------------
# Scaling
# ----------------------------

X = scaler.transform(X)

# ----------------------------
# Predict Probabilities
# ----------------------------

print("\nGenerating MTKD probabilities...")

probs = student_model.predict_proba(X)[:, 1]

# ----------------------------
# Save Output
# ----------------------------

output_df = pd.DataFrame({
    "text": texts,
    "label": labels,
    "P_cb": probs
})

output_path = OUTPUT_DIR / "mtkd_probs.csv"

output_df.to_csv(output_path, index=False)

print(f"\nSaved MTKD probabilities to: {output_path}")