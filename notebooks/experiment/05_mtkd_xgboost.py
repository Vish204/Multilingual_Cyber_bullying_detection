import os
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
from transformers import AutoTokenizer, AutoModel
import torch
from textblob import TextBlob
import re

# ------------------------
# Configuration
# ------------------------

BATCH_SIZE = 64
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_SAVE_DIR = PROJECT_ROOT / "models" / "student"

MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------
# Helper Functions
# ------------------------

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
            batch_texts = texts[i:i+BATCH_SIZE]
            encodings = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).to(DEVICE)

            outputs = model(**encodings)
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            all_embeddings.append(embeddings)

    return np.vstack(all_embeddings)

# ------------------------
# Load Data
# ------------------------

print("\nLoading data...")

train_df = pd.read_csv(DATA_DIR / "train_data.csv")
val_df = pd.read_csv(DATA_DIR / "val_data.csv")

train_texts = train_df["text"].astype(str).tolist()
val_texts = val_df["text"].astype(str).tolist()

y_train = train_df["label"].values
y_val = val_df["label"].values

print(f"Train samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")

# ------------------------
# TF-IDF (Fitted Here Only)
# ------------------------

print("\nFitting TF-IDF vectorizer...")

tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X_train_tfidf = tfidf_vectorizer.fit_transform(train_texts).toarray()
X_val_tfidf = tfidf_vectorizer.transform(val_texts).toarray()

print(f"TF-IDF dimension: {X_train_tfidf.shape[1]}")

# ------------------------
# Teacher Embeddings
# ------------------------

teacher_models_info = [
    {"name": "mbert", "path": "models/teacher/mbert/final_model"},
    {"name": "xlmr", "path": "models/teacher/xlmr/final_model"},
    {"name": "muril", "path": "models/teacher/muril/final_model"},
]

train_teacher_embeddings = []
val_teacher_embeddings = []

for teacher in teacher_models_info:
    print(f"\nLoading teacher: {teacher['name']}")

    tokenizer = AutoTokenizer.from_pretrained(teacher["path"])
    model = AutoModel.from_pretrained(teacher["path"])

    train_emb = get_teacher_embeddings(model, tokenizer, train_texts)
    val_emb = get_teacher_embeddings(model, tokenizer, val_texts)

    train_teacher_embeddings.append(train_emb)
    val_teacher_embeddings.append(val_emb)

# Combine embeddings
X_train_teacher = np.hstack(train_teacher_embeddings)
X_val_teacher = np.hstack(val_teacher_embeddings)

# ------------------------
# Handcrafted Features
# ------------------------

print("\nExtracting handcrafted features...")

X_train_hand = extract_handcrafted_features(train_texts)
X_val_hand = extract_handcrafted_features(val_texts)

# ------------------------
# Combine All Features
# ------------------------

X_train = np.hstack([X_train_teacher, X_train_hand, X_train_tfidf])
X_val = np.hstack([X_val_teacher, X_val_hand, X_val_tfidf])

print(f"\nFinal feature dimension: {X_train.shape[1]}")

# ------------------------
# Scaling
# ------------------------

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# ------------------------
# Train Student Model
# ------------------------

print("\nTraining XGBoost student model...")

student_model = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

student_model.fit(X_train, y_train)

# ------------------------
# Evaluate
# ------------------------

val_preds = student_model.predict(X_val)

val_acc = accuracy_score(y_val, val_preds)
val_f1 = f1_score(y_val, val_preds)

print("\nValidation Results")
print(f"Accuracy: {val_acc:.4f}")
print(f"F1 Score: {val_f1:.4f}")

# ------------------------
# Save Everything
# ------------------------

joblib.dump(student_model, MODEL_SAVE_DIR / "mtkd_student_xgb.pkl")
joblib.dump(scaler, MODEL_SAVE_DIR / "scaler.pkl")
joblib.dump(tfidf_vectorizer, MODEL_SAVE_DIR / "tfidf_vectorizer.pkl")

print("\n✅ MTKD student model, scaler, and TF-IDF saved successfully!")
