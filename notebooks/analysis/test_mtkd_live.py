import os
import joblib
import numpy as np
import torch
import re
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from textblob import TextBlob
from sklearn.preprocessing import StandardScaler

# ==========================
# Configuration
# ==========================

BATCH_SIZE = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "models" / "student"
TEACHER_DIR = PROJECT_ROOT / "models" / "teacher"

# ==========================
# Load Saved Components
# ==========================

print("Loading student model...")
student_model = joblib.load(MODEL_DIR / "mtkd_student_xgb.pkl")

print("Loading scaler...")
scaler = joblib.load(MODEL_DIR / "scaler.pkl")

print("Loading TF-IDF...")
tfidf_vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.pkl")

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

def extract_handcrafted_features(text):
    feats = [
        sentiment_polarity(text),
        code_mixing_index(text)
    ]
    feats.extend(stylometric_features(text))
    return np.array(feats).reshape(1, -1)

def get_teacher_embedding(model, tokenizer, text):
    model.to(DEVICE)
    model.eval()
    with torch.no_grad():
        enc = tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(DEVICE)
        output = model(**enc)
        embedding = output.last_hidden_state.mean(dim=1)
    return embedding.cpu().numpy()

# ==========================
# Load Teachers
# ==========================

teachers = []

teacher_configs = [
    ("mbert", TEACHER_DIR / "mbert" / "final_model"),
    ("xlmr", TEACHER_DIR / "xlmr" / "final_model"),
    ("muril", TEACHER_DIR / "muril" / "final_model"),
]

for name, path in teacher_configs:
    print(f"Loading teacher: {name}")
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModel.from_pretrained(path)
    teachers.append((name, tokenizer, model))

print("\n✅ MTKD Live Testing Ready!")
print("Type 'exit' to quit.\n")

# ==========================
# Interactive Loop
# ==========================

while True:
    text = input("Enter text: ")

    if text.lower() == "exit":
        break

    # Teacher embeddings
    embeddings_list = []
    for name, tokenizer, model in teachers:
        emb = get_teacher_embedding(model, tokenizer, text)
        embeddings_list.append(emb)

    combined_embeddings = np.hstack(embeddings_list)

    # TF-IDF features
    tfidf_feats = tfidf_vectorizer.transform([text]).toarray()

    # Handcrafted features
    handcrafted_feats = extract_handcrafted_features(text)

    # Final input
    X = np.hstack([combined_embeddings, handcrafted_feats, tfidf_feats])
    X = scaler.transform(X)

    # Prediction
    pred = student_model.predict(X)[0]
    prob = student_model.predict_proba(X)[0][1]

    label = "Cyberbullying" if pred == 1 else "Non-Cyberbullying"

    print(f"\nPrediction: {label}")
    print(f"Confidence: {prob:.4f}\n")
