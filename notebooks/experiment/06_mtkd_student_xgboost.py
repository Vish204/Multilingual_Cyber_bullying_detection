import pandas as pd
import numpy as np
import joblib
import json
import re
from pathlib import Path
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack

from xgboost import XGBRegressor

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from textblob import TextBlob


# =========================================================
# CONFIG
# =========================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = PROJECT_ROOT / "data/processed/train_data.csv"
VAL_PATH = PROJECT_ROOT / "data/processed/val_data.csv"

KEYWORDS_DIR = PROJECT_ROOT / "resources/keywords/multilingual_keywords"

SOFT_DIR = PROJECT_ROOT / "data/probs"
SOFT_DIR.mkdir(exist_ok=True, parents=True)

TRAIN_SOFT = SOFT_DIR / "train_soft_labels.npy"
VAL_SOFT = SOFT_DIR / "val_soft_labels.npy"

MODEL_DIR = PROJECT_ROOT / "models/student_v2"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# LOAD DATA
# =========================================================

print("Loading datasets...")

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)

train_texts = train_df["text"].astype(str).tolist()
val_texts = val_df["text"].astype(str).tolist()

print("Train samples:", len(train_texts))
print("Val samples:", len(val_texts))


# =========================================================
# LOAD MULTILINGUAL KEYWORDS
# =========================================================

print("Loading multilingual keyword files...")

keywords = set()

for file in KEYWORDS_DIR.glob("*.json"):

    with open(file, encoding="utf-8") as f:
        data = json.load(f)

    for kw in data["keywords"]:
        keywords.add(kw.lower())

print("Total keywords loaded:", len(keywords))


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
# WORD TFIDF
# =========================================================

print("Training Word TF-IDF...")

word_vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2),
    min_df=2,
    sublinear_tf=True
)

X_train_word = word_vectorizer.fit_transform(train_texts)
X_val_word = word_vectorizer.transform(val_texts)


# =========================================================
# CHAR TFIDF
# =========================================================

print("Training Char TF-IDF...")

char_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3,5),
    max_features=3000,
    sublinear_tf=True
)

X_train_char = char_vectorizer.fit_transform(train_texts)
X_val_char = char_vectorizer.transform(val_texts)


# =========================================================
# HANDCRAFTED FEATURES
# =========================================================

print("Extracting handcrafted features...")

X_train_hand = extract_handcrafted_features(train_texts)
X_val_hand = extract_handcrafted_features(val_texts)


# =========================================================
# KEYWORD FEATURES
# =========================================================

print("Extracting keyword features...")

X_train_key = keyword_features(train_texts)
X_val_key = keyword_features(val_texts)


# =========================================================
# SCALE NUMERIC FEATURES
# =========================================================

scaler = StandardScaler()

X_train_numeric = np.hstack([X_train_hand, X_train_key])
X_val_numeric = np.hstack([X_val_hand, X_val_key])

X_train_numeric = scaler.fit_transform(X_train_numeric)
X_val_numeric = scaler.transform(X_val_numeric)


# =========================================================
# COMBINE FEATURES
# =========================================================

print("Combining features...")

X_train = hstack([
    X_train_word,
    X_train_char,
    X_train_numeric
])

X_val = hstack([
    X_val_word,
    X_val_char,
    X_val_numeric
])

print("Final feature dimension:", X_train.shape)


# =========================================================
# LOAD TEACHER MODELS
# =========================================================

teacher_info = [
    {"name":"mbert","path":PROJECT_ROOT / "models/teacher/mbert/final_model"},
    {"name":"xlmr","path":PROJECT_ROOT / "models/teacher/xlmr/final_model"},
    {"name":"muril","path":PROJECT_ROOT / "models/teacher/muril/final_model"}
]

teachers = []

for t in teacher_info:

    print("Loading teacher:", t["name"])

    tokenizer = AutoTokenizer.from_pretrained(t["path"])
    model = AutoModelForSequenceClassification.from_pretrained(t["path"])

    model.to(DEVICE)
    model.eval()

    teachers.append((tokenizer, model))


# =========================================================
# SOFT LABEL GENERATION
# =========================================================

def generate_soft_labels(texts):

    all_probs = []

    for start in tqdm(range(0, len(texts), BATCH_SIZE)):

        batch = texts[start:start+BATCH_SIZE]

        teacher_batch = []

        for tokenizer, model in teachers:

            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).to(DEVICE)

            with torch.no_grad():

                out = model(**enc)

                probs = torch.softmax(out.logits, dim=1)[:,1]

            teacher_batch.append(probs.cpu().numpy())

        teacher_batch = np.vstack(teacher_batch)

        avg = np.mean(teacher_batch, axis=0)

        all_probs.extend(avg)

    return np.array(all_probs)


# =========================================================
# LOAD OR GENERATE SOFT LABELS
# =========================================================

if TRAIN_SOFT.exists():

    print("Loading cached train soft labels")

    y_train = np.load(TRAIN_SOFT)

else:

    print("Generating train soft labels")

    y_train = generate_soft_labels(train_texts)

    np.save(TRAIN_SOFT, y_train)


if VAL_SOFT.exists():

    print("Loading cached val soft labels")

    y_val = np.load(VAL_SOFT)

else:

    print("Generating val soft labels")

    y_val = generate_soft_labels(val_texts)

    np.save(VAL_SOFT, y_val)


# =========================================================
# TRAIN STUDENT MODEL
# =========================================================

print("Training student model...")

student = XGBRegressor(

    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    random_state=42
)

student.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=True
)


# =========================================================
# SAVE ARTIFACTS
# =========================================================

print("Saving artifacts...")

joblib.dump(student, MODEL_DIR / "student_xgb_model.pkl")
joblib.dump(word_vectorizer, MODEL_DIR / "word_tfidf.pkl")
joblib.dump(char_vectorizer, MODEL_DIR / "char_tfidf.pkl")
joblib.dump(scaler, MODEL_DIR / "scaler.pkl")

print("Student training completed successfully!")