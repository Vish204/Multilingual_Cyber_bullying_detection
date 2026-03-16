import pandas as pd
import numpy as np
import joblib
import json
import re
import unicodedata
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

MODEL_DIR = PROJECT_ROOT / "models/student_v2"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

KEYWORDS_PKL = MODEL_DIR / "keywords.pkl"

SOFT_DIR = PROJECT_ROOT / "data/probs"
SOFT_DIR.mkdir(exist_ok=True, parents=True)

TRAIN_SOFT = SOFT_DIR / "train_soft_labels.npy"
VAL_SOFT = SOFT_DIR / "val_soft_labels.npy"


# =========================================================
# TEXT NORMALIZATION (Google Toxicity trick)
# =========================================================

def normalize_text(text):

    text = unicodedata.normalize("NFKC", text)

    text = text.lower()

    text = re.sub(r"\s+", " ", text)

    return text


# =========================================================
# LOAD DATA
# =========================================================

print("Loading datasets...")

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)

train_texts = [normalize_text(t) for t in train_df["text"].astype(str).tolist()]
val_texts = [normalize_text(t) for t in val_df["text"].astype(str).tolist()]

print("Train samples:", len(train_texts))
print("Val samples:", len(val_texts))


# =========================================================
# KEYWORD EXTRACTION
# =========================================================

def extract_keywords_recursive(obj, keywords):

    if isinstance(obj, dict):

        for k, v in obj.items():

            if k in ["native", "roman", "english"]:

                if isinstance(v, str) and v.strip():
                    keywords.add(v.lower().strip())

            extract_keywords_recursive(v, keywords)

    elif isinstance(obj, list):

        for item in obj:
            extract_keywords_recursive(item, keywords)


def load_multilingual_keywords():

    if KEYWORDS_PKL.exists():

        print("Loading cached keywords.pkl")

        return joblib.load(KEYWORDS_PKL)

    print("Parsing multilingual keyword JSON files...")

    keywords = set()

    for file in KEYWORDS_DIR.glob("*.json"):

        before = len(keywords)

        with open(file, encoding="utf-8") as f:
            data = json.load(f)

        if "keywords" in data:

            for kw in data["keywords"]:
                if isinstance(kw, str) and kw.strip():
                    keywords.add(kw.lower().strip())

        extract_keywords_recursive(data, keywords)

        after = len(keywords)

        print(f"{file.name} → {after-before} keywords added")

    print("\nTotal unique keywords:", len(keywords))

    joblib.dump(keywords, KEYWORDS_PKL)

    print("Saved keywords.pkl")

    return keywords


keywords = load_multilingual_keywords()


# =========================================================
# BUILD REGEX PATTERNS
# =========================================================

def build_keyword_patterns(keywords):

    patterns = []

    for kw in keywords:

        chars = list(kw)

        # pattern = r"[\W_]*".join(map(re.escape, chars))
        pattern = r"\b" + re.escape(kw) + r"\b"
        patterns.append(re.compile(pattern, re.IGNORECASE))

    return patterns


keyword_patterns = build_keyword_patterns(keywords)


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
# REGEX KEYWORD FEATURES
# =========================================================

def keyword_features(texts):

    feats = []

    for text in texts:

        count = 0

        for pattern in keyword_patterns:

            if pattern.search(text):
                count += 1

        present = 1 if count > 0 else 0

        ratio = count / max(1, len(text.split()))

        feats.append([present, count, ratio])

    return np.array(feats)


# =========================================================
# WORD TFIDF (UPGRADED)
# =========================================================

print("\nTraining Word TF-IDF...")

word_vectorizer = TfidfVectorizer(
    max_features=15000,
    ngram_range=(1,2),
    min_df=2,
    sublinear_tf=True
)

X_train_word = word_vectorizer.fit_transform(train_texts)
X_val_word = word_vectorizer.transform(val_texts)


# =========================================================
# CHAR TFIDF (UPGRADED)
# =========================================================

print("Training Char TF-IDF...")

char_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3,5),
    max_features=8000,
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

print("Extracting regex keyword features...")

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

print("\nTraining student model...")

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

print("\nSaving artifacts...")

joblib.dump(student, MODEL_DIR / "student_xgb_model.pkl")
joblib.dump(word_vectorizer, MODEL_DIR / "word_tfidf.pkl")
joblib.dump(char_vectorizer, MODEL_DIR / "char_tfidf.pkl")
joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
joblib.dump(list(keywords), MODEL_DIR / "keywords.pkl")

print("\nStudent training completed successfully!")