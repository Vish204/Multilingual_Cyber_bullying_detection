import torch
import numpy as np
import pandas as pd
import json
import re
from pathlib import Path
from textblob import TextBlob
from scipy.sparse import hstack

# ------------------------------------------------
# Paths
# ------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[3]

KEYWORDS_DIR = BASE_DIR / "resources/keywords/multilingual_keywords"

VOCAB_PATH = BASE_DIR / "models/sarcasm/vocab.json"

MAX_LEN = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------
# Load sarcasm vocab
# ------------------------------------------------

with open(VOCAB_PATH, "r") as f:
    SARCASM_VOCAB = json.load(f)

PAD_ID = SARCASM_VOCAB["<PAD>"]
UNK_ID = SARCASM_VOCAB["<UNK>"]

# ------------------------------------------------
# Load multilingual keywords
# ------------------------------------------------

keywords = set()

for file in KEYWORDS_DIR.glob("*.json"):

    with open(file, encoding="utf-8") as f:
        data = json.load(f)

    for kw in data["keywords"]:
        keywords.add(kw.lower())

# ------------------------------------------------
# Handcrafted features
# ------------------------------------------------

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


def extract_handcrafted_features(text):

    feats = [
        sentiment_polarity(text),
        code_mixing_index(text)
    ]

    feats.extend(stylometric_features(text))

    return np.array(feats)


# ------------------------------------------------
# Keyword features
# ------------------------------------------------

def keyword_features(text):

    words = text.lower().split()

    count = sum(1 for w in words if w in keywords)

    present = 1 if count > 0 else 0

    ratio = count / max(1, len(words))

    return np.array([present, count, ratio])


# ------------------------------------------------
# Sarcasm encoding
# ------------------------------------------------

def encode_sarcasm_text(text):

    tokens = str(text).lower().strip().split()

    ids = [SARCASM_VOCAB.get(word, UNK_ID) for word in tokens]

    if len(ids) < MAX_LEN:
        ids += [PAD_ID] * (MAX_LEN - len(ids))
    else:
        ids = ids[:MAX_LEN]

    return torch.tensor(ids).unsqueeze(0)


# ------------------------------------------------
# Student_V2 prediction
# ------------------------------------------------

def predict_cyberbullying(text, models):

    student = models["student"]
    word_vec = models["word_vectorizer"]
    char_vec = models["char_vectorizer"]
    scaler = models["scaler"]

    text = str(text).lower()

    # Word TF-IDF
    X_word = word_vec.transform([text])

    # Char TF-IDF
    X_char = char_vec.transform([text])

    # Handcrafted
    X_hand = extract_handcrafted_features(text).reshape(1, -1)

    # Keyword
    X_key = keyword_features(text).reshape(1, -1)

    X_numeric = np.hstack([X_hand, X_key])

    # Scale numeric
    X_numeric = scaler.transform(X_numeric)

    # Combine
    X = hstack([X_word, X_char, X_numeric])

    # Predict
    p_cb = student.predict(X)[0]

    # clip to probability range
    p_cb = float(np.clip(p_cb, 0, 1))

    return p_cb


# ------------------------------------------------
# Sarcasm prediction
# ------------------------------------------------

def predict_sarcasm(text, sarcasm_model):

    input_ids = encode_sarcasm_text(text)

    with torch.no_grad():
        prob = sarcasm_model(input_ids).item()

    return float(prob)


# ------------------------------------------------
# Emotion prediction
# ------------------------------------------------

def predict_emotion(text, emotion_tokenizer, emotion_model):

    inputs = emotion_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    with torch.no_grad():

        outputs = emotion_model(**inputs)

        logits = outputs.logits.squeeze(0)

        probs = torch.softmax(logits, dim=0).cpu().numpy()

    p_neutral = probs[0]
    p_aggression = probs[1]
    p_distress = probs[2]

    p_emotion = float(p_aggression + p_distress)

    return p_emotion


# ------------------------------------------------
# Run all component models
# ------------------------------------------------

def run_component_predictions(text_list, models):

    sarcasm_model = models["sarcasm"]

    emotion_tokenizer = models["emotion_tokenizer"]
    emotion_model = models["emotion_model"]

    results = []

    for text in text_list:

        p_cb = predict_cyberbullying(text, models)

        p_sarcasm = predict_sarcasm(text, sarcasm_model)

        p_emotion = predict_emotion(text, emotion_tokenizer, emotion_model)

        results.append({
            "text": text,
            "p_cb": p_cb,
            "p_sarcasm": p_sarcasm,
            "p_emotion": p_emotion
        })

    return pd.DataFrame(results)