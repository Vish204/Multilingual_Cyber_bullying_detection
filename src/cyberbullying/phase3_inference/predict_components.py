import torch
import numpy as np
import pandas as pd
import json
import re
from pathlib import Path
from textblob import TextBlob

# ------------------------------------------------
# Paths
# ------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[3]

VOCAB_PATH = BASE_DIR / "models" / "sarcasm" / "vocab.json"

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
# Emotion category mapping
# ------------------------------------------------

AGGRESSION_EMOTIONS = ["anger", "annoyance", "disgust", "disapproval"]

DISTRESS_EMOTIONS = [
    "sadness",
    "fear",
    "embarrassment",
    "remorse",
    "nervousness",
    "grief"
]

NEUTRAL_EMOTION = ["neutral"]

# ------------------------------------------------
# Handcrafted features (same as training)
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

    eng_words = sum(1 for w in text.split() if re.match(r'[a-zA-Z]+', w))
    total_words = max(1, len(text.split()))

    return eng_words / total_words


def extract_handcrafted_features(text):

    feats = [
        sentiment_polarity(text),
        code_mixing_index(text)
    ]

    feats.extend(stylometric_features(text))

    return np.array(feats)


# ------------------------------------------------
# Teacher embeddings
# ------------------------------------------------

def get_teacher_embedding(text, tokenizer, model):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    ).to(DEVICE)

    with torch.no_grad():

        outputs = model(**inputs)

        embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

    return embedding


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
# MTKD prediction
# ------------------------------------------------

def predict_mtkd(text, models):

    mtkd_model = models["mtkd"]
    vectorizer = models["vectorizer"]
    scaler = models["scaler"]

    teachers = models["teachers"]

    clean_text = str(text).lower()

    # TF-IDF
    tfidf_vec = vectorizer.transform([clean_text]).toarray()

    # Handcrafted features
    hand_feats = extract_handcrafted_features(clean_text).reshape(1, -1)

    # Teacher embeddings
    emb_mbert = get_teacher_embedding(
        clean_text,
        teachers["mbert_tokenizer"],
        teachers["mbert_model"]
    )

    emb_xlmr = get_teacher_embedding(
        clean_text,
        teachers["xlmr_tokenizer"],
        teachers["xlmr_model"]
    )

    emb_muril = get_teacher_embedding(
        clean_text,
        teachers["muril_tokenizer"],
        teachers["muril_model"]
    )

    teacher_emb = np.hstack([emb_mbert, emb_xlmr, emb_muril])

    # Combine features
    X = np.hstack([teacher_emb, hand_feats, tfidf_vec])

    # Scale
    X = scaler.transform(X)

    p_cb = mtkd_model.predict_proba(X)[0][1]

    return float(p_cb)


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

    # class mapping
    p_neutral = probs[0]
    p_aggression = probs[1]
    p_distress = probs[2]

    # emotion signal used for fusion
    p_emotion = float(p_aggression + p_distress)
    print(f"Emotion probabilities - Neutral: {p_neutral:.4f}, Aggression: {p_aggression:.4f}, Distress: {p_distress:.4f}")
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

        p_cb = predict_mtkd(text, models)

        p_sarcasm = predict_sarcasm(text, sarcasm_model)

        p_emotion = predict_emotion(text, emotion_tokenizer, emotion_model)

        results.append({
            "text": text,
            "p_cb": p_cb,
            "p_sarcasm": p_sarcasm,
            "p_emotion": p_emotion
        })

    return pd.DataFrame(results)