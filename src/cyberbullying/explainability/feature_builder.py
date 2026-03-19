import re
import numpy as np
import unicodedata
from scipy.sparse import hstack
from textblob import TextBlob

# -------------------------------
# TEXT NORMALIZATION
# -------------------------------

def normalize_text(text):
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text


# -------------------------------
# HANDCRAFTED FEATURES
# -------------------------------

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


# -------------------------------
# KEYWORD FEATURES
# -------------------------------

def build_keyword_patterns(keywords):
    patterns = []
    for kw in keywords:
        pattern = r"\b" + re.escape(kw) + r"\b"
        patterns.append(re.compile(pattern, re.IGNORECASE))
    return patterns


def keyword_features(text, patterns):
    count = 0
    for p in patterns:
        if p.search(text):
            count += 1

    present = 1 if count > 0 else 0
    ratio = count / max(1, len(text.split()))

    return np.array([present, count, ratio])


# -------------------------------
# MAIN FEATURE PIPELINE
# -------------------------------

def build_features(text, artifacts):

    text = normalize_text(text)

    word_vec = artifacts["word_vectorizer"]
    char_vec = artifacts["char_vectorizer"]
    scaler = artifacts["scaler"]
    keywords = artifacts["keywords"]

    # TF-IDF
    X_word = word_vec.transform([text])
    X_char = char_vec.transform([text])

    # handcrafted
    X_hand = extract_handcrafted_features(text).reshape(1, -1)

    # keyword
    patterns = build_keyword_patterns(keywords)
    X_key = keyword_features(text, patterns).reshape(1, -1)

    # numeric combine + scale
    X_numeric = np.hstack([X_hand, X_key])
    X_numeric = scaler.transform(X_numeric)

    # final combine
    X = hstack([X_word, X_char, X_numeric])

    return X