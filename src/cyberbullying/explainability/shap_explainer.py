import shap
import numpy as np

from .load_explainer import load_artifacts
from .feature_builder import build_features

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

artifacts = load_artifacts()
model = artifacts["model"]

explainer = shap.TreeExplainer(model)

#3.4 — Generate SHAP Values
def get_shap_values(text):
    X = build_features(text, artifacts)
    shap_values = explainer.shap_values(X)

    return X, shap_values[0]

#3.5 — Feature Name Mapping
def get_feature_names():
    word_features = list(artifacts["word_vectorizer"].get_feature_names_out())
    char_features = list(artifacts["char_vectorizer"].get_feature_names_out())

    numeric_features = [
        "sentiment",
        "code_mix",
        "upper_ratio",
        "punctuation_ratio",
        "digit_ratio",
        "word_count",
        "char_length",
        "keyword_present",
        "keyword_count",
        "keyword_ratio"
    ]

    return word_features + char_features + numeric_features

#3.6 — Extract Top Features
def extract_top_features(shap_values, feature_names, top_k=10):

    indices = np.argsort(np.abs(shap_values))[::-1][:top_k]

    top_features = []
    for i in indices:
        top_features.append({
            "feature": feature_names[i],
            "impact": float(shap_values[i])
        })

    return top_features

#3.7 — CLEAN OUTPUT (FINAL)
def explain_text(text):

    X, shap_vals = get_shap_values(text)
    feature_names = get_feature_names()

    indices = np.argsort(np.abs(shap_vals))[::-1][:20]

    word_features = set(artifacts["word_vectorizer"].get_feature_names_out())

    trigger_words = []
    counter_words = []
    signals = {}

    numeric_features = {
        "sentiment",
        "code_mix",
        "upper_ratio",
        "punctuation_ratio",
        "digit_ratio",
        "word_count",
        "char_length",
        "keyword_present",
        "keyword_count",
        "keyword_ratio"
    }

    X_dense = X.toarray()[0]

    # -------------------------------
    # 🔹 SHAP LOOP
    # -------------------------------
    for i in indices:
        name = feature_names[i]
        impact = float(shap_vals[i])
        value = X_dense[i]

        if value == 0:
            continue

        if name in word_features:

            if name in ENGLISH_STOP_WORDS:
                continue

            if impact > 0:
                trigger_words.append({
                    "word": name,
                    "impact": round(impact, 3),
                    "source": "tfidf"
                })
            else:
                counter_words.append({
                    "word": name,
                    "impact": round(abs(impact), 3),
                    "source": "tfidf"
                })

        elif name in numeric_features:
            signals[name] = round(impact, 3)

    # -------------------------------
    # 🔥 KEYWORD FALLBACK (OUTSIDE LOOP)
    # -------------------------------
    if not trigger_words:
        for kw in artifacts["keywords"]:
            if kw in text.lower():
                trigger_words.append({
                    "word": kw,
                    "impact": None,
                    "source": "keyword",
                    "note": "matched from keyword list"
                })
                break

    # -------------------------------
    # 🔥 FINAL FALLBACK (RAW WORD)
    # -------------------------------
    if not trigger_words:
        words = text.lower().split()
        for w in words:
            if w not in ENGLISH_STOP_WORDS:
                trigger_words.append({
                    "word": w,
                    "impact": None,
                    "source": "Out Of Vocabulary",
                    "note": "not in tfidf vocabulary"
                })
                break

    return {
        "trigger_words": trigger_words[:5],
        "counter_words": counter_words[:5],
        "supporting_signals": signals
    }