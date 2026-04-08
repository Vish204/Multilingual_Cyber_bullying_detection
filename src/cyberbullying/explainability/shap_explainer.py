import shap
import numpy as np

from .load_explainer import load_artifacts
from .feature_builder import build_features
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

EXTRA_STOP_WORDS = {"thi", "ha", "hey", "the", "a", "an", "is", "are", "to", "of", "and"}

artifacts = load_artifacts()
model = artifacts["model"]
explainer = shap.TreeExplainer(model)

def get_shap_values(text):
    X = build_features(text, artifacts)
    shap_values = explainer.shap_values(X)
    return X, shap_values[0]

def get_feature_names():
    word_features = list(artifacts["word_vectorizer"].get_feature_names_out())
    char_features = list(artifacts["char_vectorizer"].get_feature_names_out())
    numeric_features = [
        "sentiment", "code_mix", "upper_ratio", "punctuation_ratio",
        "digit_ratio", "word_count", "char_length", "keyword_present",
        "keyword_count", "keyword_ratio"
    ]
    return word_features + char_features + numeric_features

# -------------------------------
# 🔹 THE ULTIMATE DYNAMIC SUMMARY
# -------------------------------
def generate_summary(triggers, sigs, prediction_data=None):
    reasons = []
    
    # 1. Pull in Emotion & Sarcasm if available
    if prediction_data:
        emotions = prediction_data.get("emotions", [])
        for emo in emotions:
            if emo["label"] == "aggression" and emo["score"] > 60:
                reasons.append(f"high aggression ({emo['score']}%)")
        
        if prediction_data.get("sarcasm", 0) > 60:
            reasons.append("strong sarcastic undertones")

    # 2. Pull in SHAP mathematical signals
    if sigs.get("keyword_present", 0) > 0 or sigs.get("keyword_ratio", 0) > 0:
        reasons.append("toxic keywords")
    if sigs.get("upper_ratio", 0) > 0:
        reasons.append("aggressive formatting (all caps)")
    if sigs.get("sentiment", 0) < -0.2:
        reasons.append("highly negative sentiment")
        
    # 3. Build the beautiful sentence
    if reasons:
        # Formats list into "A, B, and C"
        reason_str = ", ".join(reasons[:-1]) + (" and " + reasons[-1] if len(reasons) > 1 else reasons[0])
        return f"Flagged due to {reason_str}."
    elif triggers:
        words = [t["word"] for t in triggers[:2]]
        return f"Flagged due to targeted vocabulary ({', '.join(words)})."
    else:
        return "Flagged by the baseline AI for toxic context."

# -------------------------------
# 🔹 CLEAN OUTPUT (FINAL)
# -------------------------------
def explain_text(text, prediction_data=None): # 🔥 Added prediction_data parameter
    X, shap_vals = get_shap_values(text)
    feature_names = get_feature_names()
    indices = np.argsort(np.abs(shap_vals))[::-1]
    word_features = set(artifacts["word_vectorizer"].get_feature_names_out())

    trigger_words = []
    counter_words = []
    signals = {}

    numeric_features = {
        "sentiment", "code_mix", "upper_ratio", "punctuation_ratio",
        "digit_ratio", "word_count", "char_length", "keyword_present",
        "keyword_count", "keyword_ratio"
    }

    X_dense = X.toarray()[0]

    # 🔹 SHAP LOOP (Stopwords & Limiter)
    for i in indices:
        if len(trigger_words) >= 4 and len(counter_words) >= 4:
            break

        name = feature_names[i]
        impact = float(shap_vals[i])
        value = X_dense[i]

        if value == 0:
            continue

        if name in word_features:
            if name in ENGLISH_STOP_WORDS or name in EXTRA_STOP_WORDS or len(name) <= 2:
                continue

            if impact > 0 and len(trigger_words) < 4:
                trigger_words.append({"word": name, "impact": round(impact, 3), "source": "tfidf"})
            elif impact < 0 and len(counter_words) < 4:
                counter_words.append({"word": name, "impact": round(abs(impact), 3), "source": "tfidf"})
        elif name in numeric_features:
            signals[name] = round(impact, 3)

    # 🔥 SMART OOV FALLBACK (Finds the longest slang word)
    if not trigger_words:
        words = text.lower().split()
        valid_words = [w for w in words if w not in ENGLISH_STOP_WORDS and w not in EXTRA_STOP_WORDS and len(w) > 3]
        if valid_words:
            valid_words.sort(key=len, reverse=True)
            trigger_words.append({"word": valid_words[0], "impact": 0.1, "source": "Out Of Vocabulary", "note": "extracted via heuristic"})

    summary_text = generate_summary(trigger_words, signals, prediction_data)

    return {
        "summary": summary_text,
        "trigger_words": trigger_words,
        "counter_words": counter_words,
        "supporting_context": signals
    }