import shap
import numpy as np
import re
import json
import os

from .load_explainer import load_artifacts
from .feature_builder import build_features
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from functools import lru_cache

# # 🔥 1. THE MULTILINGUAL STOPWORD NUKE (Expanded)
# EXTRA_STOP_WORDS = {
#     "thi", "ha", "hey", "the", "a", "an", "is", "are", "to", "of", "and",
#     "hai", "kisi", "nahi", "kya", "bhai", "tu", "ka", "ki", "se", "mein", 
#     "ko", "pe", "ye", "woh", "ek", "com", "come", "www", "http", "https"
# }

# 🔥 Get the exact directory where shap_explainer.py lives
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STOPWORDS_PATH = os.path.join(BASE_DIR, "stopwords.json")

# 🔥 Load the 14-language JSON once when the server boots for O(1) speed
ALL_STOPWORDS = set()
try:
    with open(STOPWORDS_PATH, "r", encoding="utf-8") as f:
        raw_stopwords = json.load(f)
        for lang, words in raw_stopwords.items():
            ALL_STOPWORDS.update([w.lower() for w in words])
except FileNotFoundError:
    print("⚠️ Warning: stopwords.json not found at {STOPWORDS_PATH}. SHAP Fallback might be noisy.")
    

artifacts = load_artifacts()
model = artifacts["model"]
explainer = shap.TreeExplainer(model)

# 🔥 2. CACHE FOR 35ms HOT RUNS
@lru_cache(maxsize=500)
def get_shap_values(text):
    X = build_features(text, artifacts)
    shap_values = explainer.shap_values(X, approximate=True, check_additivity=False)
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
# 🔹 THE PRIORITY DYNAMIC SUMMARY
# -------------------------------
def generate_summary(triggers, sigs, prediction_data=None):
    
    # Priority 1: Trigger Words
    if triggers:
        words = [t["word"] for t in triggers]
        return f"Flagged due to targeted vocabulary ({', '.join(words)})."
    
    # Priority 2: High Aggression
    if prediction_data and prediction_data.get("components", {}).get("aggression", 0) > 75:
        return "Flagged due to highly aggressive tone and hostile context."
    
    # Priority 3: Keyword Hits / Sarcasm
    if sigs.get("keyword_present", 0) > 0 or sigs.get("keyword_ratio", 0) > 0:
        return "Flagged due to presence of targeted toxic keywords."
    if prediction_data and prediction_data.get("sarcasm", 0) > 60:
        return "Flagged due to high probability of toxic sarcasm."
        
    
        
    # Default Fallback (Since main.py guarantees this is a toxic post)
    return "Toxic context detected via semantic phrasing."

# -------------------------------
# 🔹 CLEAN OUTPUT (FINAL)
# -------------------------------
def explain_text(text, prediction_data=None):
    X, shap_vals = get_shap_values(text)
    feature_names = get_feature_names()
    indices = np.argsort(np.abs(shap_vals))[::-1]
    word_features = set(artifacts["word_vectorizer"].get_feature_names_out())

    trigger_words = []
    signals = {}

    numeric_features = {
        "sentiment", "code_mix", "upper_ratio", "punctuation_ratio",
        "digit_ratio", "word_count", "char_length", "keyword_present",
        "keyword_count", "keyword_ratio"
    }

    X_dense = X.toarray()[0]

    # 🔹 SHAP LOOP (Strict Thresholds & Limiters)
    for i in indices:
        if len(trigger_words) >= 3: # Keep UI clean with top 3
            break

        name = feature_names[i]
        impact = float(shap_vals[i])
        value = X_dense[i]

        if value == 0:
            continue

        if name in word_features:
            if name in ENGLISH_STOP_WORDS or name in ALL_STOPWORDS or len(name) <= 2:
                continue

            # 🔥 3. STRICT THRESHOLD (> 0.015) & DEDUPLICATION
            if abs(impact) > 0.0008 and len(trigger_words) < 3:
                # Make sure we don't add the same word twice
                if not any(tw["word"] == name for tw in trigger_words):
                    trigger_words.append({"word": name, "impact": round(impact, 3), "source": "tfidf"})
        
        elif name in numeric_features:
            signals[name] = round(impact, 3)

    # 🔥 4. SMART OOV FALLBACK (Emergency Obfuscation Hunt)
    # Only runs if the standard TF-IDF matrix failed to find the toxic words
    if not trigger_words:
        words = text.lower().split()
        
        # Filter out standard grammar and our extra stopwords
        valid_words = [w for w in words if w not in ALL_STOPWORDS and w not in ENGLISH_STOP_WORDS]
        
        if valid_words:
            # Stage 1: Hunt for obfuscated slang (symbols/numbers)
            obfuscated = []
            for w in valid_words:
                if any(char.isdigit() or char in "!@#$%*" for char in w):
                    # Skip pure numbers ("2026") and ordinals ("1st", "21st")
                    if w.isdigit():
                        continue
                    if re.match(r'^\d+(st|nd|rd|th)$', w.lower()):
                        continue
                    
                    obfuscated.append(w)
            
            if obfuscated:
                obfuscated.sort(key=len, reverse=True)
                # 🔥 HIGHLIGHT UP TO 2 OBFUSCATED WORDS
                for word in obfuscated[:2]: 
                    trigger_words.append({
                        "word": word, 
                        "impact": 0.1, 
                        "source": "Out Of Vocabulary", 
                        "note": "obfuscation detected"
                    })
            
            # 🔥 STAGE 2 HAS BEEN DELETED! 
            # We no longer guess the "longest word". If it's toxic but there are no obvious words or symbols, trigger_words just stays empty and the UI 
            # gracefully displays the Priority Summary instead!

    summary_text = generate_summary(trigger_words, signals, prediction_data)

    # 🔥 RETURN DICTIONARY
    # We return empty arrays/dicts for counter_words and supporting_context 
    # so that the React Frontend doesn't crash if it tries to map over them,
    # but we save server CPU by not actually calculating them!
    return {
        "summary": summary_text,
        "trigger_words": trigger_words,
        "counter_words": [],        
        "supporting_context": {}    
    }