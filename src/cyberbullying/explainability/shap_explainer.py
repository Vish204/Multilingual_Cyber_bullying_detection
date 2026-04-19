import shap
import numpy as np
import re
import json
import os
from pathlib import Path

from .load_explainer import load_artifacts
from .feature_builder import build_features
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from functools import lru_cache

# ---------------------------------------------------------
# 🔥 1. ZERO-LATENCY DICTIONARIES (O(1) Lookups)
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STOPWORDS_PATH = os.path.join(BASE_DIR, "stopwords.json")

ALL_STOPWORDS = set()
ALL_KEYWORDS = set()

# Load 14-Language Stopwords
try:
    with open(STOPWORDS_PATH, "r", encoding="utf-8") as f:
        raw_stopwords = json.load(f)
        for lang, words in raw_stopwords.items():
            ALL_STOPWORDS.update([w.lower() for w in words])
except FileNotFoundError:
    print(f"⚠️ Warning: stopwords.json not found at {STOPWORDS_PATH}.")

# Load 14-Language Toxic Keywords (For the Fast Path!)  (Pathlib traversal)
KEYWORDS_DIR = Path(__file__).resolve().parents[3] / "resources" / "keywords" / "multilingual_keywords"

if KEYWORDS_DIR.exists():
    for file in os.listdir(KEYWORDS_DIR):
        if file.startswith("keywords_") and file.endswith(".json"):
            try:
                with open(KEYWORDS_DIR / file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Your JSONs have a "keywords" array!
                    if "keywords" in data:
                        for kw in data["keywords"]:
                            kw_str = str(kw).strip().lower()
                            # We only add single words to the Fast Path to match our regex
                            if len(kw_str) > 2 and " " not in kw_str:
                                ALL_KEYWORDS.add(kw_str)
            except Exception as e:
                print(f"⚠️ Warning: Could not load {file}: {e}")
else:
    print(f"⚠️ CRITICAL: Keywords directory not found at {KEYWORDS_DIR}")

print(f"✅ Loaded {len(ALL_STOPWORDS)} Stopwords and {len(ALL_KEYWORDS)} Toxic Keywords into memory.")


# ---------------------------------------------------------
# 🔹 ML ARTIFACTS & CACHING
# ---------------------------------------------------------
artifacts = load_artifacts()
model = artifacts["model"]
explainer = shap.TreeExplainer(model)

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


# ---------------------------------------------------------
# 🔹 THE PRIORITY DYNAMIC SUMMARY
# ---------------------------------------------------------
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
        # return "Flagged due to presence of targeted toxic keywords."
        return "Flagged due to harmful intent "
    if prediction_data and prediction_data.get("sarcasm", 0) > 60:
        return "Flagged due to high probability of toxic sarcasm."
        
    # Default Fallback
    return "Toxic context detected via semantic phrasing."


# ---------------------------------------------------------
# 🚀 THE 5-STAGE EXPLAINER ENGINE (Strict Short-Circuiting)
# ---------------------------------------------------------
def explain_text(text, prediction_data=None):
    # 🔥 Issue 5 Fix: Empty input guard
    if not text or not text.strip():
        return {
            "summary": "No content to analyze.",
            "trigger_words": [],
            "counter_words": [],
            "supporting_context": {}
        }

    trigger_words = []
    signals = {}
    
    # Tokenize
    raw_words = text.split()
    clean_words = re.findall(r'\b\w+\b', text.lower())

    # =======================================================
    # ⚡ STAGE 1: FAST PATH (Keyword Match)
    # =======================================================
    for w in clean_words:
        if len(trigger_words) >= 3: break
        if w in ALL_KEYWORDS and w not in ENGLISH_STOP_WORDS and w not in ALL_STOPWORDS:
            if not any(tw["word"] == w for tw in trigger_words):
                trigger_words.append({"word": w, "impact": 0.9, "source": "keyword_match"})
    
    # 🔥 Issue 1 Fix: Absolute Early Return
    if trigger_words:
        return {
            "summary": generate_summary(trigger_words, signals, prediction_data),
            "trigger_words": trigger_words,
            "counter_words": [],
            "supporting_context": {}
        }

    # =======================================================
    # ⚡ STAGE 2: OBFUSCATION PATH (e.g. "b!tch")
    # =======================================================
    for w in raw_words:
        if len(trigger_words) >= 3: break
        w_lower = w.lower()
        
        # 🔥 Issue 2 Fix: Must be a mix of letters AND symbols/digits
        if re.search(r'[a-zA-Z].*[\d!@#$%*]|[\d!@#$%*].*[a-zA-Z]', w_lower):
            if w_lower.isdigit() or re.match(r'^\d+(st|nd|rd|th)$', w_lower):
                continue
            if not any(tw["word"] == w_lower for tw in trigger_words):
                trigger_words.append({"word": w_lower, "impact": 0.8, "source": "obfuscation"})

    if trigger_words:
        return {
            "summary": generate_summary(trigger_words, signals, prediction_data),
            "trigger_words": trigger_words,
            "counter_words": [],
            "supporting_context": {}
        }

    # =======================================================
    # ⚡ STAGE 3: SHORT TEXT EXTRACTION (<= 6 words)
    # =======================================================
    # 🔥 Issue 3 Fix: Reduced limit to 6 to prevent noise
    if len(clean_words) <= 6:
        for w in clean_words:
            if len(trigger_words) >= 3: break
            if w not in ALL_STOPWORDS and w not in ENGLISH_STOP_WORDS and len(w) > 2:
                if not any(tw["word"] == w for tw in trigger_words):
                    trigger_words.append({"word": w, "impact": 0.7, "source": "semantic_extraction"})
        
        if trigger_words:
            return {
                "summary": generate_summary(trigger_words, signals, prediction_data),
                "trigger_words": trigger_words,
                "counter_words": [],
                "supporting_context": {}
            }

    # =======================================================
    # 🧠 STAGE 4: DEEP PATH (SHAP Tax)
    # =======================================================
    # 🔥 Issue 4 Fix: SHAP only runs if ALL the above bypassed/failed
    try:
        X, shap_vals = get_shap_values(text)
        feature_names = get_feature_names()
        indices = np.argsort(np.abs(shap_vals))[::-1]
        word_features = set(artifacts["word_vectorizer"].get_feature_names_out())
        numeric_features = {"sentiment", "code_mix", "upper_ratio", "punctuation_ratio", "digit_ratio", "word_count", "char_length", "keyword_present", "keyword_count", "keyword_ratio"}
        X_dense = X.toarray()[0]

        for i in indices:
            if len(trigger_words) >= 3: break

            name = feature_names[i]
            impact = float(shap_vals[i])
            value = X_dense[i]

            if value == 0: continue

            if name in word_features:
                if name in ENGLISH_STOP_WORDS or name in ALL_STOPWORDS or len(name) <= 2:
                    continue
                
                # Using abs() to catch negative log-odds
                if abs(impact) > 0.01:
                    if not any(tw["word"] == name for tw in trigger_words):
                        trigger_words.append({"word": name, "impact": round(abs(impact), 3), "source": "tfidf"})
            
            elif name in numeric_features:
                signals[name] = round(abs(impact), 3)
                
    except Exception as e:
        print(f"SHAP Error: {e}")

    # =======================================================
    # 🏁 STAGE 5: FINAL FALLBACK (Semantic Toxicity)
    # =======================================================
    return {
        "summary": generate_summary(trigger_words, signals, prediction_data),
        "trigger_words": trigger_words,
        "counter_words": [],        
        "supporting_context": {}    
    }