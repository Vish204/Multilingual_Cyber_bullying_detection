import shap
import numpy as np

from .load_explainer import load_artifacts
from .feature_builder import build_features
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from functools import lru_cache

# 🔥 1. THE MULTILINGUAL STOPWORD NUKE (Expanded)
EXTRA_STOP_WORDS = {
    "thi", "ha", "hey", "the", "a", "an", "is", "are", "to", "of", "and",
    "hai", "kisi", "nahi", "kya", "bhai", "tu", "ka", "ki", "se", "mein", 
    "ko", "pe", "ye", "woh", "ek", "com", "come", "www", "http", "https"
}

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
    # Priority 1: High Aggression
    if prediction_data and prediction_data.get("components", {}).get("aggression", 0) > 75:
        return "Flagged due to highly aggressive tone and hostile context."
    
    # Priority 2: Keyword Hits / Sarcasm
    if sigs.get("keyword_present", 0) > 0 or sigs.get("keyword_ratio", 0) > 0:
        return "Flagged due to presence of targeted toxic keywords."
    if prediction_data and prediction_data.get("sarcasm", 0) > 60:
        return "Flagged due to high probability of toxic sarcasm."
        
    # Priority 3: Trigger Words
    if triggers:
        words = [t["word"] for t in triggers]
        return f"Flagged due to targeted vocabulary ({', '.join(words)})."
        
    # Default Fallback (Since main.py guarantees this is a toxic post)
    return "Flagged due to aggressive tone and contextual signals."

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
            if name in ENGLISH_STOP_WORDS or name in EXTRA_STOP_WORDS or len(name) <= 2:
                continue

            # 🔥 3. STRICT THRESHOLD (> 0.015) & DEDUPLICATION
            if impact > 0.015 and len(trigger_words) < 3:
                # Make sure we don't add the same word twice
                if not any(tw["word"] == name for tw in trigger_words):
                    trigger_words.append({"word": name, "impact": round(impact, 3), "source": "tfidf"})
        
        elif name in numeric_features:
            signals[name] = round(impact, 3)

    # 🔥 4. SMART OOV FALLBACK (Two-Stage)
    if not trigger_words:
        # Since main.py guarantees this only runs for cyberbullying posts, we don't need label checks.
        words = text.lower().split()
        
        # Filter out standard grammar and our extra stopwords
        valid_words = [w for w in words if w not in EXTRA_STOP_WORDS and w not in ENGLISH_STOP_WORDS]
        
        if valid_words:
            # Stage 1: Hunt for obfuscated slang (symbols/numbers)
            obfuscated = [w for w in valid_words if any(char.isdigit() or char in "!@#$%*" for char in w)]
            
            if obfuscated:
                obfuscated.sort(key=len, reverse=True)
                trigger_words.append({
                    "word": obfuscated[0], 
                    "impact": 0.1, 
                    "source": "Out Of Vocabulary", 
                    "note": "obfuscation detected"
                })
            else:
                # Stage 2: Clean Slur Hunt (Grab the longest unusual valid word)
                valid_words.sort(key=len, reverse=True)
                trigger_words.append({
                    "word": valid_words[0], 
                    "impact": 0.1, 
                    "source": "Out Of Vocabulary", 
                    "note": "extracted via heuristic"
                })

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