import sys
import json
import re
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.cyberbullying.phase3_inference.load_models import load_all_models
from src.cyberbullying.phase3_inference.predict_components import run_component_predictions


# ---------------------------------------------------
# KEYWORD DIRECTORY
# ---------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[3]
KEYWORD_DIR = BASE_DIR / "resources/keywords/multilingual_keywords"


# ---------------------------------------------------
# EXTRACT WORDS FROM JSON
# ---------------------------------------------------

def extract_words(obj):

    words = []

    if isinstance(obj, list):
        for item in obj:
            words.extend(extract_words(item))

    elif isinstance(obj, dict):
        for v in obj.values():
            words.extend(extract_words(v))

    elif isinstance(obj, str):
        words.append(obj)

    return words


# ---------------------------------------------------
# LOAD KEYWORDS
# ---------------------------------------------------

def load_keywords():

    keywords = set()

    files = list(KEYWORD_DIR.glob("*.json"))

    if not files:
        print("WARNING: No keyword files found")

    for file in files:

        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)

            if "keywords" in data:
                words = data["keywords"]
            else:
                words = []

            clean_words = [
                w.lower().strip()
                for w in words
                if isinstance(w, str) and len(w.strip()) > 1
            ]

            keywords.update(clean_words)

            print(f"{file.name} → {len(clean_words)} keywords loaded")

        except Exception as e:
            print(f"Error loading {file.name}: {e}")

    print(f"\nTotal unique keywords loaded: {len(keywords)}\n")

    return keywords

# ---------------------------------------------------
# TEXT NORMALIZATION
# ---------------------------------------------------

def normalize_text(text):

    text = text.lower()

    text = re.sub(r"[^\w\s]", "", text)

    text = re.sub(r"(.)\1{2,}", r"\1\1", text)

    return text


# ---------------------------------------------------
# KEYWORD DETECTION
# ---------------------------------------------------

def keyword_detected(text, keywords):

    text = normalize_text(text)

    words = set(text.split())

    matches = words.intersection(keywords)

    if matches:
        return True, list(matches)[0]

    return False, None


# ---------------------------------------------------
# FUSION WEIGHTS
# ---------------------------------------------------

W_CB = 0.5
W_SAR = 0.3
W_EMO = 0.2


# ---------------------------------------------------
# SEVERITY LEVEL
# ---------------------------------------------------

def get_severity(score):

    if score >= 0.8:
        return "SEVERE"
    elif score >= 0.6:
        return "MODERATE"
    elif score >= 0.4:
        return "MILD"
    else:
        return "NONE"


# ---------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------

print("Loading models...")

models = load_all_models()

print("Loading multilingual keywords...")

keywords = load_keywords()

print("System ready.\n")


# ---------------------------------------------------
# INTERACTIVE LOOP
# ---------------------------------------------------

while True:

    text = input("Enter text (type 'exit' to stop): ")

    if text.lower() == "exit":
        break

    # ---------------------------------------------------
    # RUN MODEL COMPONENTS
    # ---------------------------------------------------

    results = run_component_predictions(text, models)

    p_cb = float(results["p_cb"].mean())
    p_sar = float(results["p_sarcasm"].mean())
    p_emo = float(results["p_emotion"].mean())

    # ---------------------------------------------------
    # KEYWORD CONTEXT BOOST
    # ---------------------------------------------------

    detected, matched_word = keyword_detected(text, keywords)

    if detected:

        keyword_boost = 0.25

        # sarcasm intensifies toxicity
        if p_sar > 0.5:
            keyword_boost += 0.10

        # aggressive emotion
        if p_emo > 0.4:
            keyword_boost += 0.05

        p_cb = min(p_cb + keyword_boost, 1.0)

        print(f"Keyword detected → '{matched_word}' | Boost applied: {keyword_boost}")

    # ---------------------------------------------------
    # FUSION
    # ---------------------------------------------------

    fusion_score = (
        W_CB * p_cb +
        W_SAR * p_sar +
        W_EMO * p_emo
    )

    # ---------------------------------------------------
    # FINAL DECISION
    # ---------------------------------------------------

    if fusion_score >= 0.5:
        final_prediction = "CYBERBULLYING"
    else:
        final_prediction = "NORMAL"

    severity = get_severity(fusion_score)

    # ---------------------------------------------------
    # OUTPUT
    # ---------------------------------------------------

    print("\n----- RESULT -----")

    print(f"Cyberbullying Probability : {round(p_cb,4)}")
    print(f"Sarcasm Probability       : {round(p_sar,4)}")
    print(f"Emotion Probability       : {round(p_emo,4)}")
    print(f"Fusion Score              : {round(fusion_score,4)}")
    print(f"Final Prediction          : {final_prediction}")
    print(f"Severity Level            : {severity}")

    print("-------------------\n")