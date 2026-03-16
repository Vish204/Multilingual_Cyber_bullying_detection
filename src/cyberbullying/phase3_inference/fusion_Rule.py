import json
import re
import numpy as np
from pathlib import Path

from load_models import load_all_models
from predict_components import (
    predict_cyberbullying,
    predict_sarcasm,
    predict_emotion
)

# ======================================================
# CONFIG
# ======================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

KEYWORDS_DIR = PROJECT_ROOT / "resources/keywords/multilingual_keywords"

BOOST_VALUE = 0.25


# ======================================================
# LOAD KEYWORDS
# ======================================================

def extract_keywords_recursive(obj, keywords):

    if isinstance(obj, dict):

        for k, v in obj.items():

            if k in ["native", "roman", "english"]:
                if isinstance(v, str):
                    keywords.add(v.lower().strip())

            extract_keywords_recursive(v, keywords)

    elif isinstance(obj, list):

        for item in obj:
            extract_keywords_recursive(item, keywords)


def load_multilingual_keywords():

    print("Loading multilingual keywords...")

    keywords = set()

    for file in KEYWORDS_DIR.glob("*.json"):

        before = len(keywords)

        with open(file, encoding="utf-8") as f:
            data = json.load(f)

        if "keywords" in data:

            for kw in data["keywords"]:
                if isinstance(kw, str):
                    keywords.add(kw.lower().strip())

        extract_keywords_recursive(data, keywords)

        after = len(keywords)

        print(f"{file.name} → {after-before} keywords loaded")

    print("\nTotal unique keywords loaded:", len(keywords))

    return keywords


# ======================================================
# KEYWORD DETECTOR
# ======================================================

def detect_keywords(text, keywords):

    text = text.lower()

    found = []

    for kw in keywords:

        if kw in text:
            found.append(kw)

    return list(set(found))


# ======================================================
# MAIN
# ======================================================

def main():

    print("Loading models...")

    models = load_all_models()

    sarcasm_model = models["sarcasm"]
    emotion_tokenizer = models["emotion_tokenizer"]
    emotion_model = models["emotion_model"]

    print("All models loaded successfully!")

    keywords = load_multilingual_keywords()

    print("\nSystem ready.\n")

    while True:

        text = input("Enter text (type 'exit' to stop): ")

        if text.lower() == "exit":
            break

        # ==============================
        # PREDICT COMPONENTS
        # ==============================

        p_cb = predict_cyberbullying(text, models)

        p_sar = predict_sarcasm(text, sarcasm_model)

        p_emo = predict_emotion(text, emotion_tokenizer, emotion_model)

        # ==============================
        # KEYWORD BOOST
        # ==============================

        found_keywords = detect_keywords(text, keywords)

        boosted = False

        if len(found_keywords) > 0:

            p_cb = min(1.0, p_cb + BOOST_VALUE)

            boosted = True

        # ==============================
        # FUSION
        # ==============================

        fusion_score = (
            0.6 * p_cb +
            0.25 * p_sar +
            0.15 * p_emo
        )

        # ==============================
        # FINAL DECISION
        # ==============================

        if fusion_score >= 0.6:
            prediction = "CYBERBULLYING"
            severity = "HIGH"

        elif fusion_score >= 0.4:
            prediction = "CYBERBULLYING"
            severity = "MEDIUM"

        elif fusion_score >= 0.25:
            prediction = "CYBERBULLYING"
            severity = "LOW"

        else:
            prediction = "NORMAL"
            severity = "NONE"

        # ==============================
        # OUTPUT
        # ==============================

        print("\n----- RESULT -----")

        print(f"Cyberbullying Probability : {p_cb:.4f}")
        print(f"Sarcasm Probability       : {p_sar:.4f}")
        print(f"Emotion Probability       : {p_emo:.4f}")
        print(f"Fusion Score              : {fusion_score:.4f}")

        if boosted:
            print(f"Keyword Boost Applied     : YES ({found_keywords})")
        else:
            print("Keyword Boost Applied     : NO")

        print(f"Final Prediction          : {prediction}")
        print(f"Severity Level            : {severity}")

        print("-------------------\n")


if __name__ == "__main__":
    main()