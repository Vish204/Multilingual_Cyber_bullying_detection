import re
import json
import os
from pathlib import Path
import pandas as pd
import time
#from predict_components import run_component_predictions #works only in src/cyberbullying/phase3_inference
from cyberbullying.phase3_inference.predict_components import run_component_predictions #works for phase4 testing


KEYWORDS_DIR = Path(__file__).resolve().parents[3] / "resources/keywords/multilingual_keywords"
print(KEYWORDS_DIR)

SINGLE_WORDS = set()
PHRASES = []

# NEW: Dictionary to map words to their languages
KEYWORD_LANG_MAP = {}

def load_keywords():

    print(" Loading multilingual keywords for fusion...")
    start_time = time.time()

    if KEYWORDS_DIR.exists():
        for file in os.listdir(KEYWORDS_DIR):
            if file.endswith(".json"):
                lang_name = file.replace(".json", "").capitalize()

                try:
                    with open(KEYWORDS_DIR / file, "r", encoding="utf-8") as f:
                        data = json.load(f)

                        if "keywords" in data:
                            for k in data["keywords"]:
                                k = str(k).strip().lower()
                                if len(k) <= 2:
                                    continue
                                if " " in k:
                                    PHRASES.append(k)   # no regex
                                else:
                                    SINGLE_WORDS.add(k)
                                
                                KEYWORD_LANG_MAP[k] = lang_name
                except Exception as e:
                    print(f" Warning: Could not load {file}: {e}")
    else:
        print(f" CRITICAL: Keywords directory not found at {KEYWORDS_DIR}")

    end_time = time.time()
    elapsed_ms = (end_time - start_time) * 1000
    total_loaded = len(SINGLE_WORDS) + len(PHRASES)
    
    print(f" Fusion Keywords Loaded: {total_loaded} total ({len(SINGLE_WORDS)} words, {len(PHRASES)} phrases) in {elapsed_ms:.2f} ms")

#  To check what keywords language its loading 
    # print("\n" + "="*40)
    # print(" KEYWORD BREAKDOWN BY LANGUAGE")
    # print("="*40)
    
    # # 1. Count how many words belong to each language
    # lang_counts = {}
    # for word, lang in KEYWORD_LANG_MAP.items():
    #     lang_counts[lang] = lang_counts.get(lang, 0) + 1
        
    # for lang, count in lang_counts.items():
    #     print(f" - {lang}: {count} keywords")
        
    # print("="*40 + "\n")





    #   View Samples 
    # print("\n" + "="*50)
    # print(" KEYWORD SAMPLES BY TARGET LANGUAGE")
    # print("="*50)
    

#Check keywords and its lang
    # 1. Group words by language
    # lang_words = {}
    # for word, lang in KEYWORD_LANG_MAP.items():
    #     if lang not in lang_words:
    #         lang_words[lang] = []
    #     lang_words[lang].append(word)
        
    # # 2. The specific languages you want to check
    # target_langs = ["Keywords_kannada", "Keywords_telugu", "Keywords_tamil", "Keywords_marathi", "Keywords_oriya"]
    
    # for lang in target_langs:
    #     if lang in lang_words:
    #         # Grab the first 10 words
    #         sample = lang_words[lang][:10] 
    #         count = len(lang_words[lang])
    #         print(f" - {lang} ({count} total):")
    #         print(f"   Samples: {', '.join(sample)}\n")
    #     else:
    #         print(f" - {lang}: Not found in map!\n")
            
    # print("="*50 + "\n")

#  load once
load_keywords()
# ----------------------------------------------------
# Fusion Logic
# ----------------------------------------------------

# def compute_fusion_score(p_cb, p_sarcasm, p_emotion):
#     """
#     Compute final cyberbullying score using fusion.
#     """

#     fusion_score = (
#         0.50 * p_cb +
#         0.30 * p_sarcasm +
#         0.20 * p_emotion
#     )

#     #fusion_score = max(0.0, min(1.0, fusion_score))

#     #  Context boost (NO retraining, NO keywords)
#     if p_cb < 0.3 and p_emotion > 0.7 and p_sarcasm < 0.4:
#         fusion_score += 0.15

#     fusion_score = min(1.0, fusion_score)
#     return fusion_score

def keyword_match(text):
    text_lower = text.lower()

    # fast token match
    words = set(re.findall(r"\w+", text_lower))
    for w in words:
        if w in SINGLE_WORDS:
            return True

    # phrase match (simple substring, no regex)
    for p in PHRASES:
        if p in text_lower:
            return True

    return False


def compute_hybrid_fusion_score(p_cb, p_sarcasm, p_aggression, p_distress, p_neutral, text=None):

   
# 1. TIERED OOV FAILSAFE (Restores Mild, Moderate, Severe)
    # Lowered the aggression threshold from 0.50 to 0.35 because 
    # 0.44 aggression in a 3-class model is actually a very strong signal.
    # if p_cb < 0.25: was giving problem in marathi and some other language
    if p_cb < 0.36:
        if p_aggression > 0.60:
            p_cb = 0.90  # High Aggression -> Calibrate High
        elif p_aggression > 0.45:
            p_cb = 0.85  # Moderate Aggression -> Calibrate Mid
        elif p_aggression > 0.30:
            p_cb = 0.65  # Borderline Aggression -> Calibrate Low (Results in "Mild")
    
# 2. MALICIOUS SARCASM RULE (Fixes "jump off a building")
    # If Sarcasm is huge (>0.70) and the text isn't purely friendly/neutral,
    # force the base model to wake up and acknowledge the passive-aggressive threat.
    if p_sarcasm > 0.70 and p_neutral < 0.55:
        p_cb = 0.85

# PROBLEM OF SHORT WORD LIKE "CHUTIYA"    
# # 3. KEYWORD MATCH (Yes, absolutely keep this!)
#     # This is your ultimate safety net for obvious slurs.
#     if text and keyword_match(text) and p_neutral < 0.60:
#         p_cb = 1.0
    

# 3. EXACT KEYWORD MATCH (The Safety Net)
    # If it's a very short text (<= 3 words), ban the slur immediately (ignores Neutral).
    # If it's a longer sentence, only ban it if Neutral is less than 0.95.
    word_count = len(text.split())
    print(f"DEBUG - Did we find a keyword?: {keyword_match(text)}")

    if text and keyword_match(text):
        if word_count <= 3:
            p_cb = max(p_cb, 0.8)  # Instant catch for single-word slurs like "Chutiya"
        elif p_neutral < 0.95:
            p_cb = 0.85  # Catch for slurs in sentences, unless it's overwhelmingly friendly


# 4. DISTRESS MULTIPLIER
    if p_distress > 0.70 and p_aggression > 0.30:
        p_cb = max(p_cb, 0.85)
    elif p_distress > 0.50:
        p_cb = min(1.0, p_cb + 0.25)



    # Base weighted fusion
    # fusion_score = 0.5 * p_cb + 0.3 * p_sarcasm + 0.2 * (p_aggression + p_distress)

    fusion_score = (0.70 * p_cb) + (0.30 * max(p_aggression, p_distress))

    # Add Sarcasm as a malicious bonus, not a penalty
    if p_sarcasm > 0.60 and p_neutral < 0.60:
        fusion_score += (p_sarcasm * 0.25)

    # # Conditional adjustments
    # if p_sarcasm > 0.70:
    #     fusion_score += 0.25

    # if p_aggression > 0.6:
    #     fusion_score += 0.35

    # if p_distress > 0.6:
    #     fusion_score -= 0.1

    # # Exact keyword match override (if we want)
    # # Example: simple regex check
    # if text and keyword_match(text):
    #     fusion_score += 0.25   # or whatever boost you want

    # Ensure score is between 0 and 1
    fusion_score = min(max(fusion_score, 0.0), 1.0)

    return fusion_score, p_cb

# ----------------------------------------------------
# Apply Fusion on Predictions
# ----------------------------------------------------

def run_fusion(text_list, models, threshold=0.5):

    df = run_component_predictions(text_list, models)

    fusion_scores = []
    labels = []

    for _, row in df.iterrows():

        # score = compute_fusion_score(
        #     row["p_cb"],
        #     row["p_sarcasm"],
        #     row["p_emotion"]
        # )
        score, calibrated_p_cb = compute_hybrid_fusion_score(
            row["p_cb"],
            row["p_sarcasm"],
            row["p_aggression"],
            row["p_distress"],
            row["p_neutral"],
            row["text"]
        )

        fusion_scores.append(score)

        if score >= threshold:
            labels.append("cyberbullying")
        else:
            labels.append("non-cyberbullying")

    df["fusion_score"] = fusion_scores
    df["prediction"] = labels

    return df


# ----------------------------------------------------
# Standalone Test
# ----------------------------------------------------

if __name__ == "__main__":

    from load_models import load_all_models

    models = load_all_models()

    test_texts = [
        "You are such an idiot",
        "Wow amazing job genius",
        "I hate you so much",
        "Great work! Proud of you"
    ]

    results = run_fusion(test_texts, models)

    print("\nFusion Inference Results:\n")
    print(results)