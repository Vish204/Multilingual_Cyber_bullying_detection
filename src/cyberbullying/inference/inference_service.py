from cyberbullying.phase3_inference.load_models import load_all_models
from cyberbullying.phase3_inference.predict_components import run_component_predictions
from cyberbullying.phase3_inference.fusion_inference import compute_hybrid_fusion_score, KEYWORD_LANG_MAP

import unicodedata
import re
from langdetect import detect

# ------------------------------------------------
# Load models ONCE
# ------------------------------------------------

print("Loading models...")
models = load_all_models()
print("System ready.\n")


# ------------------------------------------------
# Language Detection
# ------------------------------------------------

LANGUAGE_MAP = {
    "en": "english",
    "hi": "hindi",
    "mr": "marathi",
    "bn": "bengali",
    "ta": "tamil",
    "te": "telugu",
    "kn": "kannada",
    "ml": "malayalam",
    "gu": "gujarati",
    "pa": "punjabi",
    "ur": "urdu"
}


def detect_language(text: str):
    try:
        # 1. Handle empty strings
        if not text or len(text.strip()) == 0:
            return {"code": "unknown", "name": "unknown"}

        text_lower = text.lower()
        words = set(re.findall(r"\w+", text_lower))

        # 2. THE EMOJI / SYMBOL SAFETY NET (Fix 5)
        # If there are no actual letters/numbers (e.g., text is just "😂😂😂")
        if len(words) == 0:
            return {"code": "unknown", "name": "unknown"}

        # 3. THE ROMANIZED / HINGLISH FIX
        if text.isascii():
            for w in words:
                # Fix 1: Safer dictionary access
                detected_lang = KEYWORD_LANG_MAP.get(w)
                if detected_lang:
                    if detected_lang.lower() == "english":
                        return {"code": "en", "name": "english"}
                    else:
                        # Fix 2: Clean, DB-friendly strings
                        return {"code": "mix", "name": f"{detected_lang.lower()}_romanized"}
            
            # If no regional words found, default to English
            return {"code": "en", "name": "english"}

        # 4. THE UNICODE CHECK (For Native Scripts)
        # Note: We use 'return' to act as an instant 'break' when a script is found.
        # This prevents the loop from failing if it hits an emoji first.
        for char in text:
            if not char.isascii():
                name = unicodedata.name(char, "").upper()
                
                if "DEVANAGARI" in name:
                    for w in words:
                        detected_lang = KEYWORD_LANG_MAP.get(w)
                        if detected_lang:
                            detected_lang = detected_lang.lower()
                            code = "hi" if "hindi" in detected_lang else "mr"
                            return {"code": code, "name": detected_lang}
                    return {"code": "hi/mr", "name": "hindi_or_marathi"}
                
                elif "BENGALI" in name: return {"code": "bn", "name": "bengali"}
                elif "TAMIL" in name: return {"code": "ta", "name": "tamil"}
                elif "TELUGU" in name: return {"code": "te", "name": "telugu"}
                elif "GUJARATI" in name: return {"code": "gu", "name": "gujarati"}
                elif "ARABIC" in name: return {"code": "ur", "name": "urdu"}

        # 5. Fallback to langdetect if everything else fails
        code = detect(text)
        return {
            "code": code,
            "name": LANGUAGE_MAP.get(code, "unknown")
        }

    except Exception:
        # Failsafe so the API never crashes
        return {"code": "unknown", "name": "unknown"}

# ------------------------------------------------
# Emotion helper
# ------------------------------------------------

def get_top_emotions(p_neutral, p_aggression, p_distress):
    emotions = {
        "NEUTRAL": p_neutral,
        "AGGRESSION": p_aggression,
        "DISTRESS": p_distress
    }
    sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
    return sorted_emotions[:2]


# ------------------------------------------------
# MAIN FUNCTION
# ------------------------------------------------

def predict_post(text: str):

    if not text or len(text.strip()) == 0:
        return {"error": "Empty input"}

    # ---------------------------
    # Model Predictions
    # ---------------------------
    df = run_component_predictions([text], models)

    p_cb = float(df["p_cb"].iloc[0])
    p_sarcasm = float(df["p_sarcasm"].iloc[0])

    p_neutral = float(df["p_neutral"].iloc[0])
    p_aggression = float(df["p_aggression"].iloc[0])
    p_distress = float(df["p_distress"].iloc[0])

    # ---------------------------
    # Fusion Score
    # ---------------------------
    # fusion_score = compute_fusion_score(
    #     p_cb,
    #     p_sarcasm,
    #     (p_aggression + p_distress)
    # )
    fusion_score, calibrated_p_cb = compute_hybrid_fusion_score(
        p_cb,
        p_sarcasm,
        p_aggression,
        p_distress,
        p_neutral,
        text
    )

    # ---------------------------
    # Prediction
    # ---------------------------
    prediction = "cyberbullying" if fusion_score >= 0.5 else "normal"

    # ---------------------------
    # Severity
    # ---------------------------
    if fusion_score >= 0.8:
        severity = "severe"
    elif fusion_score >= 0.65:
        severity = "moderate"
    elif fusion_score >= 0.5:
        severity = "mild"
    else:
        severity = "none"

    # ---------------------------
    # Alert Flag (Trust & Safety Logic)
    # ---------------------------
    # Alerts are reserved for extreme cases to prevent moderator "Alarm Fatigue".
    
    is_extreme_threat = p_distress > 0.65
    is_aggressive_attack = (calibrated_p_cb >= 0.85 and p_aggression > 0.70)
    is_absolute_certainty = fusion_score > 0.90
    
    if is_extreme_threat or is_aggressive_attack or is_absolute_certainty:
        alert = True
    else:
        alert = False

    # ---------------------------
    # Language Detection (NEW)
    # ---------------------------
    language = detect_language(text)

    # ---------------------------
    # Top Emotions
    # ---------------------------
    top_emotions = get_top_emotions(
        p_neutral,
        p_aggression,
        p_distress
    )

    # ---------------------------
    # Final Output (CLEAN + STANDARDIZED)
    # ---------------------------
    
    return {

        "label": prediction,
        "severity": severity,
        "confidence": round(float(fusion_score), 4),

        "emotion": top_emotions[0][0].lower(),   # primary emotion
        "sarcasm": round(p_sarcasm, 4),
        "language": language,
        "alert": alert,

        "components": {
            "cyberbullying": round(calibrated_p_cb, 4),
            "sarcasm": round(p_sarcasm, 4),
            "neutral": round(p_neutral, 4),
            "aggression": round(p_aggression, 4),
            "distress": round(p_distress, 4)
        },

        "emotions": [
            {
                "label": e[0].lower(),
                "score": round(float(e[1]), 3)
            }
            for e in top_emotions
        ]
    }