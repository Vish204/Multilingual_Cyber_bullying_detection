from cyberbullying.phase3_inference.load_models import load_all_models
from cyberbullying.phase3_inference.predict_components import run_component_predictions
from cyberbullying.phase3_inference.fusion_inference import compute_fusion_score

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


# def detect_language(text: str):
#     try:
#         if len(text.split()) < 3:
#             return "unknown"
#         lang_code = detect(text)
#         return {
#             "code": lang_code,
#             "name": LANGUAGE_MAP.get(lang_code, "unknown")
#         }
#     except:
#         return "unknown"

def detect_language(text: str):

    try:
        # 🔹 1. Handle empty
        if not text or len(text.strip()) == 0:
            return {"code": "unknown", "name": "unknown"}

        # 🔹 2. SHORT TEXT FIX (VERY IMPORTANT)
        if len(text.split()) <= 3:
            return {"code": "en", "name": "english"}

        # 🔹 3. ASCII CHECK (very effective)
        if text.isascii():
            return {"code": "en", "name": "english"}

        # 🔹 4. Normal detection
        from langdetect import detect
        code = detect(text)

        return {
            "code": code,
            "name": LANGUAGE_MAP.get(code, "unknown")
        }

    except Exception:
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
    fusion_score = compute_fusion_score(
        p_cb,
        p_sarcasm,
        (p_aggression + p_distress)
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
    # Alert Flag (NEW)
    # ---------------------------
    alert = True if severity == "severe" else False

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
            "cyberbullying": round(p_cb, 4),
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