from cyberbullying.phase3_inference.load_models import load_all_models
from cyberbullying.phase3_inference.predict_components import run_component_predictions
from cyberbullying.phase3_inference.fusion_inference import compute_fusion_score

# ------------------------------------------------
# Load models ONCE
# ------------------------------------------------

print("Loading models...")
models = load_all_models()
print("System ready.\n")


# ------------------------------------------------
# Emotion helper (same as yours)
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
# MAIN FUNCTION (this replaces input loop)
# ------------------------------------------------

def predict_post(text: str):

    if not text or len(text.strip()) == 0:
        return {"error": "Empty input"}

    df = run_component_predictions([text], models)

    p_cb = float(df["p_cb"].iloc[0])
    p_sarcasm = float(df["p_sarcasm"].iloc[0])

    p_neutral = float(df["p_neutral"].iloc[0])
    p_aggression = float(df["p_aggression"].iloc[0])
    p_distress = float(df["p_distress"].iloc[0])

    # SAME fusion logic (unchanged)
    fusion_score = compute_fusion_score(
        p_cb,
        p_sarcasm,
        (p_aggression + p_distress)
    )

    # Prediction
    prediction = "CYBERBULLYING" if fusion_score >= 0.5 else "NORMAL"

    # Severity (same thresholds)
    if fusion_score >= 0.8:
        severity = "SEVERE"
    elif fusion_score >= 0.65:
        severity = "MODERATE"
    elif fusion_score >= 0.5:
        severity = "MILD"
    else:
        severity = "NONE"

    # Top emotions
    top_emotions = get_top_emotions(
        p_neutral,
        p_aggression,
        p_distress
    )

    # RETURN instead of print
    # return {
    #     "text": text,
    #     #"prediction": prediction,
    #     "label": prediction,
    #     "severity": severity,
    #     "fusion_score": float(fusion_score),

    #     "probabilities": {
    #         "cyberbullying": p_cb,
    #         "sarcasm": p_sarcasm,
    #         "neutral": p_neutral,
    #         "aggression": p_aggression,
    #         "distress": p_distress
    #     },

    #     "top_emotions": [
    #         {"label": e[0], "score": float(e[1])}
    #         for e in top_emotions
    #     ]
    # }
    return {
    "label": prediction.lower(),              # "cyberbullying"
    "severity": severity.lower(),             # "severe"
    "confidence": float(fusion_score),        # renamed from fusion_score

    "components": {
        "cyberbullying": p_cb,
        "sarcasm": p_sarcasm
    },

    "emotions": [
        {
            "label": e[0].lower(),            # "aggression"
            "score": round(float(e[1]), 3)    # 0.819
        }
        for e in top_emotions
    ]
}