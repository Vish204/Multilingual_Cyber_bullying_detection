from load_models import load_all_models
from predict_components import run_component_predictions
from fusion_inference import compute_fusion_score

print("Loading models...")
models = load_all_models()

keyword_patterns = models["keyword_patterns"]

print("System ready.\n")


# -----------------------------------------
# Emotion helper
# -----------------------------------------
def get_top_emotions(p_neutral, p_aggression, p_distress):
    emotions = {
        "NEUTRAL": p_neutral,
        "AGGRESSION": p_aggression,
        "DISTRESS": p_distress
    }
    sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
    return sorted_emotions[:2]


# -----------------------------------------
# Keyword detection
# -----------------------------------------
def detect_keywords(text, keyword_patterns):

    text = text.lower()
    found = []

    for pattern in keyword_patterns:
        if pattern.search(text):
            found.append(pattern.pattern)

    return list(set(found))


while True:

    text = input("Enter text (type 'exit' to stop): ").strip()

    if text.lower() == "exit":
        break

    if len(text) == 0:
        print("Please enter some text.\n")
        continue

    df = run_component_predictions([text], models)

    p_cb = df["p_cb"].iloc[0]
    p_sarcasm = df["p_sarcasm"].iloc[0]

    # NEW emotion split
    p_neutral = df["p_neutral"].iloc[0]
    p_aggression = df["p_aggression"].iloc[0]
    p_distress = df["p_distress"].iloc[0]

    # ------------------------------
    # KEYWORD BOOST (additive)
    # ------------------------------
    found_keywords = detect_keywords(text, keyword_patterns)

    keyword_boost = 0
    boosted = False

    if len(found_keywords) == 1:
        keyword_boost = 0.08
        boosted = True
    elif len(found_keywords) >= 2:
        keyword_boost = 0.15
        boosted = True

    # ------------------------------
    # BASE FUSION
    # ------------------------------
    fusion_score = compute_fusion_score(
        p_cb,
        p_sarcasm,
        (p_aggression + p_distress)
    )

    fusion_score += keyword_boost

    # ------------------------------
    # EMOTION LOGIC (NEW)
    # ------------------------------

    # Aggression → increase
    if p_aggression > 0.6:
        fusion_score += 0.15
    elif p_aggression > 0.4:
        fusion_score += 0.08

    # Distress → decrease (only if CB is low)
    if p_distress > 0.6 and p_cb < 0.4:
        fusion_score -= 0.15
    elif p_distress > 0.4 and p_cb < 0.3:
        fusion_score -= 0.08

    # Clamp
    fusion_score = max(0.0, min(1.0, fusion_score))

    prediction = "CYBERBULLYING" if fusion_score >= 0.5 else "NORMAL"

    if fusion_score >= 0.8:
        severity = "SEVERE"
    elif fusion_score >= 0.65:
        severity = "MODERATE"
    elif fusion_score >= 0.5:
        severity = "MILD"
    else:
        severity = "NONE"

    # top emotions
    top_emotions = get_top_emotions(p_neutral, p_aggression, p_distress)

    print("\n----- RESULT -----")

    print("Cyberbullying Probability :", round(p_cb, 4))
    print("Sarcasm Probability       :", round(p_sarcasm, 4))

    print("Emotion Probabilities     :")
    print(f"  Neutral: {p_neutral:.4f}, Aggression: {p_aggression:.4f}, Distress: {p_distress:.4f}")

    print("Top Emotions              :", ", ".join([f"{e[0]} ({e[1]:.2f})" for e in top_emotions]))

    print("Fusion Score              :", round(fusion_score, 4))

    if boosted:
        print(f"Keyword Boost Applied     : YES ({len(found_keywords)} matches)")
    else:
        print("Keyword Boost Applied     : NO")

    print("Final Prediction          :", prediction)
    print("Severity Level            :", severity)

    print("-------------------\n")