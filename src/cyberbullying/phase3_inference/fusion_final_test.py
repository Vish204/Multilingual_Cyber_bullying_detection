from load_models import load_all_models
from predict_components import run_component_predictions
from fusion_inference import compute_fusion_score


print("Loading models...")

models = load_all_models()

print("System ready.\n")


while True:

    text = input("Enter text (type 'exit' to stop): ")

    if text.lower() == "exit":
        break

    df = run_component_predictions([text], models)

    p_cb = df["p_cb"].iloc[0]
    p_sarcasm = df["p_sarcasm"].iloc[0]
    p_emotion = df["p_emotion"].iloc[0]

    fusion_score = compute_fusion_score(
        p_cb,
        p_sarcasm,
        p_emotion
    )

    prediction = "CYBERBULLYING" if fusion_score >= 0.5 else "NORMAL"

    if fusion_score >= 0.8:
        severity = "SEVERE"
    elif fusion_score >= 0.65:
        severity = "MODERATE"
    elif fusion_score >= 0.5:
        severity = "MILD"
    else:
        severity = "NONE"

    print("\n----- RESULT -----")

    print("Cyberbullying Probability :", round(p_cb, 4))
    print("Sarcasm Probability       :", round(p_sarcasm, 4))
    print("Emotion Probability       :", round(p_emotion, 4))
    print("Fusion Score              :", round(fusion_score, 4))

    print("Final Prediction          :", prediction)
    print("Severity Level            :", severity)

    print("-------------------\n")