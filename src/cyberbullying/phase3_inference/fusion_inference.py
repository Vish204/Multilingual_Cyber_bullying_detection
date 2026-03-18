import pandas as pd
#from predict_components import run_component_predictions #works only in src/cyberbullying/phase3_inference
from cyberbullying.phase3_inference.predict_components import run_component_predictions #works for phase4 testing

# ----------------------------------------------------
# Fusion Logic
# ----------------------------------------------------

def compute_fusion_score(p_cb, p_sarcasm, p_emotion):
    """
    Compute final cyberbullying score using fusion.
    """

    fusion_score = (
        0.50 * p_cb +
        0.30 * p_sarcasm +
        0.20 * p_emotion
    )

    #fusion_score = max(0.0, min(1.0, fusion_score))

    # 🔥 Context boost (NO retraining, NO keywords)
    if p_cb < 0.3 and p_emotion > 0.7 and p_sarcasm < 0.4:
        fusion_score += 0.15

    fusion_score = min(1.0, fusion_score)
    return fusion_score


# ----------------------------------------------------
# Apply Fusion on Predictions
# ----------------------------------------------------

def run_fusion(text_list, models, threshold=0.5):

    df = run_component_predictions(text_list, models)

    fusion_scores = []
    labels = []

    for _, row in df.iterrows():

        score = compute_fusion_score(
            row["p_cb"],
            row["p_sarcasm"],
            row["p_emotion"]
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