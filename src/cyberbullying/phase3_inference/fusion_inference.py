import pandas as pd
from load_models import load_all_models
from predict_components import run_component_predictions


# ----------------------------------------------------
# Fusion Logic
# ----------------------------------------------------

def compute_fusion_score(p_cb, p_sarcasm, p_emotion):
    """
    Compute final cyberbullying score using fusion.

    Formula used in Phase 3.5 evaluation.
    """

    fusion_score = (
        0.50 * p_cb +
        0.3 * p_sarcasm +
        0.2 * p_emotion
    )

    fusion_score = max(0.0, min(1.0, fusion_score))
    
    return fusion_score


# ----------------------------------------------------
# Apply Fusion on Predictions
# ----------------------------------------------------

def run_fusion(text_list, models, threshold=0.5):
    """
    Run full fusion pipeline.

    Returns dataframe with final predictions.
    """

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