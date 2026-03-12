import pandas as pd
from pathlib import Path

# ------------------------
# Paths
# ------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

FUSION_DIR = PROJECT_ROOT / "notebooks" /"analysis_results" / "fusion"

MTKD_PATH = FUSION_DIR / "mtkd_v2_probs.csv"
SARCASM_PATH = FUSION_DIR / "sarcasm_probs.csv"
EMOTION_PATH = FUSION_DIR / "emotion_probs.csv"

OUTPUT_PATH = FUSION_DIR / "fusion_predictions.csv"

# ------------------------
# Load probability files
# ------------------------

print("\nLoading probability outputs...")

mtkd = pd.read_csv(MTKD_PATH)
sarcasm = pd.read_csv(SARCASM_PATH)
emotion = pd.read_csv(EMOTION_PATH)

print("MTKD samples:", len(mtkd))
print("Sarcasm samples:", len(sarcasm))
print("Emotion samples:", len(emotion))

# ------------------------
# Combine into single dataframe
# ------------------------

fusion_df = pd.DataFrame({
    "text": mtkd["text"],
    "p_cb": mtkd["P_cb"],
    "p_sarcasm": sarcasm["p_sarcasm"],
    "p_emotion": emotion["p_emotion"]
})

# ------------------------
# Fusion Formula
# ------------------------

print("\nComputing fusion score...")

fusion_df["fusion_score"] = (
      0.60 * fusion_df["p_cb"]
    + 0.25 * fusion_df["p_sarcasm"]
    + 0.15 * fusion_df["p_emotion"]
)

# ------------------------
# Final Prediction
# ------------------------

fusion_df["prediction"] = (fusion_df["fusion_score"] >= 0.5).astype(int)

# ------------------------
# Save Results
# ------------------------

fusion_df.to_csv(OUTPUT_PATH, index=False)

print(f"\nFusion predictions saved to: {OUTPUT_PATH}")