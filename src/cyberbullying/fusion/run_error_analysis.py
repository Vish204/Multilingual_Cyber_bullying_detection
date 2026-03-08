import pandas as pd
import json
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------

BASE_DIR = Path(__file__).resolve().parents[3]

fusion_path = BASE_DIR / "notebooks/analysis_results/fusion/fusion_predictions.csv"
test_path = BASE_DIR / "data/processed/test_data.csv"

output_dir = BASE_DIR / "notebooks/analysis_results/fusion/error_analysis"
output_dir.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Load Data
# -----------------------------

fusion_df = pd.read_csv(fusion_path)
test_df = pd.read_csv(test_path)

# Combine predictions with ground truth
df = pd.concat([test_df, fusion_df.drop(columns=["text"])], axis=1)

# -----------------------------
# Compute Error Type
# -----------------------------

def get_error_type(row):
    if row["label"] == 1 and row["prediction"] == 1:
        return "TP"
    elif row["label"] == 0 and row["prediction"] == 0:
        return "TN"
    elif row["label"] == 0 and row["prediction"] == 1:
        return "FP"
    elif row["label"] == 1 and row["prediction"] == 0:
        return "FN"

df["error_type"] = df.apply(get_error_type, axis=1)

# -----------------------------
# Save Full Error Analysis
# -----------------------------

error_csv = output_dir / "error_analysis.csv"
df.to_csv(error_csv, index=False)

print("Saved:", error_csv)

# -----------------------------
# Extract Examples
# -----------------------------

fp = df[df["error_type"] == "FP"]
fn = df[df["error_type"] == "FN"]
tp = df[df["error_type"] == "TP"]
tn = df[df["error_type"] == "TN"]

fp.to_csv(output_dir / "false_positives.csv", index=False)
fn.to_csv(output_dir / "false_negatives.csv", index=False)
tp.to_csv(output_dir / "true_positives.csv", index=False)
tn.to_csv(output_dir / "true_negatives.csv", index=False)

# -----------------------------
# Summary Statistics
# -----------------------------

summary = {
    "total_samples": len(df),
    "TP": int(len(tp)),
    "TN": int(len(tn)),
    "FP": int(len(fp)),
    "FN": int(len(fn)),
}

summary["accuracy"] = (summary["TP"] + summary["TN"]) / summary["total_samples"]

summary["false_positive_rate"] = summary["FP"] / (summary["FP"] + summary["TN"])
summary["false_negative_rate"] = summary["FN"] / (summary["FN"] + summary["TP"])

with open(output_dir / "error_summary.json", "w") as f:
    json.dump(summary, f, indent=4)

print("Saved summary:", output_dir / "error_summary.json")

# -----------------------------
# Language Error Distribution
# -----------------------------

lang_errors = df.groupby(["language", "error_type"]).size().reset_index(name="count")
lang_errors.to_csv(output_dir / "language_error_distribution.csv", index=False)

# -----------------------------
# Platform Error Distribution
# -----------------------------

platform_errors = df.groupby(["platform", "error_type"]).size().reset_index(name="count")
platform_errors.to_csv(output_dir / "platform_error_distribution.csv", index=False)

print("Error analysis complete.")