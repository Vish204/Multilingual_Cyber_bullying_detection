import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------

BASE_DIR = Path(__file__).resolve().parents[3]

error_path = BASE_DIR / "notebooks/analysis_results/fusion/error_analysis/error_analysis.csv"
output_dir = BASE_DIR / "notebooks/analysis_results/fusion/error_analysis"

df = pd.read_csv(error_path)

# -----------------------------
# Error Type Distribution
# -----------------------------

error_counts = df["error_type"].value_counts()

plt.figure()
error_counts.plot(kind="bar")
plt.title("Error Type Distribution")
plt.xlabel("Error Type")
plt.ylabel("Count")
plt.tight_layout()

plt.savefig(output_dir / "error_distribution.png")
plt.close()

# -----------------------------
# Fusion Score Distribution
# -----------------------------

plt.figure()
df["fusion_score"].hist(bins=30)
plt.title("Fusion Score Distribution")
plt.xlabel("Fusion Score")
plt.ylabel("Frequency")
plt.tight_layout()

plt.savefig(output_dir / "fusion_score_distribution.png")
plt.close()

# -----------------------------
# Probability Distributions
# -----------------------------

for col in ["p_cb", "p_sarcasm", "p_emotion"]:
    plt.figure()
    df[col].hist(bins=30)
    plt.title(f"{col} Distribution")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()

    plt.savefig(output_dir / f"{col}_distribution.png")
    plt.close()

# -----------------------------
# Language Error Distribution
# -----------------------------

lang_errors = df.groupby("language")["error_type"].count()

plt.figure()
lang_errors.plot(kind="bar")
plt.title("Errors by Language")
plt.xlabel("Language")
plt.ylabel("Count")
plt.tight_layout()

plt.savefig(output_dir / "language_error_distribution.png")
plt.close()

# -----------------------------
# Platform Error Distribution
# -----------------------------

platform_errors = df.groupby("platform")["error_type"].count()

plt.figure()
platform_errors.plot(kind="bar")
plt.title("Errors by Platform")
plt.xlabel("Platform")
plt.ylabel("Count")
plt.tight_layout()

plt.savefig(output_dir / "platform_error_distribution.png")
plt.close()

print("Plots generated successfully.")