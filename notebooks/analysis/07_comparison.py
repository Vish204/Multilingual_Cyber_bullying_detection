import os
import json
import pandas as pd
import matplotlib.pyplot as plt

# ===============================
# 1️⃣ Define Paths
# ===============================

BASE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../analysis_results")
)

MODEL_PATHS = {
    "Baseline_XGBoost": os.path.join(BASE_PATH, "baseline_xgboost", "metrics.json"),
    "MuRIL": os.path.join(BASE_PATH, "muril", "metrics.json"),
    "mBERT": os.path.join(BASE_PATH, "mbert", "metrics.json"),
    "XLM-R": os.path.join(BASE_PATH, "xlmr", "metrics.json"),
    "MTKD_XGBoost": os.path.join(BASE_PATH, "mtkd_xgboost", "metrics.json"),  # ✅ ADDED
}

OUTPUT_DIR = os.path.join(BASE_PATH, "model_comparison")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("BASE_PATH:", BASE_PATH)

# ===============================
# 2️⃣ Load Metrics
# ===============================

results = []

for model_name, path in MODEL_PATHS.items():
    if not os.path.exists(path):
        print(f"⚠ Metrics file not found for {model_name}")
        continue

    with open(path, "r") as f:
        metrics = json.load(f)

    params = metrics.get("parameters")

    # Convert parameters to Millions (paper standard)
    if params:
        params_m = round(params / 1e6, 2)
    else:
        params_m = None

    results.append({
        "Model": model_name,
        "Params(M)": params_m,
        "Size(MB)": float(metrics.get("model_size_mb", 0)),
        "Accuracy": float(metrics.get("accuracy", 0)),
        "Precision": float(metrics.get("precision", 0)),
        "Recall": float(metrics.get("recall", 0)),
        "F1-Score": float(metrics.get("f1_score", 0)),
        "ROC-AUC": float(metrics.get("roc_auc", 0)),
    })

df = pd.DataFrame(results)

if df.empty:
    print("No metrics found. Run model evaluations first.")
    exit()

# ===============================
# 3️⃣ Sort by F1 Score
# ===============================

df = df.sort_values("F1-Score", ascending=False)

# ===============================
# 4️⃣ Save Comparison Table
# ===============================

csv_path = os.path.join(OUTPUT_DIR, "model_comparison.csv")
txt_path = os.path.join(OUTPUT_DIR, "model_comparison.txt")

df.to_csv(csv_path, index=False)

with open(txt_path, "w") as f:
    f.write(df.to_string(index=False))

print("\nModel Comparison:\n")
print(df)

# ===============================
# 5️⃣ Performance Plot
# ===============================

metrics_to_plot = ["Accuracy", "F1-Score", "ROC-AUC"]

ax = df.plot(
    x="Model",
    y=metrics_to_plot,
    kind="bar",
    figsize=(10, 6)
)

plt.title("Model Performance Comparison")
plt.xlabel("Models")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.grid(True)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "model_performance_plot.png"),
    dpi=300
)
plt.close()

# ===============================
# 6️⃣ Model Size Plot
# ===============================

ax = df.plot(
    x="Model",
    y="Size(MB)",
    kind="bar",
    figsize=(10, 6)
)

plt.title("Model Size Comparison")
plt.xlabel("Models")
plt.ylabel("Size (MB)")
plt.xticks(rotation=0)
plt.grid(True)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "model_size_plot.png"),
    dpi=300
)
plt.close()

print(f"\n✅ Comparison files saved in: {OUTPUT_DIR}")
