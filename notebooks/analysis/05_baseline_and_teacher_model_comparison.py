import os
import json
import pandas as pd
import matplotlib.pyplot as plt


# ===============================
# 1️⃣ Define Paths
# ===============================

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../analysis_results"))

MODEL_PATHS = {
    "Baseline_XGBoost": os.path.join(BASE_PATH, "baseline_xgboost", "metrics.json"),
    "MuRIL": os.path.join(BASE_PATH, "muril", "metrics.json"),
    "mBERT": os.path.join(BASE_PATH, "mbert", "metrics.json"),
    "XLM-R": os.path.join(BASE_PATH, "xlmr", "metrics.json"),
}

OUTPUT_DIR = os.path.join(BASE_PATH, "model_comparison")
os.makedirs(OUTPUT_DIR, exist_ok=True)
for model_name, path in MODEL_PATHS.items():
    print(f"{model_name}: {path}")

print("CWD:", os.getcwd())
print("__file__:", __file__)
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

    results.append({
        "Model": model_name,
        "Accuracy": metrics.get("accuracy", 0),
        "Precision": metrics.get("precision", 0),
        "Recall": metrics.get("recall", 0),
        "F1-Score": metrics.get("f1_score", 0),
        "ROC-AUC": metrics.get("roc_auc", 0)
    })

# Convert to DataFrame
df = pd.DataFrame(results)

# ===============================
# 3️⃣ Save Comparison Table
# ===============================

csv_path = os.path.join(OUTPUT_DIR, "model_comparison.csv")
txt_path = os.path.join(OUTPUT_DIR, "model_comparison.txt")

df.to_csv(csv_path, index=False)

with open(txt_path, "w") as f:
    f.write(df.to_string(index=False))

print("\n Model Comparison:\n")
print(df)

# ===============================
# 4️⃣ Plot Comparison
# ===============================

metrics_to_plot = ["Accuracy", "F1-Score", "ROC-AUC"]

plt.figure(figsize=(10, 6))

for metric in metrics_to_plot:
    plt.plot(df["Model"], df[metric], marker='o', label=metric)

plt.title("Model Performance Comparison")
plt.xlabel("Models")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.legend()
plt.grid(True)

plot_path = os.path.join(OUTPUT_DIR, "model_comparison_plot.png")
plt.savefig(plot_path)
plt.close()

print(f"\n✅ Comparison files saved in: {OUTPUT_DIR}")
