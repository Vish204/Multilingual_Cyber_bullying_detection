import os
import json
import pandas as pd
import matplotlib.pyplot as plt


# ===============================
# PATHS
# ===============================

BASE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../analysis_results")
)

MODEL_PATHS = {

    "Student_Old_MTKD": os.path.join(BASE_PATH, "mtkd_xgboost", "metrics.json"),

    "Student_V2": os.path.join(BASE_PATH, "student_v2", "metrics.json"),
}

OUTPUT_DIR = os.path.join(BASE_PATH, "student_comparison")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===============================
# LOAD METRICS
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

        "Size(MB)": float(metrics.get("model_size_mb", 0)),

        "Accuracy": float(metrics.get("accuracy", 0)),
        "Precision": float(metrics.get("precision", 0)),
        "Recall": float(metrics.get("recall", 0)),
        "F1-Score": float(metrics.get("f1_score", 0)),
        "ROC-AUC": float(metrics.get("roc_auc", 0)),
    })


df = pd.DataFrame(results)

if df.empty:

    print("No metrics found.")
    exit()


print("\nStudent Model Comparison:\n")
print(df)


# ===============================
# SAVE TABLE
# ===============================

df.to_csv(
    os.path.join(OUTPUT_DIR, "student_comparison.csv"),
    index=False
)


# ===============================
# METRIC BAR PLOT
# ===============================

metrics_to_plot = ["Accuracy", "F1-Score", "ROC-AUC"]

df.plot(
    x="Model",
    y=metrics_to_plot,
    kind="bar",
    figsize=(8,5)
)

plt.title("Student Model Comparison")

plt.ylabel("Score")

plt.ylim(0,1)

plt.xticks(rotation=0)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "student_performance_plot.png"),
    dpi=300
)

plt.close()


# ===============================
# SIZE COMPARISON
# ===============================

df.plot(
    x="Model",
    y="Size(MB)",
    kind="bar",
    figsize=(8,5)
)

plt.title("Student Model Size Comparison")

plt.ylabel("Size (MB)")

plt.xticks(rotation=0)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "student_size_plot.png"),
    dpi=300
)

plt.close()


print(f"\n✅ Student comparison saved in: {OUTPUT_DIR}")