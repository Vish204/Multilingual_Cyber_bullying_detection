import json
from pathlib import Path

# ======================
# PATHS
# ======================

BASE_DIR = Path(__file__).resolve().parents[3]

MODEL_DIR = BASE_DIR / "models" / "emotion" / "final"
METRICS_PATH = BASE_DIR / "notebooks" / "analysis_results" / "emotion" / "metrics.json"

# ======================
# LOAD METRICS
# ======================

with open(METRICS_PATH, "r") as f:
    metrics_dict = json.load(f)

# ======================
# MODEL SIZE
# ======================

def get_model_size_mb(model_dir):
    total_size = 0
    for path in Path(model_dir).rglob("*"):
        if path.is_file():
            total_size += path.stat().st_size
    return total_size / (1024 * 1024)

# ======================
# BASELINE METRICS
# ======================

baseline_metrics = {
    "model_name": "emotion_xlmroberta",
    "accuracy": float(metrics_dict["accuracy"]),
    "precision": float(metrics_dict["precision"]),
    "recall": float(metrics_dict["recall"]),
    "f1_score": float(metrics_dict["weighted_f1"]),
    "roc_auc": float(metrics_dict["roc_auc"]) if metrics_dict.get("roc_auc") else None,
    "parameters": None,
    "model_size_mb": float(get_model_size_mb(MODEL_DIR))
}

# ======================
# PRINT
# ======================

print("\n===== Emotion Baseline Metrics =====")
print(json.dumps(baseline_metrics, indent=4))