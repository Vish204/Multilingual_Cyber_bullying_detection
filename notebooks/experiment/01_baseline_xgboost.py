# 01_baseline_xgboost.py

import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import joblib
from pathlib import Path
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------
# Project Paths
# ----------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models" / "baseline_xgboost"


# ----------------------------
# Feature Extraction
# ----------------------------

def extract_features(texts):
    features = []
    for text in texts:
        text = str(text)
        features.append([
            len(text),  # Character length
            len(text.split()),  # Word count
            sum(1 for c in text if c.isupper()),  # Uppercase count
            text.count('!'),  # Exclamation marks
            text.count('?'),  # Question marks
        ])
    return features


# ----------------------------
# Main
# ----------------------------

def main():

    logger.info("=" * 60)
    logger.info("BASELINE XGBOOST TRAINING")
    logger.info("=" * 60)

    # Check required files
    required_files = ["train_data.csv", "val_data.csv", "test_data.csv"]
    for file in required_files:
        if not (DATA_DIR / file).exists():
            logger.error(f"Missing file: {file}")
            logger.error("Run preprocessing first.")
            return

    # Load data
    train_df = pd.read_csv(DATA_DIR / "train_data.csv")
    val_df = pd.read_csv(DATA_DIR / "val_data.csv")
    test_df = pd.read_csv(DATA_DIR / "test_data.csv")

    logger.info(f"Train samples: {len(train_df)}")
    logger.info(f"Validation samples: {len(val_df)}")
    logger.info(f"Test samples: {len(test_df)}")

    logger.info(f"Languages: {train_df['language'].nunique()}")

    # Prepare features
    logger.info("Extracting features...")

    X_train = extract_features(train_df["text"])
    y_train = train_df["label"].values

    X_val = extract_features(val_df["text"])
    y_val = val_df["label"].values

    X_test = extract_features(test_df["text"])
    y_test = test_df["label"].values

    # Train model
    logger.info("Training XGBoost model...")

    model = xgb.XGBClassifier(
        max_depth=5,
        n_estimators=100,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss"
    )

    model.fit(X_train, y_train)

    # Evaluate
    def evaluate(y_true, y_pred, split_name):
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted"
        )
        acc = accuracy_score(y_true, y_pred)

        logger.info(f"\n{split_name} Results:")
        logger.info(f"Accuracy : {acc:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall   : {recall:.4f}")
        logger.info(f"F1 Score : {f1:.4f}")

        return {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    logger.info("\nEvaluating model...")

    train_metrics = evaluate(y_train, model.predict(X_train), "Training")
    val_metrics = evaluate(y_val, model.predict(X_val), "Validation")
    test_metrics = evaluate(y_test, model.predict(X_test), "Test")

    # Save model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "baseline_xgboost.pkl"
    joblib.dump(model, model_path)

    logger.info(f"\n✓ Model saved to: {model_path}")

    # Save metrics
    metrics = {
        "train": train_metrics,
        "validation": val_metrics,
        "test": test_metrics
    }

    metrics_path = MODEL_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    logger.info(f"✓ Metrics saved to: {metrics_path}")

    logger.info("\n" + "=" * 60)
    logger.info("BASELINE TRAINING COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
