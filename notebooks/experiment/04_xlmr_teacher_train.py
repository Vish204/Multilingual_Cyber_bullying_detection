import os
import json
import logging
import warnings
import mlflow
import mlflow.pytorch
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
)

# ----------------------------
# Basic Setup
# ----------------------------

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"


# ----------------------------
# Custom Trainer (for class weights)
# ----------------------------

class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fct = torch.nn.CrossEntropyLoss(
            weight=self.class_weights.to(device)
        )
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss


# ----------------------------
# Trainer Class
# ----------------------------

class TeacherTrainer:

    def __init__(self, model_name, model_folder, num_labels=2):
        self.model_name = model_name

        self.model_dir = PROJECT_ROOT / "models" / "teacher" / model_folder
        self.model_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Loading model: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label={0: "non_toxic", 1: "toxic"},
            label2id={"non_toxic": 0, "toxic": 1}
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        self.model.to(device)

    # ----------------------------
    # Dataset Preparation
    # ----------------------------
    def prepare_dataset(self, texts, labels):

        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=256
        )

        return Dataset.from_dict({
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": labels
        })

    # ----------------------------
    # Metrics
    # ----------------------------
    @staticmethod
    def compute_metrics(eval_pred):

        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels,
            predictions,
            average="weighted"
        )

        acc = accuracy_score(labels, predictions)

        return {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    # ----------------------------
    # Training
    # ----------------------------
    def train(self, train_df, val_df):

        logger.info("Preparing datasets...")

        train_dataset = self.prepare_dataset(
            train_df["text"].tolist(),
            train_df["label"].tolist()
        )

        val_dataset = self.prepare_dataset(
            val_df["text"].tolist(),
            val_df["label"].tolist()
        )

        # Class weights
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(train_df["label"]),
            y=train_df["label"]
        )

        class_weights = torch.tensor(class_weights, dtype=torch.float)

        logger.info(f"Class weights: {class_weights}")

        training_args = TrainingArguments(
            output_dir=str(self.model_dir),
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=6,
            weight_decay=0.01,
            warmup_ratio=0.1,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            logging_dir=str(self.model_dir / "logs"),
            logging_steps=50,
            save_total_limit=1,
            fp16=torch.cuda.is_available(),
            report_to="tensorboard",
        )

        trainer = WeightedTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            data_collator=DataCollatorWithPadding(self.tokenizer),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
            class_weights=class_weights
        )

        logger.info("Starting training...")
        trainer.train()

        # Validation metrics
        metrics = trainer.evaluate()
        mlflow.log_metrics(metrics)

        # Save model to MLflow
        mlflow.pytorch.log_model(self.model, "model")

        # Save local model
        final_model_path = self.model_dir / "final_model"
        trainer.save_model(str(final_model_path))
        self.tokenizer.save_pretrained(str(final_model_path))

        logger.info(f"✓ Model saved to {final_model_path}")

        return trainer

    # ----------------------------
    # Test Evaluation
    # ----------------------------
    def evaluate_test(self, trainer, test_df):

        logger.info("Evaluating on TEST set...")

        test_dataset = self.prepare_dataset(
            test_df["text"].tolist(),
            test_df["label"].tolist()
        )

        metrics = trainer.evaluate(test_dataset)

        metrics_path = self.model_dir / "test_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)

        logger.info("Test Metrics:")
        for k, v in metrics.items():
            logger.info(f"{k}: {v:.4f}")

        logger.info(f"✓ Test metrics saved to {metrics_path}")

        # Log test metrics
        mlflow.log_metrics({
            "test_accuracy": metrics["eval_accuracy"],
            "test_precision": metrics["eval_precision"],
            "test_recall": metrics["eval_recall"],
            "test_f1": metrics["eval_f1"]
        })


# ----------------------------
# Main Execution
# ----------------------------

def main():

    logger.info("=" * 60)
    logger.info("TRAINING TEACHER MODEL")
    logger.info("=" * 60)

    train_df = pd.read_csv(DATA_DIR / "train_data.csv")
    val_df = pd.read_csv(DATA_DIR / "val_data.csv")
    test_df = pd.read_csv(DATA_DIR / "test_data.csv")

    train_df["label"] = train_df["label"].astype(int)
    val_df["label"] = val_df["label"].astype(int)
    test_df["label"] = test_df["label"].astype(int)

    logger.info(f"Train samples: {len(train_df)}")
    logger.info(f"Val samples: {len(val_df)}")
    logger.info(f"Test samples: {len(test_df)}")

    # -----------------------------------
    # CHANGE MODEL HERE ONLY
    # -----------------------------------

    MODEL_NAME = "xlm-roberta-base"
    MODEL_FOLDER = "xlmr"

    trainer_obj = TeacherTrainer(
        model_name=MODEL_NAME,
        model_folder=MODEL_FOLDER
    )

    # -----------------------------------
    # MLflow Run (CORRECT PLACE)
    # -----------------------------------

    mlflow.set_experiment("cyberbullying_teacher_models")

    with mlflow.start_run(run_name=MODEL_NAME):

        mlflow.log_param("model", MODEL_NAME)
        mlflow.log_param("epochs", 6)
        mlflow.log_param("batch_size", 8)
        mlflow.log_param("lr", 2e-5)

        mlflow.log_param("train_samples", len(train_df))
        mlflow.log_param("val_samples", len(val_df))
        mlflow.log_param("test_samples", len(test_df))

        trainer = trainer_obj.train(train_df, val_df)
        trainer_obj.evaluate_test(trainer, test_df)


if __name__ == "__main__":
    main()
