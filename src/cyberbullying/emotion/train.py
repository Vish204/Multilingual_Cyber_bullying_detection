import pandas as pd
import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from transformers.trainer_utils import get_last_checkpoint
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from model import load_tokenizer, load_model
from transformers import EarlyStoppingCallback

# ======================
# PATHS
# ======================

BASE_DIR = Path(__file__).resolve().parents[3]
SPLIT_DIR = BASE_DIR / "data" / "emotion" / "splits"

TRAIN_PATH = SPLIT_DIR / "train.csv"
VAL_PATH = SPLIT_DIR / "val.csv"

# ======================
# LOAD DATA
# ======================

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)

print("Train samples:", len(train_df))
print("Validation samples:", len(val_df))

# ======================
# LOAD TOKENIZER
# ======================

tokenizer = load_tokenizer()

# ======================
# DATASET CLASS
# ======================

class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }

# ======================
# CREATE DATASETS
# ======================

train_dataset = EmotionDataset(
    train_df["text"].values,
    train_df["label"].values,
    tokenizer
)

val_dataset = EmotionDataset(
    val_df["text"].values,
    val_df["label"].values,
    tokenizer
)

print("Datasets created successfully.")


# ======================
# LOAD MODEL
# ======================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = load_model()
model.to(device)

print("Model loaded and moved to device.")




# ======================
# METRICS FUNCTION
# ======================

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted"
    )

    acc = accuracy_score(labels, predictions)

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }


# ======================
# TRAINING ARGUMENTS
# ======================

training_args = TrainingArguments(
    output_dir="models/emotion",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    warmup_ratio=0.1,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_dir="models/emotion/logs",
    logging_steps=100,
    save_total_limit=1,
    fp16=torch.cuda.is_available(),
    report_to="none"
)


# ======================
# TRAINER
# ======================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)


# ======================
# START TRAINING
# ======================

print("Starting training...")
checkpoint = None
if os.path.isdir(training_args.output_dir):
    checkpoint = get_last_checkpoint(training_args.output_dir)

print("Resuming from:", checkpoint)

trainer.train(resume_from_checkpoint=checkpoint)

print("Training complete.")

# ======================
# FINAL EVALUATION
# ======================

metrics = trainer.evaluate()
print("Final Evaluation Metrics:", metrics)