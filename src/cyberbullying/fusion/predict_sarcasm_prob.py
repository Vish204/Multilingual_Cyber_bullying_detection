import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT / "src"))
sys.path.append(str(PROJECT_ROOT / "src/cyberbullying/sarcasm"))

import torch
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from model import SarcasmModel


# ------------------------
# Config
# ------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "test_data.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "sarcasm" / "best_model.pt"
VOCAB_PATH = PROJECT_ROOT / "models" / "sarcasm" / "vocab.json"

OUTPUT_PATH = PROJECT_ROOT / "notebooks" / "analysis_results" / "fusion" / "sarcasm_probs.csv"


# ------------------------
# Load Data
# ------------------------

print("\nLoading test dataset...")

df = pd.read_csv(DATA_PATH)
texts = df["text"].astype(str).tolist()

print(f"Samples: {len(texts)}")


# ------------------------
# Load Model
# ------------------------

print("\nLoading sarcasm model...")

model = torch.load(MODEL_PATH, map_location=DEVICE)
model.to(DEVICE)
model.eval()

print("Model loaded")


# ------------------------
# Tokenization helper
# ------------------------

def tokenize(text):
    return text.lower().split()


# ------------------------
# Load vocabulary
# ------------------------

with open(VOCAB_PATH) as f:
    vocab = json.load(f)

MAX_LEN = 50

def encode(text):
    tokens = tokenize(text)

    ids = [vocab.get(t, vocab.get("<UNK>", 1)) for t in tokens]

    if len(ids) < MAX_LEN:
        ids += [vocab.get("<PAD>", 0)] * (MAX_LEN - len(ids))
    else:
        ids = ids[:MAX_LEN]

    return ids


# ------------------------
# Prediction
# ------------------------

print("\nGenerating sarcasm probabilities...")

probs = []

with torch.no_grad():
    for text in tqdm(texts):

        encoded = encode(text)
        tensor = torch.tensor(encoded).unsqueeze(0).to(DEVICE)

        output = model(tensor)
        prob = output.item()

        probs.append(prob)


# ------------------------
# Save
# ------------------------

out_df = pd.DataFrame({
    "text": texts,
    "p_sarcasm": probs
})

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
out_df.to_csv(OUTPUT_PATH, index=False)

print(f"\nSaved sarcasm probabilities to: {OUTPUT_PATH}")