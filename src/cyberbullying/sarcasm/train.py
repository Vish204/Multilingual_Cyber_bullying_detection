import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from pathlib import Path
from collections import Counter
from model import SarcasmModel
import json

# =========================
# CONFIG
# =========================
BASE_DIR = Path(__file__).resolve().parents[3]
SPLIT_DIR = BASE_DIR / "data" / "sarcasm" / "splits"
MODEL_DIR = BASE_DIR / "models" / "sarcasm"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-3
PATIENCE = 3
MAX_LEN = 50
EMBED_DIM = 128
HIDDEN_DIM = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", DEVICE)

# =========================
# LOAD DATA (Train + Val Only)
# =========================
train_df = pd.read_csv(SPLIT_DIR / "train.csv")
val_df = pd.read_csv(SPLIT_DIR / "val.csv")

# =========================
# BUILD VOCAB (TRAIN ONLY)
# =========================
def build_vocab(df, min_freq=2):
    counter = Counter()

    for text in df["text"]:
        words = text.lower().strip().split()
        counter.update(words)

    vocab = {
        "<PAD>": 0,
        "<UNK>": 1
    }

    idx = 2
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = idx
            idx += 1

    return vocab

vocab = build_vocab(train_df)
VOCAB_SIZE = len(vocab)

print("Vocabulary size:", VOCAB_SIZE)

VOCAB_PATH = MODEL_DIR / "vocab.json"

with open(VOCAB_PATH, "w") as f:
    json.dump(vocab, f)

print(f"Vocabulary saved at: {VOCAB_PATH}")

# =========================
# DATASET CLASS
# =========================
class SarcasmDataset(Dataset):
    def __init__(self, df, vocab, max_len):
        self.texts = df["text"].tolist()
        self.labels = df["label"].tolist()
        self.vocab = vocab
        self.max_len = max_len

    def encode(self, text):
        tokens = text.lower().strip().split()
        ids = [self.vocab.get(word, self.vocab["<UNK>"]) for word in tokens]

        if len(ids) < self.max_len:
            ids += [self.vocab["<PAD>"]] * (self.max_len - len(ids))
        else:
            ids = ids[:self.max_len]

        return ids

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        token_ids = self.encode(self.texts[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.float)

        return torch.tensor(token_ids, dtype=torch.long), label


train_dataset = SarcasmDataset(train_df, vocab, MAX_LEN)
val_dataset = SarcasmDataset(val_df, vocab, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# =========================
# MODEL INITIALIZATION
# =========================
model = SarcasmModel(
    vocab_size=VOCAB_SIZE,
    embedding_dim=EMBED_DIM,
    hidden_size=HIDDEN_DIM
)

model.to(DEVICE)

# Since model already has sigmoid → use BCELoss
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# =========================
# TRAINING LOOP
# =========================
best_val_f1 = 0
patience_counter = 0

for epoch in range(EPOCHS):

    # ===== TRAIN =====
    model.train()
    train_loss_list = []
    train_preds, train_labels = [], []

    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(x_batch).squeeze(1)  # already sigmoid output
        loss = criterion(outputs, y_batch)

        loss.backward()
        optimizer.step()

        train_loss_list.append(loss.item())

        preds = (outputs > 0.5).float().detach().cpu().numpy()
        train_preds.extend(preds)
        train_labels.extend(y_batch.detach().cpu().numpy())

    train_f1 = f1_score(train_labels, train_preds)
    train_acc = accuracy_score(train_labels, train_preds)
    train_loss = sum(train_loss_list) / len(train_loss_list)

    # ===== VALIDATION =====
    model.eval()
    val_loss_list = []
    val_preds, val_labels = [], []

    with torch.no_grad():
        for x_val, y_val in val_loader:
            x_val = x_val.to(DEVICE)
            y_val = y_val.to(DEVICE)

            outputs = model(x_val).squeeze(1)
            loss = criterion(outputs, y_val)

            val_loss_list.append(loss.item())

            preds = (outputs > 0.5).float().detach().cpu().numpy()
            val_preds.extend(preds)
            val_labels.extend(y_val.detach().cpu().numpy())

    val_f1 = f1_score(val_labels, val_preds)
    val_acc = accuracy_score(val_labels, val_preds)
    val_loss = sum(val_loss_list) / len(val_loss_list)

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f}, F1: {train_f1:.4f}, Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f}, F1: {val_f1:.4f}, Acc: {val_acc:.4f}"
    )

    # ===== EARLY STOPPING =====
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        patience_counter = 0

        torch.save(model, MODEL_DIR / "best_model.pt")
        print("✅ Checkpoint saved (best_model.pt)")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("⛔ Early stopping triggered")
            break

# ===== SAVE FINAL MODEL =====
torch.save(model, MODEL_DIR / "final_model.pt")
print("✅ Final model saved at:", MODEL_DIR / "final_model.pt")