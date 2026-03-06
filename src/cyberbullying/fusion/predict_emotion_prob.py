import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

# ------------------------
# Config
# ------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "test_data.csv"

MODEL_PATH = PROJECT_ROOT / "models" / "emotion" / "final"

OUTPUT_PATH = PROJECT_ROOT / "notebooks" / "analysis_results" / "fusion" / "emotion_probs.csv"


# ------------------------
# Load Data
# ------------------------

print("\nLoading test dataset...")

df = pd.read_csv(DATA_PATH)
texts = df["text"].astype(str).tolist()

print(f"Samples: {len(texts)}")


# ------------------------
# Load Emotion Model
# ------------------------

print("\nLoading emotion model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

model.to(DEVICE)
model.eval()


# ------------------------
# Prediction
# ------------------------

print("\nGenerating emotion probabilities...")

emotion_scores = []

with torch.no_grad():

    for i in tqdm(range(0, len(texts), BATCH_SIZE)):

        batch = texts[i:i+BATCH_SIZE]

        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(DEVICE)

        outputs = model(**enc)

        probs = F.softmax(outputs.logits, dim=1)

        # Use max emotion intensity as emotion score
        max_probs = probs.max(dim=1).values.cpu().numpy()

        emotion_scores.extend(max_probs)


# ------------------------
# Save
# ------------------------

out_df = pd.DataFrame({
    "text": texts,
    "p_emotion": emotion_scores
})

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
out_df.to_csv(OUTPUT_PATH, index=False)

print(f"\nSaved emotion probabilities to: {OUTPUT_PATH}")