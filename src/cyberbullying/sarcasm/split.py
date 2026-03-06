import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
DATA_PATH = Path("data/sarcasm/processed/sarcasm_clean.csv")
SAVE_DIR = Path("data/sarcasm/splits")

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv(DATA_PATH)

# Remove duplicate texts
before = len(df)
df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
after = len(df)

print(f"Removed duplicates: {before - after}")
print(f"Dataset after dedupe: {after}")

print("Total dataset:", len(df))
print("\nLabel distribution:")
print(df["label"].value_counts())

# First split → 70% train + 30% temp
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["label"],
    random_state=42
)

# Second split → 15% val + 15% test
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["label"],
    random_state=42
)

print("\nSplit sizes:")
print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))

# -----------------------------
# Label Distribution Check
# -----------------------------
print("\nTrain labels:")
print(train_df["label"].value_counts())

print("\nValidation labels:")
print(val_df["label"].value_counts())

print("\nTest labels:")
print(test_df["label"].value_counts())

# -----------------------------
# Duplicate Check inside splits
# -----------------------------
print("\nChecking duplicates inside splits...")

print("Train duplicates:", train_df.duplicated("text").sum())
print("Validation duplicates:", val_df.duplicated("text").sum())
print("Test duplicates:", test_df.duplicated("text").sum())

# -----------------------------
# Overlap Check between splits
# -----------------------------
print("\nChecking overlap between splits...")

train_text = set(train_df["text"])
val_text = set(val_df["text"])
test_text = set(test_df["text"])

print("Train ∩ Val:", len(train_text.intersection(val_text)))
print("Train ∩ Test:", len(train_text.intersection(test_text)))
print("Val ∩ Test:", len(val_text.intersection(test_text)))

# -----------------------------
# Save splits
# -----------------------------
train_df.to_csv(SAVE_DIR / "train.csv", index=False)
val_df.to_csv(SAVE_DIR / "val.csv", index=False)
test_df.to_csv(SAVE_DIR / "test.csv", index=False)

print("\nFiles saved:")
print("train.csv")
print("val.csv")
print("test.csv")