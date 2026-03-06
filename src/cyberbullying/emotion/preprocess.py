import os
from pathlib import Path
import pandas as pd

# =========================
# CONFIG
# =========================

BASE_DIR = Path(__file__).resolve().parents[3]

RAW_DATA_PATH = BASE_DIR / "data" / "emotion" / "raw"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "emotion" / "processed"
OUTPUT_FILE = "emotion_clean.csv"

# Bullying-relevant emotions
AGGRESSION_EMOTIONS = ["anger", "annoyance", "disgust", "disapproval"]
DISTRESS_EMOTIONS = ["sadness", "fear", "embarrassment", "remorse", "nervousness", "grief"]
NEUTRAL_EMOTION = ["neutral"]

# =========================
# LOAD DATA
# =========================

def load_data():
    files = [
        "goemotions_1.csv",
        "goemotions_2.csv",
        "goemotions_3.csv"
    ]

    dfs = []
    for file in files:
        path = os.path.join(RAW_DATA_PATH, file)
        print(f"Loading {path}")
        df = pd.read_csv(path)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    print("Total samples:", len(combined))
    return combined


# =========================
# PREPROCESS
# =========================

def preprocess(df):
    print("Starting preprocessing...")

    # Keep only text + relevant emotions
    emotion_columns = AGGRESSION_EMOTIONS + DISTRESS_EMOTIONS + NEUTRAL_EMOTION
    df = df[["text"] + emotion_columns]

    # Remove null text
    df = df.dropna(subset=["text"])

    # Strip whitespace
    df["text"] = df["text"].str.strip()

    # Remove empty strings
    df = df[df["text"] != ""]

    # Remove duplicates
    before_dup = len(df)
    df = df.drop_duplicates(subset=["text"])
    after_dup = len(df)

    print(f"Removed {before_dup - after_dup} duplicate rows")

    # Create grouped scores
    df["aggression_score"] = df[AGGRESSION_EMOTIONS].sum(axis=1)
    df["distress_score"] = df[DISTRESS_EMOTIONS].sum(axis=1)

    # Remove rows with no emotion signal
    df = df[(df["aggression_score"] > 0) |
            (df["distress_score"] > 0) |
            (df["neutral"] == 1)]

    # Create final 3-class label
    def assign_label(row):
        if row["aggression_score"] > 0:
            return 1
        elif row["distress_score"] > 0:
            return 2
        else:
            return 0

    df["label"] = df.apply(assign_label, axis=1)

    print("Remaining samples after filtering:", len(df))

    return df


# =========================
# SAVE DATA
# =========================

def save_data(df):
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    output_path = os.path.join(PROCESSED_DATA_PATH, OUTPUT_FILE)

    df.to_csv(output_path, index=False)
    print(f"Saved cleaned data to {output_path}")


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    data = load_data()
    clean_data = preprocess(data)
    save_data(clean_data)