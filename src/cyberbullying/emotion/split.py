import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# =========================
# PATH CONFIG
# =========================

BASE_DIR = Path(__file__).resolve().parents[3]
PROCESSED_PATH = BASE_DIR / "data" / "emotion" / "processed" / "emotion_clean.csv"
SPLIT_DIR = BASE_DIR / "data" / "emotion" / "splits"

# =========================
# LOAD DATA
# =========================

def load_data():
    df = pd.read_csv(PROCESSED_PATH)
    print("Loaded dataset:", len(df))
    return df

# =========================
# STRATIFIED SPLIT
# =========================

def stratified_split(df):

    # First split: Train (70%) + Temp (30%)
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df["label"],
        random_state=42
    )

    # Second split: Temp -> Val (15%) + Test (15%)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["label"],
        random_state=42
    )

    return train_df, val_df, test_df

# =========================
# SAVE SPLITS
# =========================

def save_splits(train_df, val_df, test_df):

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(SPLIT_DIR / "train.csv", index=False)
    val_df.to_csv(SPLIT_DIR / "val.csv", index=False)
    test_df.to_csv(SPLIT_DIR / "test.csv", index=False)

    print("Splits saved successfully.")

# =========================
# LEAKAGE CHECK
# =========================

def check_leakage(train_df, val_df, test_df):

    train_texts = set(train_df["text"])
    val_texts = set(val_df["text"])
    test_texts = set(test_df["text"])

    print("\nLeakage Check:")
    print("Train-Val overlap:", len(train_texts & val_texts))
    print("Train-Test overlap:", len(train_texts & test_texts))
    print("Val-Test overlap:", len(val_texts & test_texts))

# =========================
# DISTRIBUTION CHECK
# =========================

def print_distribution(name, df):
    print(f"\n{name} Distribution:")
    print(df["label"].value_counts())

# =========================
# MAIN
# =========================

if __name__ == "__main__":

    df = load_data()

    train_df, val_df, test_df = stratified_split(df)

    save_splits(train_df, val_df, test_df)

    check_leakage(train_df, val_df, test_df)

    print_distribution("Train", train_df)
    print_distribution("Validation", val_df)
    print_distribution("Test", test_df)