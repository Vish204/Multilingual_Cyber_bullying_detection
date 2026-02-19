import pandas as pd

train_df = pd.read_csv("data/processed/train_data.csv")
test_df = pd.read_csv("data/processed/test_data.csv")
val_df = pd.read_csv("data/processed/val_data.csv")

train_texts = set(train_df["text"])
test_texts = set(test_df["text"])
val_texts = set(val_df["text"])

print("Train-Test overlap:", len(train_texts & test_texts))
print("Train-Val overlap:", len(train_texts & val_texts))
print("Test-Val overlap:", len(test_texts & val_texts))
