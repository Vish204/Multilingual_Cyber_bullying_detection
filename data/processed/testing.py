# import pandas as pd

# train_df = pd.read_csv("data/processed/train_data.csv")
# test_df = pd.read_csv("data/processed/test_data.csv")
# val_df = pd.read_csv("data/processed/val_data.csv")

# train_texts = set(train_df["text"])
# test_texts = set(test_df["text"])
# val_texts = set(val_df["text"])

# print("Train-Test overlap:", len(train_texts & test_texts))
# print("Train-Val overlap:", len(train_texts & val_texts))
# print("Test-Val overlap:", len(test_texts & val_texts))

#####
#Measure current dataset distribution.
#####

import pandas as pd



print("Training set distribution:")
df = pd.read_csv("data/processed/train_data.csv")

# Total distribution per language
print(df["language"].value_counts())

# Percentage distribution
print(df["language"].value_counts(normalize=True) * 100)

# Label distribution per language
print(pd.crosstab(df["language"], df["label"]))


print("val set distribution:")
df = pd.read_csv("data/processed/val_data.csv")

# Total distribution per language
print(df["language"].value_counts())

# Percentage distribution
print(df["language"].value_counts(normalize=True) * 100)

# Label distribution per language
print(pd.crosstab(df["language"], df["label"]))




print("Testing set distribution:")
df = pd.read_csv("data/processed/test_data.csv")

# Total distribution per language
print(df["language"].value_counts())

# Percentage distribution
print(df["language"].value_counts(normalize=True) * 100)

# Label distribution per language
print(pd.crosstab(df["language"], df["label"]))