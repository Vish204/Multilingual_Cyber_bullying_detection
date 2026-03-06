import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]

RAW = BASE_DIR / "data" / "sarcasm" / "raw"
PROCESSED = BASE_DIR / "data" / "sarcasm" / "processed"

PROCESSED.mkdir(exist_ok=True)

# ======================
# TWITTER HASHTAG
# ======================

def load_twitter_hashtag():

    text_file = RAW / "twitter_hashtag" / "train.txt"
    label_file = RAW / "twitter_hashtag" / "labels_train.txt"

    texts = open(text_file, encoding="utf-8").read().splitlines()
    labels = open(label_file, encoding="utf-8").read().splitlines()

    cleaned_text = []
    cleaned_label = []

    for t,l in zip(texts,labels):

        parts = t.split("\t")

        if len(parts) == 3:
            _, label_from_text, tweet = parts

            cleaned_text.append(tweet)
            cleaned_label.append(int(label_from_text))

    df = pd.DataFrame({
        "text": cleaned_text,
        "label": cleaned_label
    })

    print("Twitter Hashtag:",len(df))

    df = df.drop_duplicates()

    print("After dedupe:",len(df))

    df = df.sample(n=15000, random_state=42)

    return df


# ======================
# SEMEVAL
# ======================

def load_semeval():

    path = RAW / "twitter_semeval" / "semeval_train.txt"

    df = pd.read_csv(path,sep="\t")

    df.columns = ["id","label","text"]

    df = df[["text","label"]]

    df = df[df["label"].isin([0,1])]

    print("SemEval:",len(df))

    df = df.drop_duplicates()

    return df


# ======================
# REDDIT
# ======================

def load_reddit():

    path = RAW / "reddit_sarcasm" / "reddit_sarcasm.csv"

    df = pd.read_csv(path)

    df = df[["comment","label"]]

    df = df.rename(columns={"comment":"text"})

    print("Reddit:",len(df))

    df = df.drop_duplicates()

    print("After dedupe:",len(df))

    df = df.sample(n=30000, random_state=42)

    return df


# ======================
# MERGE
# ======================

def merge_datasets():

    twitter = load_twitter_hashtag()
    semeval = load_semeval()
    reddit = load_reddit()

    final_df = pd.concat([twitter,semeval,reddit])

    final_df = final_df.sample(frac=1).reset_index(drop=True)

    print("\nFinal Dataset Size:",len(final_df))

    print("\nLabel Distribution:")
    print(final_df["label"].value_counts())

    save_path = PROCESSED / "sarcasm_clean.csv"

    final_df.to_csv(save_path,index=False)

    print("\nSaved to:",save_path)


# ======================
# MAIN
# ======================

if __name__ == "__main__":

    merge_datasets()