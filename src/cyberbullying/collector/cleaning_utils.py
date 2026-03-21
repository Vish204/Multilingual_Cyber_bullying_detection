import re

def clean_text(text: str):

    if not text:
        return ""

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    # Keep emojis (IMPORTANT - your decision)
    # Remove weird symbols only
    text = re.sub(r"[^\w\s\u0900-\u097F\u0B80-\u0BFF\u0980-\u09FF.,!?]", "", text)

    return text.strip()