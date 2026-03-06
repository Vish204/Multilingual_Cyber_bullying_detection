from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "xlm-roberta-base"
NUM_LABELS = 3


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return tokenizer


def load_model():
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS
    )
    return model