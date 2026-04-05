import joblib
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pathlib import Path
import sys
import re
import time

# ------------------------------------------------
# Path Setup
# ------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

sys.path.append(str(PROJECT_ROOT / "src"))
sys.path.append(str(PROJECT_ROOT / "src/cyberbullying/sarcasm"))

from model import SarcasmModel

BASE_DIR = PROJECT_ROOT

# ------------------------------------------------
# Model Paths
# ------------------------------------------------

# Student V2
STUDENT_MODEL_PATH = BASE_DIR / "models" / "student_v2" / "student_xgb_model.pkl"
WORD_TFIDF_PATH = BASE_DIR / "models" / "student_v2" / "word_tfidf.pkl"
CHAR_TFIDF_PATH = BASE_DIR / "models" / "student_v2" / "char_tfidf.pkl"
SCALER_PATH = BASE_DIR / "models" / "student_v2" / "scaler.pkl"
KEYWORDS_PATH = BASE_DIR / "models" / "student_v2" / "keywords.pkl"

# Sarcasm
SARCASM_MODEL_PATH = BASE_DIR / "models" / "sarcasm" / "best_model.pt"

# Emotion
EMOTION_MODEL_PATH = BASE_DIR / "models" / "emotion" / "final"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------
# Load Student Model (with time measurement)
# ------------------------------------------------

def build_keyword_patterns(keywords):

    patterns = []

    for kw in keywords:

        pattern = r"\b" + re.escape(kw) + r"\b"
        patterns.append(re.compile(pattern, re.IGNORECASE))

    return patterns

def load_student_model():

    print("Loading Student_V2 model...")
    start = time.time()

    student_model = joblib.load(STUDENT_MODEL_PATH)

    word_vectorizer = joblib.load(WORD_TFIDF_PATH)

    char_vectorizer = joblib.load(CHAR_TFIDF_PATH)

    scaler = joblib.load(SCALER_PATH)

    keywords = joblib.load(KEYWORDS_PATH)

    keyword_patterns = build_keyword_patterns(keywords)

    duration = (time.time() - start) * 1000
    print(f"Student model loaded in {duration:.2f} ms")

    return student_model, word_vectorizer, char_vectorizer, scaler, keyword_patterns, duration


# ------------------------------------------------
# Load Sarcasm Model
# ------------------------------------------------

def load_sarcasm_model():

    print("Loading Sarcasm model...")
    start = time.time()
    sarcasm_model = torch.load(SARCASM_MODEL_PATH, map_location="cpu")

    sarcasm_model.eval()

    duration = (time.time() - start) * 1000
    print(f"Sarcasm model loaded in {duration:.2f} ms")

    return sarcasm_model, duration


# ------------------------------------------------
# Load Emotion Model
# ------------------------------------------------

def load_emotion_model():

    print("Loading Emotion model...")
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(EMOTION_MODEL_PATH)

    model = AutoModelForSequenceClassification.from_pretrained(
        EMOTION_MODEL_PATH
    )

    model.eval()
    duration = (time.time() - start) * 1000
    print(f"Emotion model loaded in {duration:.2f} ms")

    return tokenizer, model, duration


# ------------------------------------------------
# Load All Models
# ------------------------------------------------

def load_all_models():
    total_start = time.time()

    student_model, word_vec, char_vec, scaler, keyword_patterns, t_cb = load_student_model()

    sarcasm_model, t_sarcasm = load_sarcasm_model()

    emotion_tokenizer, emotion_model, t_emotion = load_emotion_model()

    print("All models loaded successfully!")

    total_duration = (time.time() - total_start) * 1000
    print(f"\n✅ Total models loaded in {total_duration:.2f} ms\n")

    return {

        "student": student_model,

        "word_vectorizer": word_vec,

        "char_vectorizer": char_vec,

        "scaler": scaler,

        "keyword_patterns": keyword_patterns,

        "sarcasm": sarcasm_model,

        "emotion_tokenizer": emotion_tokenizer,

        "emotion_model": emotion_model,
        "load_times": {
            "student": t_cb,
            "sarcasm": t_sarcasm,
            "emotion": t_emotion,
            "total": total_duration
        }
}

# ------------------------------------------------
# Standalone Test
# ------------------------------------------------

if __name__ == "__main__":

    models = load_all_models()

    print("\nLoaded models:")

    for key in models:
        print(f" - {key}")