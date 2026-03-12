import joblib
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pathlib import Path
import sys

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

# Sarcasm
SARCASM_MODEL_PATH = BASE_DIR / "models" / "sarcasm" / "best_model.pt"

# Emotion
EMOTION_MODEL_PATH = BASE_DIR / "models" / "emotion" / "final"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------
# Load Student Model
# ------------------------------------------------

def load_student_model():

    print("Loading Student_V2 model...")

    student_model = joblib.load(STUDENT_MODEL_PATH)

    word_vectorizer = joblib.load(WORD_TFIDF_PATH)

    char_vectorizer = joblib.load(CHAR_TFIDF_PATH)

    scaler = joblib.load(SCALER_PATH)

    return student_model, word_vectorizer, char_vectorizer, scaler


# ------------------------------------------------
# Load Sarcasm Model
# ------------------------------------------------

def load_sarcasm_model():

    print("Loading Sarcasm model...")

    sarcasm_model = torch.load(SARCASM_MODEL_PATH, map_location="cpu")

    sarcasm_model.eval()

    return sarcasm_model


# ------------------------------------------------
# Load Emotion Model
# ------------------------------------------------

def load_emotion_model():

    print("Loading Emotion model...")

    tokenizer = AutoTokenizer.from_pretrained(EMOTION_MODEL_PATH)

    model = AutoModelForSequenceClassification.from_pretrained(
        EMOTION_MODEL_PATH
    )

    model.eval()

    return tokenizer, model


# ------------------------------------------------
# Load All Models
# ------------------------------------------------

def load_all_models():

    student_model, word_vec, char_vec, scaler = load_student_model()

    sarcasm_model = load_sarcasm_model()

    emotion_tokenizer, emotion_model = load_emotion_model()

    print("All models loaded successfully!")

    return {

        "student": student_model,

        "word_vectorizer": word_vec,

        "char_vectorizer": char_vec,

        "scaler": scaler,

        "sarcasm": sarcasm_model,

        "emotion_tokenizer": emotion_tokenizer,

        "emotion_model": emotion_model
    }


# ------------------------------------------------
# Standalone Test
# ------------------------------------------------

if __name__ == "__main__":

    models = load_all_models()

    print("\nLoaded models:")

    for key in models:
        print(f" - {key}")