import joblib
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel
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

# Student
MTKD_MODEL_PATH = BASE_DIR / "models" / "student" / "mtkd_student_xgb.pkl"
TFIDF_PATH = BASE_DIR / "models" / "student" / "tfidf_vectorizer.pkl"
SCALER_PATH = BASE_DIR / "models" / "student" / "scaler.pkl"

# Teachers
MBERT_PATH = BASE_DIR / "models" / "teacher" / "mbert" / "final_model"
XLMR_PATH = BASE_DIR / "models" / "teacher" / "xlmr" / "final_model"
MURIL_PATH = BASE_DIR / "models" / "teacher" / "muril" / "final_model"

# Sarcasm
SARCASM_MODEL_PATH = BASE_DIR / "models" / "sarcasm" / "best_model.pt"

# Emotion
EMOTION_MODEL_PATH = BASE_DIR / "models" / "emotion" / "final"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------
# Load MTKD Student
# ------------------------------------------------

def load_mtkd_model():

    print("Loading MTKD XGBoost student model...")

    mtkd_model = joblib.load(MTKD_MODEL_PATH)
    vectorizer = joblib.load(TFIDF_PATH)
    scaler = joblib.load(SCALER_PATH)

    return mtkd_model, vectorizer, scaler


# ------------------------------------------------
# Load Teacher Models
# ------------------------------------------------

def load_teacher_models():

    print("Loading teacher models...")

    teachers = {}

    # mBERT
    teachers["mbert_tokenizer"] = AutoTokenizer.from_pretrained(MBERT_PATH)
    teachers["mbert_model"] = AutoModel.from_pretrained(MBERT_PATH).to(DEVICE)
    teachers["mbert_model"].eval()

    # XLMR
    teachers["xlmr_tokenizer"] = AutoTokenizer.from_pretrained(XLMR_PATH)
    teachers["xlmr_model"] = AutoModel.from_pretrained(XLMR_PATH).to(DEVICE)
    teachers["xlmr_model"].eval()

    # MuRIL
    teachers["muril_tokenizer"] = AutoTokenizer.from_pretrained(MURIL_PATH)
    teachers["muril_model"] = AutoModel.from_pretrained(MURIL_PATH).to(DEVICE)
    teachers["muril_model"].eval()

    return teachers


# ------------------------------------------------
# Load Sarcasm Model
# ------------------------------------------------

def load_sarcasm_model():

    print("Loading Sarcasm BiGRU model...")

    sarcasm_model = torch.load(SARCASM_MODEL_PATH, map_location="cpu")

    sarcasm_model.eval()

    return sarcasm_model


# ------------------------------------------------
# Load Emotion Model
# ------------------------------------------------

def load_emotion_model():

    print("Loading Emotion BERT model...")

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

    mtkd_model, vectorizer, scaler = load_mtkd_model()

    teacher_models = load_teacher_models()

    sarcasm_model = load_sarcasm_model()

    emotion_tokenizer, emotion_model = load_emotion_model()

    print("All models loaded successfully!")

    return {
        "mtkd": mtkd_model,
        "vectorizer": vectorizer,
        "scaler": scaler,

        "teachers": teacher_models,

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