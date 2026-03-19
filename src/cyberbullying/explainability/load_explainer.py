import joblib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_DIR = PROJECT_ROOT / "models/student_v2"

def load_artifacts():
    student = joblib.load(MODEL_DIR / "student_xgb_model.pkl")
    word_vectorizer = joblib.load(MODEL_DIR / "word_tfidf.pkl")
    char_vectorizer = joblib.load(MODEL_DIR / "char_tfidf.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    keywords = joblib.load(MODEL_DIR / "keywords.pkl")

    return {
        "model": student,
        "word_vectorizer": word_vectorizer,
        "char_vectorizer": char_vectorizer,
        "scaler": scaler,
        "keywords": keywords
    }