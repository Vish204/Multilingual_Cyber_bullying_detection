import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from pathlib import Path
import logging
import warnings
import os
from tqdm import tqdm

# Force CPU to avoid MPS errors on Mac
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_soft_labels(model_path, data_df, batch_size=8):
    """Run data through Teacher model to get probabilities (Soft Labels)"""
    logger.info(f"Loading Teacher Model from {model_path}...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.eval() # Set to evaluation mode
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None

    soft_labels = []
    texts = data_df['text'].tolist()
    
    logger.info("Generating soft labels from Teacher...")
    
    # Process in batches
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize
        inputs = tokenizer(
            batch_texts, 
            padding=True, 
            truncation=True, 
            max_length=128, 
            return_tensors="pt"
        )
        
        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            # Get probabilities using Softmax
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            # We want the probability of class 1 (Bullying)
            bullying_probs = probs[:, 1].numpy()
            soft_labels.extend(bullying_probs)
            
    return np.array(soft_labels)

def extract_student_features(train_texts, val_texts):
    """Create features for the Student (XGBoost)"""
    logger.info("Extracting TF-IDF features for Student...")
    
    # Use TF-IDF (Simple, Fast)
    vectorizer = TfidfVectorizer(
        max_features=2000,
        ngram_range=(1, 2),
        stop_words='english'
    )
    
    X_train = vectorizer.fit_transform(train_texts).toarray()
    X_val = vectorizer.transform(val_texts).toarray()
    
    # Add meta-features (length, etc.)
    def add_meta(texts, features):
        meta = []
        for t in texts:
            t = str(t)
            meta.append([
                len(t), 
                len(t.split()),
                sum(1 for c in t if c.isupper())
            ])
        return np.hstack([features, np.array(meta)])
        
    X_train = add_meta(train_texts, X_train)
    X_val = add_meta(val_texts, X_val)
    
    return X_train, X_val, vectorizer

def main():
    logger.info("="*60)
    logger.info("STARTING KNOWLEDGE DISTILLATION")
    logger.info("="*60)
    
    # 1. Load Data
    data_dir = Path("data/processed_final")
    if not (data_dir / "train_data.csv").exists():
        logger.error("Data not found. Run preprocessing first.")
        return

    train_df = pd.read_csv(data_dir / "train_data.csv")
    val_df = pd.read_csv(data_dir / "val_data.csv")
    
    logger.info(f"Loaded {len(train_df)} training samples")

    # 2. Generate Soft Labels (Teacher's Knowledge)
    teacher_path = "models/teacher_models/muril_full/final_model"
    if not Path(teacher_path).exists():
        logger.error(f"Teacher model not found at {teacher_path}. Did training finish?")
        return

    # Use CPU for generation
    soft_targets_train = generate_soft_labels(teacher_path, train_df)
    if soft_targets_train is None: return

    # 3. Prepare Student Features
    X_train, X_val, vectorizer = extract_student_features(
        train_df['text'].fillna(""), 
        val_df['text'].fillna("")
    )
    
    # Get hard labels for validation (to check accuracy)
    y_val_hard = val_df['label'].values

    # 4. Train Student Model (Distilled XGBoost)
    # We train a Regressor to predict the Teacher's Probability Score
    logger.info("Training Distilled Student (XGBoost)...")
    
    student_model = xgb.XGBRegressor(
        objective='reg:squarederror', 
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        n_jobs=-1
    )
    
    student_model.fit(X_train, soft_targets_train)
    
    # 5. Evaluate
    logger.info("Evaluating Student...")
    val_preds_proba = student_model.predict(X_val)
    
    # Convert probabilities to 0/1 for accuracy check
    val_preds_class = [1 if p > 0.5 else 0 for p in val_preds_proba]
    
    acc = accuracy_score(y_val_hard, val_preds_class)
    f1 = f1_score(y_val_hard, val_preds_class)
    
    logger.info("\n" + "="*60)
    logger.info("DISTILLATION RESULTS")
    logger.info("="*60)
    logger.info(f"Student Accuracy: {acc:.4f}")
    logger.info(f"Student F1 Score: {f1:.4f}")
    
    # 6. Save Student
    out_dir = Path("models/student_models/distilled")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(student_model, out_dir / "distilled_xgboost.pkl")
    joblib.dump(vectorizer, out_dir / "student_vectorizer.pkl")
    logger.info(f"Distilled model saved to {out_dir}")

if __name__ == "__main__":
    main()
    