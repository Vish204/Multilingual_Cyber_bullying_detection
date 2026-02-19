# src/model_training/student_trainer.py
import xgboost as xgb
import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.metrics import accuracy_score, f1_score, classification_report
from pathlib import Path

logger = logging.getLogger(__name__)

class StudentModelTrainer:
    def __init__(self, output_dir="models/student_models/xgboost"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
    
    def extract_features(self, texts):
        """Extract simple features for XGBoost"""
        features = []
        
        for text in texts:
            if not isinstance(text, str):
                text = str(text)
            
            # Basic text features
            text_len = len(text)
            word_count = len(text.split())
            char_count = len(text.replace(" ", ""))
            
            # Sentiment-like features
            negative_words = sum(1 for word in ['stupid', 'idiot', 'hate', 'kill', 'die'] 
                                if word in text.lower())
            uppercase_ratio = sum(1 for c in text if c.isupper()) / max(1, len(text))
            
            # Language indicators (simple)
            has_hindi = any(ord(c) > 127 for c in text)
            
            features.append([
                text_len, word_count, char_count, 
                negative_words, uppercase_ratio, has_hindi
            ])
        
        return np.array(features)
    
    def train(self, train_texts, train_labels, val_texts=None, val_labels=None, 
              params=None):
        """Train XGBoost model"""
        
        # Default parameters
        if params is None:
            params = {
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            }
        
        # Extract features
        logger.info("Extracting features...")
        X_train = self.extract_features(train_texts)
        y_train = np.array(train_labels)
        
        # Create DMatrix for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train)
        
        # Train model
        logger.info("Training XGBoost model...")
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=params.get('n_estimators', 100),
            evals=[(dtrain, 'train')],
            verbose_eval=False
        )
        
        # Save model
        model_path = self.output_dir / "xgboost_model.json"
        self.model.save_model(str(model_path))
        logger.info(f"Model saved to {model_path}")
        
        # Evaluate on training data
        train_preds = self.predict_proba(train_texts)
        train_preds_binary = (train_preds > 0.5).astype(int)
        train_acc = accuracy_score(y_train, train_preds_binary)
        train_f1 = f1_score(y_train, train_preds_binary)
        
        logger.info(f"Training Accuracy: {train_acc:.4f}")
        logger.info(f"Training F1 Score: {train_f1:.4f}")
        
        # Evaluate on validation data if provided
        if val_texts is not None and val_labels is not None:
            X_val = self.extract_features(val_texts)
            y_val = np.array(val_labels)
            dval = xgb.DMatrix(X_val)
            val_preds = self.model.predict(dval)
            val_preds_binary = (val_preds > 0.5).astype(int)
            
            val_acc = accuracy_score(y_val, val_preds_binary)
            val_f1 = f1_score(y_val, val_preds_binary)
            
            logger.info(f"Validation Accuracy: {val_acc:.4f}")
            logger.info(f"Validation F1 Score: {val_f1:.4f}")
        
        return self.model
    
    def predict_proba(self, texts):
        """Predict probabilities"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        X = self.extract_features(texts)
        dmatrix = xgb.DMatrix(X)
        predictions = self.model.predict(dmatrix)
        return predictions
    
    def predict(self, texts, threshold=0.5):
        """Predict binary labels"""
        probabilities = self.predict_proba(texts)
        return (probabilities > threshold).astype(int)

def train_xgboost_student(config):
    """Main function to train XGBoost student model"""
    logger.info("Starting XGBoost student model training...")
    
    # Load training data
    try:
        train_df = pd.read_csv("data/processed/train_data.csv")
        val_df = pd.read_csv("data/processed/val_data.csv")
        
        logger.info(f"Training samples: {len(train_df)}")
        logger.info(f"Validation samples: {len(val_df)}")
        
    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
        logger.info("Creating sample data for testing...")
        
        # Create sample data
        train_texts = [
            "You are stupid and worthless",
            "I hate you so much",
            "You should die",
            "Hello how are you",
            "Nice to meet you",
            "Good morning"
        ]
        train_labels = [1, 1, 1, 0, 0, 0]
        
        val_texts = [
            "You idiot",
            "Have a nice day"
        ]
        val_labels = [1, 0]
        
        trainer = StudentModelTrainer()
        trainer.train(train_texts, train_labels, val_texts, val_labels)
        return True
    
    # Extract texts and labels
    train_texts = train_df['text'].tolist()
    train_labels = train_df['label'].tolist()
    
    val_texts = val_df['text'].tolist()
    val_labels = val_df['label'].tolist()
    
    # Get XGBoost parameters from config
    xgb_params = config.get("xgboost_params", {})
    
    # Train model
    trainer = StudentModelTrainer()
    model = trainer.train(
        train_texts, train_labels, 
        val_texts, val_labels,
        params=xgb_params
    )
    
    logger.info("XGBoost student model training completed!")
    return True