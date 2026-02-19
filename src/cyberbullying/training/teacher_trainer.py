# src/model_training/teacher_trainer.py
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback
)
from datasets import Dataset, load_metric
import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class TeacherModelTrainer:
    def __init__(self, model_name, output_dir):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=2
        )
        
        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    def load_data(self, data_path):
        """Load training data"""
        df = pd.read_csv(data_path)
        
        # Convert to dataset format
        dataset = Dataset.from_pandas(df)
        
        # Tokenize
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"], 
                padding="max_length", 
                truncation=True,
                max_length=128
            )
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        return tokenized_dataset
    
    def train(self, train_dataset, val_dataset, training_args):
        """Train the teacher model"""
        
        # Define compute_metrics function
        def compute_metrics(eval_pred):
            metric = load_metric("f1")
            logits, labels = eval_pred
            predictions = logits.argmax(axis=-1)
            return metric.compute(predictions=predictions, references=labels)
        
        # Training arguments
        args = TrainingArguments(
            output_dir=str(self.output_dir),
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=training_args.get("learning_rate", 2e-5),
            per_device_train_batch_size=training_args.get("batch_size", 16),
            per_device_eval_batch_size=training_args.get("batch_size", 16),
            num_train_epochs=training_args.get("epochs", 10),
            weight_decay=training_args.get("weight_decay", 0.01),
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            logging_dir=str(self.output_dir / "logs"),
            report_to="none"
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Train
        logger.info(f"Training {self.model_name}...")
        trainer.train()
        
        # Save model
        trainer.save_model(str(self.output_dir / "final_model"))
        self.tokenizer.save_pretrained(str(self.output_dir / "final_model"))
        
        logger.info(f"Model saved to {self.output_dir / 'final_model'}")
        
        return trainer

def train_all_teachers(config):
    """Train all teacher models"""
    teacher_models = config["teacher_models"]
    
    for model_name, model_config in teacher_models.items():
        if model_config.get("fine_tune", False):
            logger.info(f"Fine-tuning {model_name}...")
            
            trainer = TeacherModelTrainer(
                model_name=model_config["model_name"],
                output_dir=f"models/teacher_models/{model_name}"
            )
            
            # Load data (you need to implement data loading)
            train_data = trainer.load_data("data/processed/train_data.csv")
            val_data = trainer.load_data("data/processed/val_data.csv")
            
            # Train
            trainer.train(train_data, val_data, config["training"])