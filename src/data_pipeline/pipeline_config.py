# src/data_pipeline/pipeline_config.py

"""
Configuration settings for the Phase 1 Historical Data Pipeline.
Note: All API keys must be securely loaded from the root .env file.
"""

# --- COLLECTION SETTINGS ---
# This list controls which languages the master script will process.
TARGET_LANGUAGES = [
    'english', 'hinglish', 'hindi', 'bengali', 'tamil', 'telugu', 'marathi', 
    'gujarati', 'kannada', 'malayalam', 'punjabi', 'oriya', 'urdu', 'sanskrit'
]

# Number of keywords to process per batch during data collection
BATCH_SIZE = 10