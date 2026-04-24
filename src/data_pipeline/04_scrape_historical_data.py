#!/usr/bin/env python3
# src/data_pipeline/04_scrape_historical_data.py

import sys
import os
import json
import math
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# 🔹 SEM8 PATH & ENV SETUP 🔹
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

# Add 'src' to Python path so we can import the legacy collectors
sys.path.append(str(PROJECT_ROOT / "src"))

from cyberbullying.data_collection import twitter_collector, reddit_collector, youtube_collector
from pipeline_config import TARGET_LANGUAGES, BATCH_SIZE

# Safely build the API_KEYS dictionary from the .env file
API_KEYS = {
    "twitter": {
        "bearer_token": os.getenv("TWITTER_BEARER_TOKEN")
    },
    "youtube": {
        "api_key": os.getenv("YOUTUBE_API_KEY")
    },
    "reddit": {
        "client_id": os.getenv("REDDIT_CLIENT_ID"),
        "client_secret": os.getenv("REDDIT_CLIENT_SECRET"),
        "user_agent": os.getenv("REDDIT_USER_AGENT", "cyberbullying_research_v1.0")
    }
}

TARGET_PLATFORMS = {
    'reddit': reddit_collector,
    'youtube': youtube_collector,
    'twitter': twitter_collector
}

# Keep progress file in data/raw to avoid cluttering the src folder
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROGRESS_FILE = RAW_DIR / "collection_progress.json"

def load_progress():
    """Loads the progress file."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_progress(progress_data):
    """Saves the current progress."""
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump(progress_data, f, indent=2)

def main():
    """Runs data collection in a round-robin fashion, processing one batch per language at a time."""
    progress = load_progress()
    
    print("🚀🚀🚀 STARTING ROUND-ROBIN DATA COLLECTION 🚀🚀🚀")
    print(f"Languages: {', '.join(TARGET_LANGUAGES)}")
    print(f"Batch Size: {BATCH_SIZE} keywords")
    print("="*60)
    
    for platform_name, collector_module in TARGET_PLATFORMS.items():
        print(f"\n🔥🔥🔥 Starting Platform: {platform_name.upper()} 🔥🔥🔥")
        if platform_name not in progress:
            progress[platform_name] = {}

        # Determine the total number of batches needed
        max_batches = 0
        work_queue = {}
        try:
            keyword_file = PROJECT_ROOT / "resources" / "keywords" / "consolidated_keywords.json"
            with open(keyword_file, 'r', encoding='utf-8') as f:
                all_keyword_data = json.load(f)
        except FileNotFoundError:
            print(f"❌ Consolidated keyword file not found at {keyword_file}. Skipping platform.")
            break
            
        for language in TARGET_LANGUAGES:
            if language not in progress[platform_name]:
                progress[platform_name][language] = []
            
            all_keywords = all_keyword_data.get(language, [])
            completed_keywords = set(progress[platform_name][language])
            keywords_to_do = [kw for kw in all_keywords if kw not in completed_keywords]
            work_queue[language] = keywords_to_do
            
            num_batches = math.ceil(len(keywords_to_do) / BATCH_SIZE)
            if num_batches > max_batches:
                max_batches = num_batches
        
        if max_batches == 0:
            print(f"✅ All keywords for all languages on {platform_name.upper()} are already collected.")
            continue

        # Loop through BATCHES first, then languages
        for batch_num in range(max_batches):
            print(f"\n--- Processing Batch #{batch_num + 1} / {max_batches} for all languages ---")
            
            for language in TARGET_LANGUAGES:
                keywords_for_lang = work_queue.get(language, [])
                
                start_index = batch_num * BATCH_SIZE
                end_index = start_index + BATCH_SIZE
                
                # Check if this language has any keywords left for this batch
                if start_index >= len(keywords_for_lang):
                    continue
                
                batch_to_process = keywords_for_lang[start_index:end_index]
                
                print(f"   ↳ Processing {language.title()}...")
                
                try:
                    # Note: Legacy collectors execute assuming they are run from the project root.
                    # They will save output to 'data/raw/[platform]/[language]/data.csv'
                    collector_module.collect(language, batch_to_process, API_KEYS)
                    
                    # Update and save progress for this specific batch
                    progress[platform_name][language].extend(batch_to_process)
                    
                except Exception as e:
                    print(f"❌❌❌ CRITICAL ERROR for {platform_name} - {language}: {e} ❌❌❌")

            # Save progress after completing a full round-robin for one batch
            save_progress(progress)
            print(f"💾 Progress saved after completing Batch #{batch_num + 1}.")
    
    print("\n🎉🎉🎉 ALL DATA COLLECTION TASKS COMPLETE! 🎉🎉🎉")

if __name__ == "__main__":
    main()