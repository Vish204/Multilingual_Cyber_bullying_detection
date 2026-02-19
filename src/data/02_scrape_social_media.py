# /cyberbullying-detection/start_data_collection.py

import sys
import os
import json
import math
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from data_collection import twitter_collector, reddit_collector, youtube_collector
from config import API_KEYS, TARGET_LANGUAGES

TARGET_PLATFORMS = {
    'reddit': reddit_collector,
     'youtube': youtube_collector,
      'twitter':twitter_collector
    
    
}
PROGRESS_FILE = 'collection_progress.json'
BATCH_SIZE = 10 # Process 10 keywords at a time

def load_progress():
    """Loads the progress file."""
    if os.path.exists(PROGRESS_FILE):
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
            keyword_file = os.path.join('config', 'consolidated_keywords.json')
            with open(keyword_file, 'r', encoding='utf-8') as f:
                all_keyword_data = json.load(f)
        except FileNotFoundError:
            print(f"❌ Consolidated keyword file not found. Skipping platform."); break
            
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