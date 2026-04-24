# src/data_pipeline/02_validate_database/check_keywords.py
import json
from pathlib import Path
import pandas as pd

print("=" * 60)
print("KEYWORD DATABASE DIAGNOSTIC")
print("=" * 60)

# 🔹 SEM8 PATH UPDATE 🔹
project_root = Path(__file__).resolve().parents[3]
config_dir = project_root / "resources" / "keywords"
keywords_dir = config_dir / "multilingual_keywords"
complete_db = config_dir / "complete_multilingual_database.json"
data_dir = project_root / "data"

print(f"\n1. Keywords directory exists: {config_dir.exists()}")
if config_dir.exists():
    print(f"   Contents: {[f.name for f in config_dir.iterdir() if f.is_file()]}")

print(f"\n2. Multilingual keywords directory exists: {keywords_dir.exists()}")

if keywords_dir.exists():
    keyword_files = list(keywords_dir.glob("*.json"))
    print(f"   Found {len(keyword_files)} keyword files")
    
    for keyword_file in keyword_files:
        try:
            with open(keyword_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"   {keyword_file.name}: {len(data) if isinstance(data, list) else 'dict'}")
        except Exception as e:
            print(f"   {keyword_file.name}: ERROR - {e}")

print(f"\n3. Complete database exists: {complete_db.exists()}")

if complete_db.exists():
    try:
        with open(complete_db, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"   File size: {complete_db.stat().st_size / 1024:.1f} KB")
        
        if "complete_cyberbullying_database" in data:
            db = data["complete_cyberbullying_database"]
            print(f"   Found 'complete_cyberbullying_database' with {len(db)} categories")
            
            print("\n   Categories and sample items:")
            for category, items in db.items():
                if isinstance(items, list):
                    print(f"     {category}: {len(items)} items")
                    print(f"       Sample: {items[:3]}")
                elif isinstance(items, dict):
                    print(f"     {category}: dictionary with {len(items)} languages")
                    for lang, lang_items in list(items.items())[:3]:  # Show first 3 languages
                        print(f"       {lang}: {len(lang_items) if isinstance(lang_items, list) else '?'} items")
                else:
                    print(f"     {category}: {type(items)}")
        
        else:
            print("   ERROR: 'complete_cyberbullying_database' key not found! Check structure.")
            print(f"   Available keys: {list(data.keys())}")
            
    except Exception as e:
        print(f"   ERROR reading file: {e}")

print(f"\n4. Checking data directory:")
if data_dir.exists():
    print(f"   data/ exists")
    print(f"   Contents: {[f.name for f in data_dir.iterdir()]}")

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)