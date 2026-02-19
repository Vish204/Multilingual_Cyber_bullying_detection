# explore_database.py
import json
from pathlib import Path

def explore_database():
    with open("config/complete_multilingual_database.json", 'r') as f:
        data = json.load(f)
    
    print("=" * 60)
    print("DATABASE STRUCTURE EXPLORATION")
    print("=" * 60)
    
    # Check metadata
    if 'metadata' in data:
        print(f"\n1. METADATA:")
        for key, value in data['metadata'].items():
            print(f"   {key}: {value}")
    
    # Check languages
    if 'languages' in data:
        print(f"\n2. LANGUAGES ({len(data['languages'])}):")
        for lang in data['languages']:
            print(f"   - {lang}")
    
    # Check categories
    if 'categories' in data:
        print(f"\n3. CATEGORIES:")
        for category, cat_data in data['categories'].items():
            print(f"\n   {category}:")
            if isinstance(cat_data, dict):
                for key, value in cat_data.items():
                    if isinstance(value, list):
                        print(f"     {key}: {len(value)} items")
                        if len(value) > 0:
                            print(f"       Sample: {value[:3]}")
                    elif isinstance(value, dict):
                        print(f"     {key}: dictionary with {len(value)} keys")
                    else:
                        print(f"     {key}: {type(value)}")
    
    # Check if there's actual keyword data
    print(f"\n4. SEARCHING FOR KEYWORD DATA...")
    
    def find_keywords(obj, path="", depth=0):
        if depth > 3:  # Limit recursion depth
            return
        
        if isinstance(obj, list):
            if len(obj) > 0 and isinstance(obj[0], str):
                print(f"   Found {len(obj)} keywords at path: {path}")
                print(f"     Sample: {obj[:5]}")
                return True
        elif isinstance(obj, dict):
            for key, value in obj.items():
                new_path = f"{path}.{key}" if path else key
                find_keywords(value, new_path, depth + 1)
    
    find_keywords(data)

if __name__ == "__main__":
    explore_database()