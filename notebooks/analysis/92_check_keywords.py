# check_individual_keywords.py
import json
from pathlib import Path

def check_individual_files():
    keywords_dir = Path("config/multilingual_keywords")
    
    print("=" * 60)
    print("INDIVIDUAL LANGUAGE KEYWORD FILES")
    print("=" * 60)
    
    for keyword_file in keywords_dir.glob("*.json"):
        try:
            with open(keyword_file, 'r') as f:
                data = json.load(f)
            
            print(f"\n{keyword_file.name}:")
            if isinstance(data, dict):
                print(f"  Type: Dictionary with {len(data)} keys")
                for key, value in data.items():
                    if isinstance(value, list):
                        print(f"    {key}: {len(value)} items")
                        if len(value) > 0:
                            print(f"      Sample: {value[:3]}")
            elif isinstance(data, list):
                print(f"  Type: List with {len(data)} items")
                if len(data) > 0:
                    print(f"    Sample: {data[:5]}")
            else:
                print(f"  Type: {type(data)}")
                
        except Exception as e:
            print(f"{keyword_file.name}: ERROR - {e}")

if __name__ == "__main__":
    check_individual_files()