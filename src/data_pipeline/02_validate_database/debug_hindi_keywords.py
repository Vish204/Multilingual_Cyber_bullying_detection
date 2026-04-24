# src/data_pipeline/02_validate_database/debug_hindi_keywords.py
import json
import re
from pathlib import Path

# 🔹 SEM8 PATH UPDATE 🔹
project_root = Path(__file__).resolve().parents[3]
db_path = project_root / "resources" / "keywords" / "complete_multilingual_database.json"

if not db_path.exists():
    print(f"Error: {db_path} not found.")
    exit(1)

# Load Hindi keywords
with open(db_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Failsafe if structure varies slightly
if 'languages' in data and 'hindi' in data['languages']:
    hindi_keywords = data['languages']['hindi']['keywords']
else:
    print("Error: Could not find Hindi keywords in the expected JSON structure.")
    exit(1)

# Check for common Hindi bullying words
test_words = ["मूर्ख", "बेवकूफ", "गधा", "कमीना", "हरामी"]
print("Checking Hindi keywords for common bullying words:")
for word in test_words:
    matches = [kw for kw in hindi_keywords if word in kw]
    print(f"  '{word}': Found {len(matches)} matches")
    if matches:
        print(f"    Sample: {matches[:3]}")

# Check what Hindi keywords actually contain
print(f"\nTotal Hindi keywords: {len(hindi_keywords)}")
print("Sample Hindi keywords:")
for i, kw in enumerate(hindi_keywords[:10]):
    print(f"  {i+1}. {kw}")

# Test regex pattern
test_text = "तू मूर्ख है"
pattern = r'(?i)\b(' + '|'.join([re.escape(kw) for kw in hindi_keywords[:200]]) + r')\b'
matches = re.findall(pattern, test_text)
print(f"\nTesting '{test_text}' against Hindi pattern:")
print(f"  Matches: {matches}")