# debug_hindi_keywords.py
import json
import re

# Load Hindi keywords
with open("config/complete_multilingual_database.json", 'r') as f:
    data = json.load(f)

hindi_keywords = data['languages']['hindi']['keywords']

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