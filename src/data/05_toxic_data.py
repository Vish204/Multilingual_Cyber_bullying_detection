import pandas as pd
import random
import os
from pathlib import Path


random.seed(42)
# -----------------------
# TOXIC TEMPLATES
# -----------------------

toxic_templates_en = [
    "You're worthless, just die.",
    "Nobody cares about you.",
    "Go kill yourself.",
    "The world is better without you.",
    "End your life already.",
    "You are a complete failure.",
    "You deserve to suffer.",
]

toxic_templates_hi = [
    "Tu nikamma hai, mar ja.",
    "Tere jaise log marne chahiye.",
    "Kisi ko fark nahi padta tumse.",
    "Tu duniya par bojh hai.",
]

toxic_templates_ta = [
    "Nee oru waste, sethu po.",
    "Yaarum unna venam.",
    "Po da setha.",
]

# -----------------------
# NEUTRAL COUNTER
# -----------------------

neutral_templates_en = [
    "This game is killer.",
    "The battery died yesterday.",
    "I almost died laughing.",
    "That performance killed it.",
    "The server died again.",
]

neutral_templates_hi = [
    "Battery mar gayi.",
    "Phone dead ho gaya.",
    "Match killer tha.",
]

neutral_templates_ta = [
    "Phone sethupochu.",
    "Adhu killer performance.",
]

languages = ["english", "hindi", "tamil"]
platforms = ["synthetic"]

data = []

# Generate 300 toxic
for _ in range(300):
    lang = random.choice(languages)
    if lang == "english":
        text = random.choice(toxic_templates_en)
    elif lang == "hindi":
        text = random.choice(toxic_templates_hi)
    else:
        text = random.choice(toxic_templates_ta)

    data.append({
        "text": text,
        "label": 1,
        "language": lang,
        "platform": "synthetic"
    })

# Generate 150 neutral counter
for _ in range(160):
    lang = random.choice(languages)
    if lang == "english":
        text = random.choice(neutral_templates_en)
    elif lang == "hindi":
        text = random.choice(neutral_templates_hi)
    else:
        text = random.choice(neutral_templates_ta)

    data.append({
        "text": text,
        "label": 0,
        "language": lang,
        "platform": "synthetic"
    })

os.makedirs("data/raw/hard_toxic", exist_ok=True)

df = pd.DataFrame(data)

BASE_DIR = Path(__file__).resolve().parents[2]

save_path = BASE_DIR / "data" / "raw" / "hard_toxic"
save_path.mkdir(parents=True, exist_ok=True)

df.to_csv(save_path / "severe_bullying.csv", index=False, encoding="utf-8")

print("✅ Generated 450 samples (300 toxic + 150 neutral)")
print(df["label"].value_counts())
