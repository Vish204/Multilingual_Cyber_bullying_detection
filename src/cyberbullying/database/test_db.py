from cyberbullying.database.db import save_prediction, get_history

dummy_result = {
    "label": "cyberbullying",
    "severity": "severe",
    "confidence": 0.9,
    "components": {"cyberbullying": 0.9},
    "emotions": [{"label": "aggression", "score": 0.8}],
    "explanation": {"trigger_words": [{"word": "idiot"}]}
}

print("Saving test data...")
save_prediction("you are an idiot", dummy_result)

print("\nFetching history...")
data = get_history()

print(data)