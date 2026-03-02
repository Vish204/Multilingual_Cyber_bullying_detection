import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "models/teacher/mbert/final_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

model.eval()

# texts = [
#     "You are such an idiot, nobody likes you",
#     "I hope you have a great day 😊",
#     "Go kill yourself, you are useless",
#     "This movie was amazing, loved it!",
#     "You're the worst person ever",
#     "Kill yourself.",
#     "I love this movie.",
#     "Have a nice day.",
#     "You are useless.",
#     "tu duniya par bojh hai"
# ]

texts = [
    # Positive / Neutral
    "I hope you succeed in life.",
    "You did a great job.",
    "That was very kind of you.",
    "Have a wonderful evening.",
    "I appreciate your help.",
    "This food tastes amazing.",
    "Let’s work together on this.",
    "You are very talented.",
    "Keep improving every day.",
    "Wishing you good luck.",

    # Mild Negative (Non-toxic)
    "I disagree with your opinion.",
    "That idea doesn’t make sense to me.",
    "I think you are wrong.",
    "This movie was boring.",
    "That performance was disappointing.",
    "I expected better from you.",
    "This product is not good.",
    "I don’t like this design.",
    "You need to improve.",
    "That was a bad decision.",

    # Clear Toxic
    "You are completely useless.",
    "Nobody cares about you.",
    "You are a waste of space.",
    "Go die.",
    "I hate you so much.",
    "You are disgusting.",
    "Shut up, idiot.",
    "You are the worst human ever.",
    "Kill yourself.",
    "I will ruin your life.",

    # Violent Threat
    "I will kill you.",
    "I’ll beat you up.",
    "I will destroy you.",
    "I swear I will hurt you.",
    "You deserve to suffer.",

    # Hindi Toxic
    "Tu bilkul bekaar hai.",
    "Tu duniya par bojh hai.",
    "Main tujhe maar dunga.",
    "Tera koi future nahi hai.",
    "Chup ho ja bewakoof.",

    # Hindi Non-toxic
    "Aap ka din shubh ho.",
    "Mujhe aapse baat karke accha laga.",
    "Aap bahut mehnati ho.",
    "Aapka kaam bahut acha hai.",
    "Aap safal honge.",

    # Sarcasm / Edge cases
    "Wow, genius move.",
    "Great job ruining everything.",
    "You’re smart, aren’t you?",
    "Nice work, idiot.",
    "Brilliant, just brilliant.",
    "You are amazing.",
"You are wonderful.",
"You are terrible.",
"You are disgusting.",
"You are brilliant.",
"You are a genius.",
"You idiot.",
"You fool.",
"You legend.",
"You hero."

]



inputs = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors="pt"
)

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=1)
    predictions = torch.argmax(probs, dim=1)

id2label = model.config.id2label

for text, pred, prob in zip(texts, predictions, probs):
    label = id2label[pred.item()]
    confidence = prob[pred.item()].item()

    print(f"\nText: {text}")
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.4f}")
    
print(model.config.id2label)
print(model.config.label2id)
