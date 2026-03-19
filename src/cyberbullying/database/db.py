from pymongo import MongoClient
from datetime import datetime, timezone

# ------------------------------------------------
# 🔹 MongoDB Connection
# ------------------------------------------------

client = MongoClient("mongodb://localhost:27017/")

db = client["cyberbullying_db"]
collection = db["predictions"]


# ------------------------------------------------
# 🔹 Save Prediction
# ------------------------------------------------

def save_prediction(text, result, platform="manual", content_type="text"):

    document = {
        "text": text,
        "platform": platform,
        "content_type": content_type,

        "label": result.get("label"),
        "severity": result.get("severity"),
        "confidence": result.get("confidence"),

        "components": result.get("components"),
        "emotions": result.get("emotions"),
        "explanation": result.get("explanation"),

        "timestamp": datetime.now(timezone.utc)
    }
    print("INSERTING:", document)
    collection.insert_one(document)


# ------------------------------------------------
# 🔹 Get History
# ------------------------------------------------

def get_history(limit=50, platform=None, severity=None):

    query = {}

    if platform:
        query["platform"] = platform

    if severity:
        query["severity"] = severity

    results = list(
        collection
        .find(query, {"_id": 0})
        .sort("timestamp", -1)
        .limit(limit)
    )

    return results