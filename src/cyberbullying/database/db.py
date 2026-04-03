from pymongo import MongoClient
from datetime import datetime, timezone
from bson import ObjectId


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

        "emotion": result.get("emotion"),
        "sarcasm": result.get("sarcasm"),
        "language": result.get("language"),
        "alert": result.get("alert"),

        "components": result.get("components"),
        "emotions": result.get("emotions"),
        "explanation": result.get("explanation"),

        "moderator_action": None,
        "reviewed": False,

        "timestamp": datetime.now(timezone.utc)
    }
    print("INSERTING:", document)
    collection.insert_one(document)


# ------------------------------------------------
# 🔹 Get History
# ------------------------------------------------

from datetime import datetime

def get_history(
    limit=50,
    platform=None,
    severity=None,
    reviewed=None,
    alert=None,
    moderator_action=None,
    language=None,
    content_type=None,
    start_date=None,
    end_date=None
):

    query = {}

    # ------------------------
    # Basic filters
    # ------------------------

    if platform:
        query["platform"] = platform

    if severity:
        query["severity"] = severity

    if reviewed is not None:
        query["reviewed"] = reviewed

    if alert is not None:
        query["alert"] = alert

    if moderator_action:
        query["moderator_action"] = moderator_action

    if language:
        query["language.name"] = language

    if content_type:
        query["content_type"] = content_type

    # ------------------------
    # Date filter (IMPORTANT)
    # ------------------------

    if start_date or end_date:
        query["timestamp"] = {}

        if start_date:
            query["timestamp"]["$gte"] = datetime.fromisoformat(start_date)

        if end_date:
            query["timestamp"]["$lte"] = datetime.fromisoformat(end_date)

    # ------------------------
    # Execute query
    # ------------------------

    results = list(
        collection
        .find(query)
        .sort("timestamp", -1)
        .limit(limit)
    )

    # Convert ObjectId → string
    for item in results:
        item["_id"] = str(item["_id"])

    return results

# ------------------------------------------------
# 🔹 Update Moderator Action
# ------------------------------------------------

def update_moderation_action(record_id, action, reason=" "):

    result = collection.update_one(
        {"_id": ObjectId(record_id)},
        {
            "$set": {
                "moderator_action": action,
                "reviewed": True,
                "reason": reason
            }
        }
    )

    return result.modified_count

### analytics functions below ###


# ------------------------------------------------
# Severity Analytics
# ------------------------------------------------
def get_severity_stats():

    pipeline = [
        {
            "$group": {
                "_id": "$severity",
                "count": {"$sum": 1}
            }
        }
    ]

    results = list(collection.aggregate(pipeline))

    return {item["_id"]: item["count"] for item in results}

# ------------------------------------------------
# Platform Analytics
# ------------------------------------------------
def get_platform_stats():

    pipeline = [
        {
            "$group": {
                "_id": "$platform",
                "count": {"$sum": 1}
            }
        }
    ]

    results = list(collection.aggregate(pipeline))

    return {item["_id"]: item["count"] for item in results}

# ------------------------------------------------
# Trends (Time-based)
# ------------------------------------------------
def get_trend_stats():

    pipeline = [
        {
            "$group": {
                "_id": {
                    "$dateToString": {
                        "format": "%Y-%m-%d",
                        "date": "$timestamp"
                    }
                },
                "count": {"$sum": 1}
            }
        },
        {"$sort": {"_id": 1}}
    ]

    results = list(collection.aggregate(pipeline))

    return results

# ------------------------------------------------
#language analytics
# ------------------------------------------------
def get_language_distribution():

    pipeline = [
        {
            "$group": {
                "_id": "$language.name",
                "count": {"$sum": 1}
            }
        }
    ]

    results = list(collection.aggregate(pipeline))

    return {
        item["_id"] if item["_id"] else "unknown": item["count"]
        for item in results
    }



# ------------------------------------------------
# Export Data (for CSV/Excel)
# ------------------------------------------------
def export_data(limit, platform=None, severity=None, reviewed=None,
                label=None, alert=None, language=None, content_type=None):

    query = {}

    if platform:
        query["platform"] = platform

    if severity:
        query["severity"] = severity

    if reviewed is not None:
        if reviewed == False:
            query["$or"] = [
                {"reviewed": False},
                {"reviewed": {"$exists": False}}
            ]
        else:
            query["reviewed"] = True

    if label:
        query["label"] = label

    if alert is not None:
        query["alert"] = alert

    if language:
        query["language"] = language

    if content_type:
        query["content_type"] = content_type

    results = list(
        collection.find(query, {"_id": 0})
        .sort("timestamp", -1)
        .limit(limit)
    )

    return results