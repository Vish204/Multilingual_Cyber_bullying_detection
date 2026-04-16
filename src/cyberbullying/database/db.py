from pymongo import MongoClient
from datetime import datetime, timezone
from bson import ObjectId
from datetime import datetime, timedelta
import hashlib
import pytz


IST = pytz.timezone('Asia/Kolkata')

# ------------------------------------------------
# 🔹 MongoDB Connection
# ------------------------------------------------

client = MongoClient("mongodb://localhost:27017/")

db = client["cyberbullying_db"]
collection = db["predictions"]

training_collection = db["curated_data"] # The Human-in-the-Loop ML dataset

# 🔥 Create TTL Index (Delete after 45 days / 3888000 seconds)
# partialFilterExpression ensures we NEVER delete posts where flags.saved == True
try:
    collection.create_index(
        "created_at",
        expireAfterSeconds=3888000,
        partialFilterExpression={"flags.saved": False}
    )
    print("Database indices verified.")
except Exception as e:
    print(f"Index creation skipped/failed: {e}")

# ------------------------------------------------
# 🔹 Save Prediction
# ------------------------------------------------

# ------------------------------------------------
# 🔹 Save Prediction (V2 Schema)
# ------------------------------------------------

def save_prediction(text, result, platform="manual", content_type="text", platform_time=None, latency_data=None, platform_post_id=None):
    
    # Defaults for direct API calls
    if latency_data is None: latency_data = {}
    if platform_time is None: platform_time = datetime.now(timezone.utc)

    # 🔥 Generate the hash
    text_hash = hashlib.md5(text.strip().lower().encode('utf-8')).hexdigest()

    document = {
        "text": text,
        "text_hash": text_hash,
        "platform": platform,
        "platform_post_id": platform_post_id,
        "content_type": content_type,
        
        # Timestamps
        "created_at": datetime.now(IST),
        "platform_time": platform_time,

        # Core ML Output
        "prediction": {
            "label": result.get("label"),
            "severity": result.get("severity"),
            "confidence": result.get("confidence")
        },

        # Explainability & Context
        "signals": {
            "sarcasm": result.get("sarcasm"),
            "emotions": result.get("emotions", []),      # Clean array: [{"label": "anger", "score": 0.85}]
            "explanation": result.get("explanation"),
            "components": {
                "base_cyberbullying": result.get("components", {}).get("cyberbullying"),
            }   # The calibrated base scores
        },

        # System & Review Flags
        "flags": {
            "alert": result.get("alert", False),
            "reviewed": False,
            "saved": False,                              # The flag for ML retraining
            "language": result.get("language")
        },

        # Human-in-the-loop action
        "moderator": {
            "action": None,
            "reason": None
        },

        # System tracking
        "latency": latency_data
    }
    
    print("INSERTING:", document)

    # 🔥 REPLACE collection.insert_one(document) WITH THIS:
    collection.update_one(
        {"text_hash": text_hash}, # Check if this exact text exists
        {"$set": document},       # If yes, just update the stats. If no, insert it.
        upsert=True
    )
    print(f"UPSERTED: {platform.upper()} Post")


# ------------------------------------------------
# 🔹 Get History
# ------------------------------------------------


def get_history(
    limit=50,
    platform=None,
    label=None,
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

    if label:
        query["prediction.label"] = label

    if severity:
        query["prediction.severity"] = severity

    if reviewed is not None:
        query["flags.reviewed"] = reviewed

    if alert is not None:
        query["flags.alert"] = alert

    if moderator_action:
        query["moderator.action"] = moderator_action

    if language:
        query["flags.language.name"] = language

    if content_type:
        query["content_type"] = content_type

    # ------------------------
    # Date filter (IMPORTANT)
    # ------------------------

    if start_date or end_date:
        query["created_at"] = {}

        if start_date:
            query["created_at"]["$gte"] = datetime.fromisoformat(start_date)

        if end_date:
            query["created_at"]["$lte"] = datetime.fromisoformat(end_date)

    # ------------------------
    # Execute query
    # ------------------------

    results = list(
        collection
        .find(query)
        .sort("created_at", -1)
        .limit(limit)
    )

    # Convert ObjectId -> string AND flatten the nested data for the frontend
    formatted_results = []
    
    for item in results:

        # 🔥 Grab the Mongo UTC time and forcefully format it as a beautiful IST string
        raw_time = item.get("created_at")
        if raw_time:
            utc_time = raw_time.replace(tzinfo=timezone.utc) if raw_time.tzinfo is None else raw_time
            ist_string = utc_time.astimezone(IST).isoformat()  # Looks like: '2026-04-15T00:25:48+05:30'
        else:
            ist_string = None

        flat_item = {
            "id": str(item.pop("_id")),
            "text": item.get("text"),
            "platform": item.get("platform"),
            "platform_post_id": item.get("platform_post_id", "N/A"),

            "timestamp": ist_string, # Map created_at back to timestamp for UI

            "content_type": item.get("content_type", "text"),
            
            # Flattened ML Data
            "label": item.get("prediction", {}).get("label"),
            "severity": item.get("prediction", {}).get("severity"),
            "confidence": item.get("prediction", {}).get("confidence"),
            "sarcasm": item.get("signals", {}).get("sarcasm"),

            "emotions": item.get("signals", {}).get("emotions", []),
            
            # Flattened Flags
            "alert": item.get("flags", {}).get("alert"),
            "reviewed": item.get("flags", {}).get("reviewed"),
            "saved": item.get("flags", {}).get("saved"),

            "language": item.get("flags", {}).get("language", {}).get("name", "unknown"),

            "explanation": item.get("signals", {}).get("explanation"),
            
            # Flattened Moderator Actions
            "moderator_action": item.get("moderator", {}).get("action"),
            "moderator_reason": item.get("moderator", {}).get("reason", ""),

            "latency": item.get("latency", {})
        }
        formatted_results.append(flat_item)

    return formatted_results

# ------------------------------------------------
# 🔹 Update Moderator Action
# ------------------------------------------------

# ------------------------------------------------
# 🔹 Update Moderator Action & Human-in-the-Loop
# ------------------------------------------------

def update_moderation_action(record_id, action, reason="", saved=False):

    # 1. Update the main collection flags
    update_fields = {
        "flags.reviewed": True,
        "flags.saved": saved,
        "moderator.action": action,
        "moderator.reason": reason
    }
    
    result = collection.update_one(
        {"_id": ObjectId(record_id)},
        {"$set": update_fields}
    )

    # 2. 🔥 The Human-in-the-Loop feature (Data Curation)
    # If the moderator wants to save it for retraining, move it to the curated collection.
    if saved:
        doc = collection.find_one({"_id": ObjectId(record_id)})
        # if doc and doc["prediction"]["label"] == "cyberbullying":
        #     training_collection.replace_one({"_id": doc["_id"]}, doc, upsert=True)

        if doc:
            # 🔥 NEW: If the AI missed it (False Negative), but the human deleted/reported it,
            # we explicitly tag it as a human-corrected label so the AI learns next time!
            if action in ["delete", "report"] and doc["prediction"]["label"] == "non-cyberbullying":
                doc["human_corrected_label"] = "cyberbullying"

            # Save to the curated dataset
            training_collection.replace_one({"_id": doc["_id"]}, doc, upsert=True)
        else:
            # If the moderator un-saves it, we remove it from the training pool
            training_collection.delete_one({"_id": ObjectId(record_id)})

    return result.modified_count


### analytics functions below ###
# ------------------------------------------------
# Severity Analytics
# ------------------------------------------------
def get_severity_stats():
    pipeline = [
        {
            "$group": {
                "_id": "$prediction.severity",  # 🔥 FIXED NESTED PATH
                "count": {"$sum": 1}
            }
        }
    ]
    results = list(collection.aggregate(pipeline))
    return {item["_id"]: item["count"] for item in results if item["_id"]}



# ------------------------------------------------
# dashboard page summary (with 15-day window)
# ------------------------------------------------
def get_dashboard_summary():
    # 1. Define the 15-day window for metrics
    fifteen_days_ago = datetime.now(IST) - timedelta(days=15)
    
    # 2. Total Posts (Last 15 Days)
    total_recent = collection.count_documents({"created_at": {"$gte": fifteen_days_ago}})
    
    # 3. Bullying Posts (Last 15 Days) - Moderate + Severe
    bullying_recent = collection.count_documents({
        "created_at": {"$gte": fifteen_days_ago},
        "prediction.severity": {"$in": ["moderate", "severe"]}
    })
    
    # 4. Pending High Priority (All time - because work shouldn't expire)
    # Severe/Moderate posts that haven't been reviewed
    pending_priority = collection.count_documents({
        "flags.reviewed": False,
        "prediction.severity": {"$in": ["moderate", "severe"]}
    })

    # 5. Average Latency (Last 100 Posts)
    # We look at the 'latency.total_ms' field
    latency_pipeline = [
        {"$sort": {"created_at": -1}},
        {"$limit": 50},
        {"$group": {
            "_id": None,
            "avg_latency": {"$avg": "$latency.total_ms"}
        }}
    ]
    latency_res = list(collection.aggregate(latency_pipeline))
    avg_ms = round(latency_res[0]["avg_latency"], 1) if latency_res else 0

    return {
        "total_posts": total_recent,
        "bullying_percentage": round((bullying_recent / total_recent * 100), 1) if total_recent > 0 else 0,
        "pending_priority": pending_priority,
        "avg_latency_ms": avg_ms
    }

# ------------------------------------------------
# Platform Analytics (Unchanged)
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
    return {item["_id"]: item["count"] for item in results if item["_id"]}

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
                        "date": "$created_at"  # 🔥 FIXED FROM "timestamp"
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
# Language Analytics
# ------------------------------------------------
def get_language_distribution():
    pipeline = [
        {
            "$group": {
                "_id": "$flags.language.name",  # 🔥 FIXED NESTED PATH
                "count": {"$sum": 1}
            }
        }
    ]
    results = list(collection.aggregate(pipeline))
    return {
        item["_id"] if item["_id"] else "unknown": item["count"]
        for item in results
    }



# # ------------------------------------------------
# # Export Data (for CSV/Excel)
# # ------------------------------------------------
# def export_data(limit, platform=None, severity=None, reviewed=None,
#                 label=None, alert=None, language=None, content_type=None):

#     query = {}

#     if platform:
#         query["platform"] = platform

#     if severity:
#         query["severity"] = severity

#     if reviewed is not None:
#         if reviewed == False:
#             query["$or"] = [
#                 {"reviewed": False},
#                 {"reviewed": {"$exists": False}}
#             ]
#         else:
#             query["reviewed"] = True

#     if label:
#         query["label"] = label

#     if alert is not None:
#         query["alert"] = alert

#     if language:
#         query["language"] = language

#     if content_type:
#         query["content_type"] = content_type

#     results = list(
#         collection.find(query, {"_id": 0})
#         .sort("timestamp", -1)
#         .limit(limit)
#     )

#     return results