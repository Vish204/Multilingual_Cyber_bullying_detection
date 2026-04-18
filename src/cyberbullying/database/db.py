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
                "total": {"$sum": 1},
                # Conditionally count only the posts flagged as cyberbullying
                "flagged": {
                    "$sum": {
                        "$cond": [{"$eq": ["$prediction.label", "cyberbullying"]}, 1, 0]
                    }
                }
            }
        }
    ]
    results = list(collection.aggregate(pipeline))
    
    # Return structure: { "reddit": {"total": 120, "flagged": 45}, "youtube": {...} }
    return {
        item["_id"]: {
            "total": item["total"], 
            "flagged": item["flagged"]
        } 
        for item in results if item["_id"]
    }

# ------------------------------------------------
# Trends (Time-based)
# ------------------------------------------------


def get_trend_stats():
    # 1. Calculate the exact cutoff date (15 days ago from right now)
    cutoff_date = datetime.utcnow() - timedelta(days=15)

    pipeline = [
        # 2. FILTER FIRST: Only grab records newer than 15 days ago
        {
            "$match": {
                "created_at": {"$gte": cutoff_date} 
            }
        },
        # 3. GROUP & COUNT: Same logic as before, but now only on the filtered data
        {
            "$group": {
                "_id": {
                    "$dateToString": {
                        "format": "%Y-%m-%d",
                        "date": "$created_at" 
                    }
                },
                "total": {"$sum": 1},
                "flagged": {
                    "$sum": {
                        "$cond": [{"$eq": ["$prediction.label", "cyberbullying"]}, 1, 0]
                    }
                }
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


# ------------------------------------------------
# PHASE 2:ANALYTICS PAGE  UNIFIED ANALYTICS AGGREGATOR
# ------------------------------------------------

def get_analytics_overview():
    fifteen_days_ago = datetime.now(IST) - timedelta(days=15)
    
    # 2. Update this from {} to use the date filter
    total_analyzed = collection.count_documents({
        "created_at": {"$gte": fifteen_days_ago}
    })
    # 1. Base Stats
    # total_analyzed = collection.count_documents({})
    
    # 2. Severity (Unreviewed only - for 'Current Threat' view)
    sev_pipeline = [
        {"$match": {"flags.reviewed": False}},
        {"$group": {"_id": "$prediction.severity", "count": {"$sum": 1}}}
    ]
    severity_data = {item["_id"]: item["count"] for item in collection.aggregate(sev_pipeline) if item["_id"]}

    # 3. Platform Distribution
    platform_data = get_platform_stats() # Reusing your existing function

    # 4. System Trust Levels (Confidence Bins)
    # Mapping confidence (0-100) to Moderator-friendly labels
    trust_pipeline = [
        {
            "$project": {
                "level": {
                    "$cond": [
                        {"$gte": ["$prediction.confidence", 80]}, "High Trust",
                        {"$cond": [{"$gte": ["$prediction.confidence", 50]}, "Needs Review", "Requires Attention"]}
                    ]
                }
            }
        },
        {"$group": {"_id": "$level", "count": {"$sum": 1}}}
    ]
    trust_levels = {item["_id"]: item["count"] for item in collection.aggregate(trust_pipeline)}

    # 5. Language Distribution (Cleaned and Full Array)
    lang_raw = get_language_distribution() 
    
    # 🔥 THE ASSASSIN: Remove 'unknown', 'none', or empty strings before sorting
    clean_langs = {
        k: v for k, v in lang_raw.items() 
        if k and str(k).strip().lower() not in ["unknown", "none", ""]
    }
    
    # Sort the clean data highest to lowest
    sorted_langs = sorted(clean_langs.items(), key=lambda x: x[1], reverse=True)
    
    # Format as a complete list of dictionaries
    all_languages = [{"language": k.capitalize(), "count": v} for k, v in sorted_langs]


    # 6. AI vs Moderator Alignment (20-post threshold)
    reviewed_count = collection.count_documents({"flags.reviewed": True})
    alignment = None
    if reviewed_count >= 20:
        agreed = collection.count_documents({
            "flags.reviewed": True,
            "$or": [
                {"prediction.label": "cyberbullying", "moderator.action": {"$in": ["delete", "report"]}},
                {"prediction.label": "non-cyberbullying", "moderator.action": "ignore"}
            ]
        })
        alignment = {
            "agreed": agreed,
            "reevaluated": reviewed_count - agreed,
            "accuracy_rate": round((agreed / reviewed_count) * 100, 1)
        }

    # 7. Trends (Last 15 days)
    trends = get_trend_stats() # Reusing your existing function

    latency_pipeline = [
        {"$sort": {"created_at": -1}},
        {"$limit": 50},
        {"$group": {
            "_id": None,
            "avg_latency": {"$avg": "$latency.total_ms"}
        }}
    ]
    latency_res = list(collection.aggregate(latency_pipeline))
    avg_latency_ms = round(latency_res[0]["avg_latency"], 1) if latency_res else 0

    return {
        "total_analyzed_posts": total_analyzed,
        "severity": severity_data,
        "platforms": platform_data,
        "trust_levels": trust_levels,
        "languages": all_languages,
        "alignment": alignment,
        "trends": trends,
        "system_latency_ms": avg_latency_ms
    }

# ------------------------------------------------
#  PHASE 3:HISTORY
# ------------------------------------------------
def get_audit_history():
    # 1. Fetch the latest 100 reviewed posts, sorted newest first
    cursor = collection.find({"flags.reviewed": True}).sort("created_at", -1).limit(100)
    history_data = []

    for post in cursor:
        # 2. Extract values safely
        prediction = post.get("prediction", {})
        ai_severity = prediction.get("severity", "none").lower()
        
        moderator = post.get("moderator", {})
        mod_action = moderator.get("action", "ignored").lower()

        # 3. Calculate Semantic Alignment
        alignment_status = "Unknown"
        
        # Rule 1: True Positive (AI saw threat, Human agreed)
        if ai_severity in ["severe", "moderate"] and mod_action in ["delete", "report"]:
            alignment_status = "Agreed"
        # Rule 2: True Negative (AI saw nothing, Human agreed)
        elif ai_severity in ["none", "mild"] and mod_action == "ignore":
            alignment_status = "Agreed"
        # Rule 3: False Positive (AI too aggressive, Human overruled)
        elif ai_severity in ["severe", "moderate"] and mod_action == "ignore":
            alignment_status = "Overruled"
        # Rule 4: False Negative (AI missed it, Human overruled)
        elif ai_severity in ["none", "mild"] and mod_action in ["delete", "report"]:
            alignment_status = "Overruled"
        elif mod_action == "pending":
            alignment_status = "Pending Review"

        # 4. Build the clean, flat payload for the React Table
        history_data.append({
            "id": str(post.get("_id")),
            "timestamp": post.get("created_at"), 
            "platform": post.get("platform", "Unknown"),
            "text": post.get("text", ""),
            "ai_severity": ai_severity,
            "ai_confidence": prediction.get("confidence", 0),
            "moderator_action": mod_action,
            "alignment_status": alignment_status
        })

    return history_data