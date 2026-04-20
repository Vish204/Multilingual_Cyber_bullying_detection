import json
import os
import io
import csv

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import time

import re


from cyberbullying.inference.inference_service import predict_post
from cyberbullying.explainability.shap_explainer import explain_text

from cyberbullying.database.db import save_prediction, get_history
from cyberbullying.database.db import update_moderation_action
from cyberbullying.database.db import get_severity_stats
from cyberbullying.database.db import get_platform_stats
from cyberbullying.database.db import get_trend_stats
from cyberbullying.database.db import get_dashboard_summary
from cyberbullying.database.db import get_language_distribution
from cyberbullying.database.db import get_analytics_overview
from cyberbullying.database.db import get_audit_history

from cyberbullying.collector.run_collector import run_once


# ------------------------------------------------
# App init
# ------------------------------------------------

app = FastAPI()

# ------------------------------------------------
# CORS
# ------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ------------------------------------------------
# Request schema (VERY IMPORTANT)
# ------------------------------------------------

class TextRequest(BaseModel):
    text: str
    platform: str = "manual"
    content_type: str = "text"
    platform_post_id: str = None  
    platform_time: str = None     


# ------------------------------------------------
# Health check (test API running)
# ------------------------------------------------

@app.get("/")
def home():
    return {"message": "Cyberbullying Detection API is running"}


# ------------------------------------------------
# Prediction endpoint
# ------------------------------------------------


@app.post("/predict")
def predict(request: TextRequest):

    if not request.text or len(request.text.strip()) == 0:
        return {"error": "Empty input"}

    try:
        # START TIMER
        start_time = time.time()

        # 🔥 1. REGEX PREPROCESSING: Strip @mentions and URLs instantly
        clean_text = re.sub(r'http\S+|www\.\S+|@\w+', '', request.text).strip().lower()

        # 🔥 P20 FIX: Strip out giant standalone numbers/scientific notation
        # This replaces any number sequence with a blank space so the AI ignores it.
        clean_text = re.sub(r'\b\d+([.,]\d+)?([eE][+-]?\d+)?\b', '', clean_text).strip()

        print(f"🧹 Naked Text sent to AI: '{clean_text}'")

        # 🔥 2. SHORT TEXT FILTER: Kill noise like "BJP" or "By" (Improves accuracy!)
        # if len(clean_text.split()) <= 2:
        #      model_time_ms = round((time.time() - start_time) * 1000, 2)
        #      # Return a safe dummy response to skip the heavy ML math entirely
        #      return {
        #          "label": "non-cyberbullying",
        #          "severity": "none",
        #          "confidence": 0,
        #          "sarcasm": 0,
        #          "language": {"code": "en", "name": "english"},
        #          "latency": {"model_ms": model_time_ms, "shap_ms": 0, "total_ms": model_time_ms},
        #          "explanation": {
        #              "summary": "Text too short for targeted cyberbullying analysis.",
        #              "trigger_words": []
        #          }
        #      }

        # 🔹 Step 1: Prediction
        result = predict_post(clean_text)

        # 🔥 LABEL FIX: Convert "normal" to "non-cyberbullying"
        if result.get("label") == "normal":
            result["label"] = "non-cyberbullying"

        # 🔥 LANGUAGE FIX: The English Heuristic
        # If FastText thinks it's Hinglish, but it contains basic English grammar, force it to English
        if result.get("language", {}).get("code") == "mix":
            eng_check = [" is ", " are ", " you ", " this ", " the ", " a ", " to ", " i ", " am "]
            if any(word in f" {clean_text.lower()} " for word in eng_check):
                result["language"] = {"code": "en", "name": "english"}

        #  END TIMER
        model_time_ms = round((time.time() - start_time) * 1000, 2)
        latency_data = {"model_ms": model_time_ms}


        # Convert to percentage
        result["confidence"] = round(result["confidence"] * 100, 2)
        result["sarcasm"] = round(result["sarcasm"] * 100, 2)

        # Also components if needed
        for key in result.get("components", {}):
            result["components"][key] = round(result["components"][key] * 100, 2)

        # emotions list
        for emo in result.get("emotions", []):
            emo["score"] = round(emo["score"] * 100, 2)

        result["latency"] = latency_data

        # 🔹 Step 2: CONDITIONAL SHAP (The UI Fix)
        # Only run SHAP if the post is actually flagged as bullying!
        if result["label"] == "cyberbullying":
            shap_start_time = time.time()

            explanation = explain_text(request.text, prediction_data=result)
            result["explanation"] = explanation

            shap_time_ms = round((time.time() - shap_start_time) * 1000, 2)
            result["latency"]["shap_ms"] = shap_time_ms
            result["latency"]["total_ms"] = round(model_time_ms + shap_time_ms, 2) # Total Time
    
        else:
            # If it's safe, don't waste server power. Just return empty arrays.
            result["explanation"] = {
                "summary": "Flagged as safe by baseline models.",
                "trigger_words": [],
                "counter_words": [],
                "supporting_context": {}
            }
            result["latency"]["shap_ms"] = 0
            result["latency"]["total_ms"] = model_time_ms

        print("Incoming:", request.platform, request.content_type)
        
        #  Save to MongoDB
        save_prediction(
            text=request.text,
            result=result,
            platform=request.platform,
            content_type=request.content_type,
            latency_data=latency_data,
            platform_post_id=request.platform_post_id, 
            platform_time=request.platform_time
        )
        print(f"✅ Prediction Complete: {request.platform.upper()} | Time: {model_time_ms} ms | Label: {result['label'].upper()}")

        print("Saving to DB:", request.platform, request.content_type)
        return result

    except Exception as e:
        return {"error": str(e)}


# ------------------------------------------------
# History
# ------------------------------------------------
@app.get("/history")
def history(
    limit: int = 50,
    platform: str = None,
    label: str = None,
    severity: str = None,
    reviewed: bool = None,
    alert: bool = None,
    moderator_action: str = None,
    language: str = None,
    content_type: str = None,
    start_date: str = None,
    end_date: str = None
):

    try:
        if platform:
            platform = platform.lower()

        data = get_history(
            limit=limit,
            platform=platform,
            label=label,
            severity=severity,
            reviewed=reviewed,
            alert=alert,
            moderator_action=moderator_action,
            language=language,
            content_type=content_type,
            start_date=start_date,
            end_date=end_date
        )

        return {
            "count": len(data),
            "filters": {
                "platform": platform,
                "severity": severity,
                "reviewed": reviewed,
                "alert": alert,
                "moderator_action": moderator_action,
                "language": language,
                "content_type": content_type,
                "start_date": start_date,
                "end_date": end_date,
                "limit": limit
            },
            "data": data
        }

    except Exception as e:
        return {"error": str(e)}
    

# ------------------------------------------------
# Moderator action endpoint
# ------------------------------------------------
class ModerateRequest(BaseModel):
    id: str
    action: str   # ignore / report / delete
    reason: str = None
    saved: bool = False

@app.post("/moderate")
def moderate(request: ModerateRequest):

    try:
        updated = update_moderation_action(
            request.id,
            request.action,
            request.reason,
            request.saved
        )

        if updated == 0:
            return {"error": "Record not found"}

        return {
            "message": "Action updated successfully",
            "id": request.id,
            "action": request.action
        }

    except Exception as e:
        return {"error": str(e)}
    

# ------------------------------------------------
# Analytics
# ------------------------------------------------
@app.get("/analytics/severity")
def severity_analytics():
    return get_severity_stats()


@app.get("/analytics/platform")
def platform_analytics():
    return get_platform_stats()


@app.get("/analytics/trends")
def trend_analytics():
    return get_trend_stats()


@app.get("/analytics/language")
def language_analytics():

    try:
        raw_data = get_language_distribution()
        clean_data = {}
        
        # Format the ugly database names into beautiful labels for the UI
        for lang, count in raw_data.items():
            if lang == "keywords_hinglish_romanized":
                clean_data["Hinglish"] = count
            elif lang == "hindi_or_marathi":
                clean_data["Hindi / Marathi"] = count
            elif lang == "unknown":
                clean_data["Unknown"] = count
            else:
                clean_data[lang.capitalize()] = count # capitalizes 'english' to 'English'

        return clean_data
    except Exception as e:
        return {"error": str(e)}


@app.get("/analytics/dashboard-summary")
def dashboard_summary():
    try:
        return get_dashboard_summary()
    except Exception as e:
        return {"error": str(e)}
    

@app.get("/analytics/overview")
def analytics_overview():
    try:
        data = get_analytics_overview()
        return data
    except Exception as e:
        print(f"❌ Analytics Aggregation Error: {e}")
        return {"error": str(e)}

    
# ================================
# NEW ENDPOINT TO TRIGGER DATA COLLECTION
# ================================
@app.get("/collect")
def collect_data():

    try:
        result = run_once()

        return {
            "message": "Data collection completed",
            "summary": {
                "total_fetched": result["total_fetched"],
                "processed": result["processed"]
            },
            "preview": result["results"]
        }

    except Exception as e:
        return {"error": str(e)}
    


# ------------------------------------------------
# Export data as CSV
# ------------------------------------------------
def generate_flat_csv(data):
    """Helper function to flatten nested MongoDB data for clean CSVs."""
    if not data:
        return {"message": "No data found"}

    output = io.StringIO()
    
    # Define exact columns for the Research Paper export
    headers = [
        "Dataset_ID", "Platform", "Platform_Post_ID", "Text", 
        "Label", "Severity", "Confidence_Pct", "Sarcasm_Score", 
        "Language", "Primary_Emotion", "Emotion_Confidence_Pct", 
        "AI_Summary", "Trigger_Words", "Alert_Triggered",
        "Moderator_Action", "Moderator_Reason", "Created_At"
    ]
    
    writer = csv.DictWriter(output, fieldnames=headers)
    writer.writeheader()

    for index, row in enumerate(data, start=1):

        # Extract the language string cleanly
        lang_data = row.get("language")
        if isinstance(lang_data, dict):
            lang_str = lang_data.get("name", "unknown")
        else:
            lang_str = str(lang_data)


        # Safely extract SHAP Explanation
        explanation = row.get("explanation") or {}
        summary = explanation.get("summary", "")
        triggers = ", ".join([w.get("word", "") for w in explanation.get("trigger_words", [])])


        # 🔥 Extract the Top Emotion (Primary Emotion)
        emotions_list = row.get("emotions", [])
        primary_emotion = "None"
        emotion_score = 0
        if emotions_list and isinstance(emotions_list, list) and len(emotions_list) > 0:
            top_emo = emotions_list[0]
            primary_emotion = top_emo.get("label", "none").capitalize()
            emotion_score = top_emo.get("score", 0)


        flat_row = {
            "Dataset_ID": f"CB-{index:05d}",
            "Platform": row.get("platform", "unknown"),
            "Platform_Post_ID": row.get("platform_post_id", "N/A"),
            "Text": row.get("text", ""),
            
            # Flattening the nested JSON
            "Label": row.get("label", ""),
            "Severity": row.get("severity", ""),
            "Confidence_Pct": row.get("confidence", 0),
            "Sarcasm_Score": row.get("sarcasm", 0),
            "Language": lang_str.capitalize(), 

            "Primary_Emotion": primary_emotion,
            "Emotion_Confidence_Pct": emotion_score,

            "AI_Summary": summary,
            "Trigger_Words": triggers,

            "Alert_Triggered": "Yes" if row.get("alert") else "No",
            "Moderator_Action": row.get("moderator_action", "Pending"),
            "Moderator_Reason": row.get("moderator_reason", ""),
            # "Created_At": row.get("timestamp", "")  
            "Created_At": str(row.get("timestamp", "")).split(".")[0].replace("T", " ")# get_history renamed this to timestamp
        }
        writer.writerow(flat_row)

    raw_csv_string = output.getvalue()
    
    # Encode it as utf-8-sig to force Excel to read Hindi/Tamil/Bengali correctly
    csv_bytes = raw_csv_string.encode('utf-8-sig')
    
    return StreamingResponse(
        io.BytesIO(csv_bytes),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=cyberbullying_export.csv"}
    )

@app.get("/export/view")
def export_current_view(limit: int = 1000, platform: str = None,label: str = None, severity: str = None, alert: bool = None):
    """Exports only the filtered data the moderator is currently looking at."""
    try:
        data = get_history(limit=limit, platform=platform, label=label, severity=severity, alert=alert)
        return generate_flat_csv(data)
    except Exception as e:
        return {"error": str(e)}

@app.get("/export/full")
def export_full_dataset():
    """Exports the entire dataset for research/analytics."""
    try:
        # Pass a massive limit to get everything
        data = get_history(limit=100000) 
        return generate_flat_csv(data)
    except Exception as e:
        return {"error": str(e)}
    

# ------------------------------------------------
# History page
# ------------------------------------------------

@app.get("/history/reviewed")
def audit_history_route():
    try:
        data = get_audit_history()
        return {"status": "success", "data": data}
    except Exception as e:
        return {"status": "error", "message": str(e)}