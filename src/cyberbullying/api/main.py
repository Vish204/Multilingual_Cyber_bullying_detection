import json
import os
import io
import csv

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import time



from cyberbullying.inference.inference_service import predict_post
from cyberbullying.explainability.shap_explainer import explain_text
from cyberbullying.database.db import save_prediction, get_history
from cyberbullying.database.db import update_moderation_action
from cyberbullying.database.db import get_severity_stats
from cyberbullying.database.db import get_platform_stats
from cyberbullying.database.db import get_trend_stats
from cyberbullying.database.db import get_language_distribution
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

        # 🔹 Step 1: Prediction
        result = predict_post(request.text)

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
            explanation = explain_text(request.text)
            result["explanation"] = explanation
        else:
            # If it's safe, don't waste server power. Just return empty arrays.
            result["explanation"] = {
                "trigger_words": [],
                "counter_words": [],
                "supporting_signals": {}
            }

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
    

class ExplainRequest(BaseModel):
    text: str

@app.post("/explain")
def explain(request: ExplainRequest):
    """
    Dedicated endpoint for SHAP explainability. 
    Only called when a human clicks 'Explain' to save server load.
    """
    if not request.text or len(request.text.strip()) == 0:
        return {"error": "Empty input"}

    try:
        explanation = explain_text(request.text)
        return {"explanation": explanation}
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
        "Alert_Triggered", "Language", "Moderator_Action", "Created_At"
    ]
    
    writer = csv.DictWriter(output, fieldnames=headers)
    writer.writeheader()

    for index, row in enumerate(data, start=1):
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
            "Alert_Triggered": row.get("alert", False),
            "Language": row.get("language", "unknown"), 
            "Moderator_Action": row.get("moderator_action", "Pending"),
            "Created_At": row.get("timestamp", "")  # get_history renamed this to timestamp
        }
        writer.writerow(flat_row)

    output.seek(0)
    return StreamingResponse(
        output,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=cyberbullying_export.csv"}
    )

@app.get("/export/view")
def export_current_view(limit: int = 1000, platform: str = None, severity: str = None, alert: bool = None):
    """Exports only the filtered data the moderator is currently looking at."""
    try:
        data = get_history(limit=limit, platform=platform, severity=severity, alert=alert)
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