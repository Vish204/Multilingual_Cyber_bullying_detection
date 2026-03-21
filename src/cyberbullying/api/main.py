import json
import os
import io
import csv

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel



from cyberbullying.inference.inference_service import predict_post
from cyberbullying.explainability.shap_explainer import explain_text
from cyberbullying.database.db import save_prediction, get_history
from cyberbullying.database.db import update_moderation_action
from cyberbullying.database.db import get_severity_stats
from cyberbullying.database.db import get_platform_stats
from cyberbullying.database.db import get_trend_stats
from cyberbullying.database.db import get_language_distribution
from cyberbullying.database.db import export_data
from cyberbullying.collector.run_collector import run_once


# ------------------------------------------------
# App init
# ------------------------------------------------

app = FastAPI()


# ------------------------------------------------
# Request schema (VERY IMPORTANT)
# ------------------------------------------------

class TextRequest(BaseModel):
    text: str
    platform: str = "manual"
    content_type: str = "text"


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
        # 🔹 Step 1: Prediction
        result = predict_post(request.text)

        # 🔹 Step 2: SHAP Explanation
        explanation = explain_text(request.text)

        # 🔹 Step 3: Attach explanation
        result["explanation"] = explanation

        # 🔥 Save to MongoDB
        save_prediction(
            text=request.text,
            result=result,
            platform=request.platform,
            content_type=request.content_type
        )
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

@app.post("/moderate")
def moderate(request: ModerateRequest):

    try:
        updated = update_moderation_action(
            request.id,
            request.action
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
        data = get_language_distribution()
        return data

    except Exception as e:
        return {"error": str(e)}


# ------------------------------------------------
# Export data as CSV
# ------------------------------------------------
@app.get("/export")
def export_csv(
    limit: int = 1000,
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
        # 🔥 Reuse SAME logic as history
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

        if not data:
            return {"message": "No data found"}

        # -----------------------------
        # Convert to CSV
        # -----------------------------
        output = io.StringIO()

        headers = data[0].keys()
        writer = csv.DictWriter(output, fieldnames=headers)
        writer.writeheader()

        for row in data:

            # 🔥 Convert nested fields properly
            if "components" in row:
                row["components"] = json.dumps(row["components"])

            if "emotions" in row:
                row["emotions"] = json.dumps(row["emotions"])

            if "explanation" in row:
                row["explanation"] = json.dumps(row["explanation"])

            if "language" in row:
                row["language"] = json.dumps(row["language"])

            writer.writerow(row)

        output.seek(0)

        return StreamingResponse(
            output,
            media_type="text/csv",
            headers={
                "Content-Disposition": "attachment; filename=export.csv"
            }
        )

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