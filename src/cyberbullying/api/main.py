from fastapi import FastAPI
from pydantic import BaseModel
import json
import os


from cyberbullying.inference.inference_service import predict_post
from cyberbullying.explainability.shap_explainer import explain_text
from cyberbullying.database.db import save_prediction, get_history

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



@app.get("/history")
def history(
    limit: int = 50,
    platform: str = None,
    severity: str = None
):

    try:
        data = get_history(
            limit=limit,
            platform=platform,
            severity=severity
        )

        return {
            "count": len(data),
            "filters": {
                "platform": platform,
                "severity": severity,
                "limit": limit
            },
            "data": data
        }

    except Exception as e:
        return {"error": str(e)}