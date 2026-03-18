from fastapi import FastAPI
from pydantic import BaseModel
import json
import os


from cyberbullying.inference.inference_service import predict_post
from cyberbullying.api.logger import log_prediction

# ------------------------------------------------
# App init
# ------------------------------------------------

app = FastAPI()


# ------------------------------------------------
# Request schema (VERY IMPORTANT)
# ------------------------------------------------

class TextRequest(BaseModel):
    text: str


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
        result = predict_post(request.text)
        log_prediction(request.text, result)
        return result

    except Exception as e:
        return {"error": str(e)}

LOG_FILE = "logs/predictions.json"


@app.get("/history")
def get_history():

    if not os.path.exists(LOG_FILE):
        return {"message": "No history found", "data": []}

    with open(LOG_FILE, "r") as f:
        data = json.load(f)

    return {
        "count": len(data),
        "data": data
    }