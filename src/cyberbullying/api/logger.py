import json
from datetime import datetime
import os

LOG_FILE = "logs/predictions.json"


def log_prediction(text, result):

    log_entry = {
        "text": text,
        "label": result["label"],
        "confidence": result["confidence"],
        "severity": result["severity"],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # create file if not exists
    if not os.path.exists(LOG_FILE):
        os.makedirs("logs", exist_ok=True)
        with open(LOG_FILE, "w") as f:
            json.dump([], f)

    # read existing logs
    with open(LOG_FILE, "r") as f:
        data = json.load(f)

    # append new entry
    data.append(log_entry)

    # write back
    with open(LOG_FILE, "w") as f:
        json.dump(data, f, indent=4)