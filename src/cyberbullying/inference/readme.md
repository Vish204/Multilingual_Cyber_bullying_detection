# Cyberbullying Inference Service

---

# Overview

This module provides a production-ready inference layer for the cyberbullying detection system.

It acts as a bridge between:

* Trained ML models (Phase 3)
* Backend APIs / real-time systems (Phase 4)

---

# Files

## `inference_service.py`

Core file responsible for:

* Loading all trained models
* Running predictions
* Applying fusion logic
* Returning structured output

---

# How It Works

## Pipeline Flow

```
Input Text
   ↓
run_component_predictions()
   ↓
Outputs:
   - p_cb (cyberbullying)
   - p_sarcasm
   - p_neutral
   - p_aggression
   - p_distress
   ↓
compute_fusion_score()
   ↓
Final Output:
   - prediction
   - severity
   - fusion score
   - top emotions
```

---

# Main Function

```
predict_post(text: str) → dict
```

---

# Output Format

```json
{
  "text": "...",
  "prediction": "CYBERBULLYING / NORMAL",
  "severity": "NONE / MILD / MODERATE / SEVERE",
  "fusion_score": 0.0,
  "probabilities": {
    "cyberbullying": 0.0,
    "sarcasm": 0.0,
    "neutral": 0.0,
    "aggression": 0.0,
    "distress": 0.0
  },
  "top_emotions": [
    {
      "label": "...",
      "score": 0.0
    }
  ]
}
```

---

# Usage Example

```python
from cyberbullying.inference.inference_service import predict_post

result = predict_post("You are useless")
print(result)
```

---

# Notes

* Models are loaded once at startup for performance
* Uses absolute imports for compatibility
* Designed to integrate with:

  * FastAPI backend
  * Real-time monitoring systems
  * Dashboards

---

# Future Integration

This module will be used by:

* API endpoints (`/predict`)
* Data collectors (Reddit, etc.)
* Database logging systems
* Explainability modules (SHAP)
