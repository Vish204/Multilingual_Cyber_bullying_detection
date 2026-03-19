# Database Module (MongoDB Integration)

---

# Location

```
src/cyberbullying/database/
```

---

# Files Overview

## db.py

### Purpose

Core database handler for MongoDB operations.

### Responsibilities

* Establish connection to MongoDB
* Define database and collection
* Provide reusable functions for:

  * Saving predictions
  * Fetching history with filters

---

## test_db.py

### Purpose

Standalone testing script to verify database functionality.

### Responsibilities

* Insert dummy prediction data
* Fetch stored data
* Validate database connection and schema

---

# Database Structure

## Database Name

```
cyberbullying_db
```

## Collection

```
predictions
```

---

# Stored Document Schema

Each prediction is stored as:

```json id="p7f9x2"
{
  "text": "you are an idiot",
  "platform": "manual",
  "content_type": "text",

  "label": "cyberbullying",
  "severity": "moderate",
  "confidence": 0.69,

  "components": {
    "cyberbullying": 0.99,
    "sarcasm": 0.07
  },

  "emotions": [
    {
      "label": "aggression",
      "score": 0.84
    }
  ],

  "explanation": {
    "trigger_words": [...],
    "counter_words": [...],
    "supporting_signals": {...}
  },

  "timestamp": "2026-03-19T20:23:21"
}
```

---

# Core Functions

## save_prediction(text, result)

### Description

Stores prediction output into MongoDB.

### Automatically Adds

* platform (default: "manual")
* content_type (default: "text")
* timestamp

---

## get_history(limit, platform, severity)

### Description

Fetches stored predictions with optional filters.

### Supports

* Limit (number of results)
* Platform filtering
* Severity filtering
* Sorted by latest first

---

# Data Flow

```id="6e4tq1"
User Input
   ↓
FastAPI (/predict)
   ↓
ML + SHAP
   ↓
save_prediction()
   ↓
MongoDB
   ↓
/history → get_history()
```

---

# Why MongoDB

* Flexible schema (supports nested JSON)
* Well-suited for ML outputs
* Easy integration with FastAPI and React
* Scalable for future real-time systems
