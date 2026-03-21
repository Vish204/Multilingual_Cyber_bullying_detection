# Collector Module

---

# Overview

The Collector Module enables real-time data ingestion from multiple platforms and feeds it into the cyberbullying detection pipeline.

It converts the system from a static API into a live monitoring system.

---

# Folder Structure

```id="fs92kd"
src/cyberbullying/collector/

├── reddit_collector.py      # Reddit posts + comments
├── twitter_collector.py     # Twitter tweets
├── youtube_collector.py     # YouTube comments

├── cleaning_utils.py        # Text preprocessing
├── language_utils.py        # Language detection
├── balancing.py             # Multilingual balancing

├── run_collector.py         # Trigger pipeline (manual mode)
└── readme.md
```

---

# Supported Platforms

| Platform | Data Type        |
| -------- | ---------------- |
| Reddit   | Posts + Comments |
| Twitter  | Tweets           |
| YouTube  | Comments         |

---

# Core Features

## 1. Multi-Platform Collection

* Fetches real-world data from APIs
* Combines posts, comments, and tweets
* Adds metadata:

  * platform
  * content_type

---

## 2. Data Cleaning

* Removes URLs
* Removes excessive whitespace
* Keeps emojis (important for emotion detection)

---

## 3. Language Detection

* Uses `langdetect`
* Converts ISO codes to readable names

### Example

```
en → english
hi → hindi
mr → marathi
```

---

## 4. Multilingual Balancing

Ensures diversity in collected data:

* Detects language distribution
* If English dominates:

  * Adds targeted multilingual content
* Ensures presence of:

  * Hindi
  * Marathi
  * Tamil
  * Bengali (if available)

---

## 5. Deduplication

* Removes duplicate text entries
* Uses hash-based filtering

---

## 6. API Integration

Each collected item is sent to:

```id="api23x"
POST /predict
```

### Receives

* label
* severity
* confidence
* emotions
* components
* explanation

---

## 7. Database Storage

Stores results in MongoDB:

* original text
* predictions
* explanation
* metadata
* timestamp

---

# End-to-End Flow

```id="flow72p"
Platforms (Reddit/Twitter/YouTube)
        ↓
Collector Module
        ↓
Cleaning + Language Detection
        ↓
Balancing + Deduplication
        ↓
FastAPI (/predict)
        ↓
ML Models (Fusion System)
        ↓
MongoDB Storage
```

---

# Execution Mode

## Manual Trigger (Recommended)

Triggered via API or button:

```id="run88p"
POST /collect
```

### Response

```json id="res91k"
{
  "message": "Data collection completed",
  "summary": {
    "total_fetched": 22,
    "processed": 15
  }
}
```

---

# Performance Optimization

* Limit fetched data per platform
* Cap final processed data (e.g., 15 items)
* Avoid unnecessary API calls

---

# Design Decisions

## Why not force language via API?

* APIs often return Romanized text
* Leads to poor accuracy for Indian languages

### Solution

* Collect data naturally
* Detect language locally
* Balance intelligently

---

# Future Enhancements

* Real-time streaming (WebSockets)
* Platform-specific dashboards
* Language-wise analytics
* Smart sampling based on trends

---

# Final Goal

A scalable, multilingual, real-time cyberbullying monitoring system that:

* Collects live data
* Detects harmful content
* Explains predictions
* Supports moderation workflows
