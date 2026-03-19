# Collector Module

---

# Overview

The Collector Module is responsible for acquiring real-world data from external platforms (currently Reddit), sending it through the ML pipeline, and storing the analyzed results in the database.

It transforms the system from a static API into a real-time monitoring pipeline.

---

# Folder Structure

```id="c0m8a1"
src/cyberbullying/collector/
│
├── reddit_collector.py     # Main Reddit data fetcher + API integration
├── (future)
│    ├── youtube_collector.py
│    ├── twitter_collector.py
│
└── utils/ (optional later)
     ├── deduplication.py
     ├── scheduler.py
```

---

# Current Functionality

## Reddit Data Collection

* Uses Reddit API (PRAW)
* Fetches posts from selected subreddits
* Extracts:

  * text (title + body)
  * platform = "reddit"
  * content_type = "post"

---

## API Integration

Sends collected text to:

```
POST /predict
```

Receives:

* label
* severity
* confidence
* components
* emotions
* SHAP explanation

---

## Database Storage

Stores enriched results in MongoDB.

Includes:

* original text
* prediction outputs
* explanation
* metadata (platform, content_type, timestamp)

---

## End-to-End Flow

```id="y9v2bz"
Reddit → Collector → FastAPI (/predict) → ML System → MongoDB
```

---

# Planned Features

## Comments Scraping

Extend the collector to include Reddit comments.

### Extract

* comment body
* content_type = "comment"

Improves detection coverage (bullying often occurs in comments).

---

## Continuous Monitoring

Convert the collector into a loop-based system.

* Automatically fetches and processes data at intervals
* Enables near real-time monitoring

---

## Deduplication

Prevent duplicate storage using hash-based filtering.

* Ensures clean and efficient database

---

# Final Goal

A real-time cyberbullying monitoring system capable of:

* Continuously collecting data
* Detecting harmful content
* Explaining model decisions
* Storing structured insights
