# 📁 Phase 1: Data Pipeline & Keyword Engineering

**Project Phase:** Phase 1 (Foundational Dataset Generation)
**Timeline:** Semester 7 (November - December 2025)

## 📌 Purpose
This directory contains the core data engineering utilities used to bootstrap the cyberbullying detection models. Rather than relying on static, pre-existing datasets, these scripts were engineered to autonomously build, translate, validate, and synthesize our custom multilingual training corpus. 

By utilizing **Distant / Weak Supervision**, these scripts allowed us to auto-label thousands of social media posts across 14 Indian languages without requiring massive manual annotation.

## 🧩 Directory Contents

* **`01_build_keywords.py`**
  * The Translation Engine. Expands a base list of English toxic keywords into 14 regional languages using Google Translate, including programmatic generation of Romanized "Hinglish".
* **`02_scrape_social_media.py`**
  * The Historical Harvester. Interfaces with the Reddit, Twitter, and YouTube APIs to pull natural social media text based on our keyword dictionaries.
* **`03_load_keywords.py`**
  * The Compiler. Converts the massive raw JSON dictionaries into highly optimized Regex patterns for the auto-labeling pipeline.
* **`04_validate_keywords.py`**
  * The Diagnostic Suite. A set of integrity checks to ensure the JSON databases are structurally sound and native scripts (e.g., Devanagari) are parsing correctly.
* **`05_toxic_data.py`**
  * Synthetic Data Generator. Specifically engineered to synthesize extreme edge-case toxic data (e.g., severe threats, "kill", etc.) to ensure the models recognize high-severity triggers even if they are rare in natural API scraping.
* **`pipeline_config.py`**
  * pipeline_config has languages we did.

* **`__init__.py`**
  * Python package initializer.

## ⚙️ Key System Contributions
1. **Rule-Based Filtering:** Provides the initial keyword matching logic to flag obvious toxicity.
2. **Weak Supervision (Auto-Labeling):** Uses the compiled regex patterns to assign `1` (Cyberbullying) or `0` (Normal) to raw data, creating the foundational labels for Phase 2 Teacher Model training.
3. **Data Labeling Support:** Dramatically reduces human-in-the-loop workload during the initial dataset construction.

## ⚠️ Important Note on Data Storage
**This folder contains executable CODE ONLY.** To maintain strict separation of concerns, no raw datasets or scraped CSVs are stored here. All resulting output data (scraped posts, synthetic text, and CSVs) are routed to and stored at the project root: `Sem8_Cyber_bullying_detection/data/raw/` and `data/processed/`.