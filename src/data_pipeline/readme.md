# 📁 Phase 1: Data Pipeline & Keyword Engineering

**Project Phase:** Phase 1 (Foundational Dataset Generation)
**Timeline:** Semester 7 (November - December 2025)

## 📌 Purpose
This directory contains the core orchestration utilities used to bootstrap the cyberbullying detection models. Rather than relying on static, pre-existing datasets, these scripts were engineered to autonomously build, translate, validate, and harvest our custom multilingual training corpus. 

By utilizing **Distant / Weak Supervision**, this pipeline allowed us to auto-label thousands of social media posts across 14 Indian languages without requiring massive manual annotation.

## 🧩 Directory Contents & Execution Flow

* **`pipeline_config.py`**
  * **The Configuration Hub.** Centralizes system variables, specifically the `TARGET_LANGUAGES` array (English, Hinglish, and 12 regional Indian languages) and batch sizing, ensuring consistent execution across all scripts.

* **`01_build_keywords.py`**
  * **The Translation Engine.** Expands a base list of English toxic keywords into 14 regional languages using Google Translate APIs, including programmatic generation of Romanized "Hinglish" by mixing Hindi and English matrices.

* **`02_validate_database/` (Directory)**
  * **The Diagnostic Suite.** A collection of integrity-check scripts (`check_keywords.py`, `debug_hindi_keywords.py`, etc.) designed to ensure the generated JSON databases are structurally sound, file sizes are correct, and native scripts (e.g., Devanagari) parse without encoding errors.

* **`03_load_keywords.py`**
  * **The Compiler.** Converts the massive raw JSON dictionaries (over 15,000+ words) into highly optimized Regex patterns. This is the precursor to the auto-labeling pipeline.

* **`04_scrape_historical_data.py`**
  * **The Historical Harvester.** The orchestrator script that loads the compiled keywords and interfaces securely with the Reddit, X (Twitter), and YouTube APIs (via the `data_collection` modules) to pull natural, localized social media text.

## ⚙️ Key System Contributions
1. **Rule-Based Filtering:** Provides the initial keyword matching logic to flag obvious toxicity.
2. **Weak Supervision (Auto-Labeling):** Uses the compiled regex patterns to automatically assign `1` (Cyberbullying) or `0` (Normal) to raw scraped data, creating the foundational ground-truth labels for Phase 2 Teacher Model training.
3. **Data Labeling Support:** Dramatically reduces human-in-the-loop workload during the initial dataset construction phase.

## ⚠️ Important Note on Data Storage & Security
**This folder contains executable CODE ONLY.** 
* **No Secrets:** API Keys have been strictly decoupled from the codebase and are loaded dynamically via the root `.env` file.
* **No Raw Data:** To maintain strict separation of concerns, no raw datasets, JSON databases, or scraped CSVs are stored here. All resulting outputs are safely routed to `Sem8_Cyber_bullying_detection/resources/keywords/` and `data/raw/`.