data/

This folder contains keyword and auxiliary data utilities used for cyberbullying detection.

data/
├── 01_build_keywords.py
├── 02_scrape_social_media.py
├── 03_load_keywords.py
├── 04_validate_keywords.py
├── 05_toxic_data.py (synthetic data generation for words like kill,etc)
└── __init__.py


Purpose

Build and validate cyberbullying keyword lists

Support rule-based filtering and weak supervision

Assist preprocessing and data labeling stages

Note: This folder contains data-related scripts, not raw datasets.
Raw datasets are stored separately (e.g., data/raw/ at project root).