# 📂 Services Directory: `src/services/`

## 📌 Overview
The `services/` directory is responsible for the application's **Data Infrastructure**. It isolates all external communication and data cleaning logic from the UI components. This separation of concerns ensures that if the backend API structure changes (e.g., a field name is renamed in the database), only these files require updates, keeping the UI components "blind" to backend volatility.

The directory is divided into two specialized modules:
1. **Communication (`api.js`)**: The "Messenger" that speaks to the FastAPI server.
2. **Normalization (`transform.js`)**: The "Sanitizer" that prepares raw data for React consumption.

---

## 🛠️ Module Breakdown

### 1. `api.js` (The Communication Layer)
This module acts as the interface for the backend hosted at `http://localhost:8000`. It encapsulates `fetch` logic and query string construction.

**Key Exported Functions:**

* **`fetchPosts()`**: Retrieves the latest 15 records from the `/history` endpoint. It is the primary data source for the moderation feed.
* **`fetchAlerts()`**: Specific query to `/history?alert=true`. It retrieves high-priority flagged content specifically for the **Dashboard's "Recent Alerts"** section.
* **`moderatePost(id, action, reason)`**: Dispatches moderation decisions (**Ignore**, **Delete**, **Report**) to the `/moderate` endpoint. It supports a `reason` payload for audit tracking.
* **`collectData()`**: A trigger function for the `/collect` endpoint, instructing the backend to execute its scraping/ingestion pipeline.
* **`fetchSeverityStats()`**: Pulls aggregated counts (none, mild, moderate, severe) from `/analytics/severity` to populate the Dashboard's **StatsOverview**.
* **`exportPosts(filters)`**: Sanitizes active filters and generates a dynamic query string to download a CSV blob via the `/export` endpoint.

---

### 2. `transform.js` (The Normalization Layer)
The backend and frontend use different naming conventions and data scales. `transform.js` ensures the UI receives a standardized, safe "Post Object."

**Critical Normalizations:**

* **Score Normalization:** Backend scores (0–100) are converted into decimal percentages (0.0 – 1.0) to work with frontend progress bars and confidence indicators.
* **Severity Standardization:** Maps varied backend strings into four immutable UI categories: `none`, `mild`, `moderate`, and `severe`.
* **Relative Time Transformation:** Contains the `formatTime` utility that converts ISO 8601 timestamps into human-centric strings like "5m ago" or "2h ago."
* **Platform Categorization:** Uses `normalizePlatform` to parse raw source strings (e.g., "twitter_api") and map them to clean UI labels ("Twitter") and their respective icons.

---

## 🔄 The Data Pipeline

The flow follows a strict **Fetch-Transform-Consume** pattern:

1. **Fetch:** `api.js` receives raw JSON from the server.
2. **Clean:** The raw JSON is passed into `transformPost()` within `transform.js`.
3. **Defaulting:** Missing values are caught and defaulted (e.g., `sarcasm || 0`), ensuring the UI never encounters a "Crash on Null" scenario.
4. **Inject:** The Layout layer receives the "Clean" object and distributes it to components via props.

---

## 📋 Standardized Post Schema

Every post processed by the services layer adheres to this frontend contract:

| Property     | Type    | Description |
| :----------- | :------ | :---------- |
| `id`         | String  | Unique ID mapped from `_id`. |
| `text`       | String  | The actual post content. |
| `platform`   | String  | Normalized label: Twitter, Reddit, YouTube, or Manual. |
| `time`       | String  | Formatted relative time (e.g., "10m ago"). |
| `verdict`    | String  | Logic-driven: "BULLYING" or "NON-BULLYING". |
| `confidence` | Float   | 0.0 to 1.0 confidence score. |
| `emotion`    | String  | Top detected emotion (Aggression, Distress, Neutral). |
| `sarcasm`    | Float   | 0.0 to 1.0 sarcasm probability score. |
| `reviewed`   | Boolean | Tracks if the post has been handled by a moderator. |