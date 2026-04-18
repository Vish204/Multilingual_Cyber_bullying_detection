# 📁 Sub-Folder: `src/components/history/`

## 📌 Purpose
The `history/` folder acts as the **System Audit Log & Retraining Pipeline**. This is not merely a data viewer; it is the final stage of the Machine Learning lifecycle. It is specifically designed to track Human-in-the-Loop (HITL) decisions, calculate semantic alignment between the AI and the moderator, and automatically generate curated datasets for retraining the XGBoost model.

---

## 🧩 Key Components

### `HistoryLayout.jsx`
The "Boss" component that orchestrates data fetching, filtering, and the export pipeline.

* **Smart Filtering:**  
  Utilizes instant, client-side JavaScript `.filter()` logic for the toggle switch, ensuring zero-latency UX during live demos without hammering the backend API.

* **The Dynamic Export Engine:**  
  The bridge between the web dashboard and the ML Jupyter Notebooks. It dynamically generates a CSV of the currently visible table.

  * **Viva Impact:**  
    Automatically changes the output filename from:
    * `System_Audit_Log.csv` (when viewing all records)  
    * `ML_Retraining_Dataset.csv` (when filtering for model failures)  

    This demonstrates an automated data curation pipeline for future model versions.

---

### `HistoryHeader.jsx`
The streamlined control surface for the audit table.

* **Intentional UX:**  
  Replaces complex multi-dropdown filters with a single, high-value toggle:  
  **"Show AI Disagreements"**

* **Logic:**  
  Optimized for identifying edge cases where the AI failed, supporting Trust & Safety workflows.

---

### `AuditTable.jsx`
The visual audit trail, built using a **Lean Data Table** architecture.

* **Semantic Alignment Logic:**  
  Displays:
  * ✅ Agreed  
  * ⚠️ Overruled  
  * ⏳ Pending  

  Based on backend-calculated True Positive / False Positive logic (not simple string matching).

* **Data Formatting:**  
  Includes a lightweight helper to normalize and format raw database values  
  (e.g., `youtube` → `YouTube`).

* **Speed Over Bloat:**  
  Avoids heavy pagination. Instead:
  * Shows the latest ~100 reviewed actions  
  * Uses vertical scrolling for fast access and reliability during demos  

---

## 🏗️ Structural & Styling Files

### `history.css`

* **High-Contrast Badging:**  
  Uses strong visual emphasis (deep reds, bold text) for the `⚠️ Overruled` state to highlight retraining data.

* **UX Protections:**  
  * Sticky table headers (`<thead>`) for readability during scroll  
  * `text-overflow: ellipsis` to truncate long posts without breaking layout consistency  

src/
 └── components/
      └── history/
           ├── HistoryLayout.jsx   # The "Boss" component. Holds the state, fetches data, and handles the CSV export logic.
           ├── HistoryHeader.jsx   # Renders the Top Bar (Title, Toggle Switch, Export Button).
           ├── AuditTable.jsx      # Renders the actual <table>, maps the rows, and formats the alignment badges.
           └── history.css         # Clean, scoped styling for the table, sticky headers, and truncation.