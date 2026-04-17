# 📁 Sub-Folder: `src/components/analytics/`

## 📌 Purpose
The `analytics/` folder acts as the **System Insights & Diagnostics Dashboard** (the "Engineer's View"). It is responsible for providing high-level operational metrics, tracking the XLM-Roberta model's real-world accuracy, and analyzing toxicity trends across multiple platforms and languages.

---

## 🧩 Key Components

### `StatsRibbon.jsx` & `MiniStats.jsx`
The overarching data summary for the system (Top Row).

* **Contextual Metrics:**  
  Displays total processed records locked to a **15-day window** to ensure data consistency.

* **ML Performance Proof:**  
  Injects live **System Latency** (average inference speed of the last 50 posts) to demonstrate real-time readiness.

---

### `TrendChart.jsx`
A dual-line temporal analysis of platform activity.

* **Signal vs. Noise:**  
  Plots:
  * Total Volume (solid blue)  
  * Confirmed Cyberbullying (dashed red)  

* **Safety Net:**  
  Includes a graceful **"Collecting Trend Data"** empty state if insufficient data exists within the 15-day window.

---

### `PlatformChart.jsx`
Transforms basic volume counting into a **Toxicity Density Analysis**.

* **Comparative Insights:**  
  Uses a grouped bar chart to compare:
  * Total Posts  
  * Flagged Bullying  

* **Logic:**  
  Helps moderators identify platforms with the highest concentration of harmful content, not just the highest traffic.

---

### `LanguageChart.jsx`
Demonstrates the system's multilingual capabilities (14 Indian languages).

* **Data Sanitization:**  
  Filters out legacy or invalid data (e.g., "Unknown", "None") before rendering.

* **Interactive Toggle:**  
  * Defaults to Top 5 languages  
  * Includes a dynamic **"See All"** option to expand the dataset

---

### `ConfidenceChart.jsx` (Decision Alignment)
Tracks **AI vs Human moderation alignment**.

* **Alignment Insight:**  
  Compares:
  * AI decisions accepted by moderators  
  * AI decisions overridden by moderators  

* **Threshold Logic:**  
  Requires a minimum of **20 moderation actions** for statistical validity.

* **Fallback UI:**  
  Displays **"Calibrating Alignment..."** until enough data is available.

---

### `SeverityChart.jsx`
A visual breakdown of active threats.

* **Visualization:**  
  Doughnut chart using traffic-light color coding:
  * Green → None  
  * Yellow → Mild  
  * Orange → Moderate  
  * Red → Severe  

---

## 🏗️ Structural & Styling Files

### `DistributionRow.jsx`
* A grid wrapper ensuring consistent **F-Pattern layout** across analytics sections.

### `AnalyticsHeader.jsx`
* Static page header containing the title and descriptive subtext.

### `analysis.css`
* Contains analytics-specific styling.
* Includes `.empty-state` flexbox fallbacks to ensure stable rendering during live demos.