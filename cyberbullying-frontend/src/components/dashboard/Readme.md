# 📁 Sub-Folder: `src/components/dashboard/`

## 📌 Purpose
The `dashboard/` folder acts as the **Operational Landing Page** (the "Moderator's View"). Unlike the Analytics page (which focuses on historical ML performance), this module is designed for real-time triage. It synthesizes immediate workload, high-priority threats, and the system's operational scale into a single, actionable interface.

---

## 🧩 Key Components

### `StatsOverview.jsx`
The high-level operational summary for active moderators.

* **Contextual Data:**  
  Displays Total Posts and Bullying Percentage strictly locked to a 15-day window for immediate relevance.

* **Workload Tracking:**  
  Features a **"Pending Priority"** metric, directly tracking Severe/Moderate posts that require human review.

* **Multilingual Scale:**  
  Displays the **"14 Supported Languages"** metric, highlighting the XLM-Roberta model's coverage capabilities at login.

---

### `RecentAlerts.jsx`
A rapid-response mini-feed for critical content.

* **Purpose:**  
  Acts as an immediate triage queue.

* **Functionality:**  
  Fetches and isolates posts flagged with high severity (`alert=true`), allowing moderators to bypass the main history feed and jump directly into mitigating severe cyberbullying cases.

---

### `SystemFlow.jsx`
An architectural visualization of the Machine Learning pipeline.

* **Purpose:**  
  Provides explainability (XAI) to non-technical users or stakeholders.

* **Flow Representation:**  
  `Input → Feature Extraction → Late-Fusion Model (XGBoost) → Final Action`

* **Benefit:**  
  Demystifies the "black box" of the AI, illustrating how the system derives its decisions.

---

### `NavigationCards.jsx`
The primary routing mechanism for the dashboard UI.

* **Logic:**  
  Utilizes `react-router-dom` (`useNavigate`) to transition users between the core system pillars:

  * **Moderation Workspace** — Action interface  
  * **History Logs** — Audit trail  
  * **System Insights** — Analytics/Diagnostics page  

---

## 🏗️ Structural & Styling Files

### `DashboardHeader.jsx`
* Renders the static welcome message and dynamic system status context.

### `dashboard.css`
* Contains grid layouts, card hover effects, and responsive design rules.
* Ensures the dashboard scales correctly across desktop screen sizes.