# 📁 Sub-Folder: `src/components/dashboard/`

## 📌 Purpose
The `dashboard/` folder contains the specialized widgets that make up the landing page (`Dashboard.jsx`). These components focus on **Aggregated Data** rather than individual post analysis.

---

## 🧩 Key Components

### `StatsOverview.jsx`

* **Data Source:**  
  Pulls data from `fetchSeverityStats`.

* **Functionality:**  
  Displays high-level metrics such as:
  * Total Posts  
  * Bullying %  
  * Reviewed Count  

---

### `RecentAlerts.jsx`

* **Purpose:**  
  A mini-feed for high-priority content.

* **Functionality:**  
  Fetches and displays posts where `alert=true`, allowing moderators to quickly jump into severe cases directly from the dashboard.

---

### `NavigationCards.jsx`

* **Purpose:**  
  Acts as the primary UI routing mechanism.

* **Logic:**  
  Uses `useNavigate` (React Router) to move users between:
  * Moderation  
  * History  
  * Analysis  

---

### `SystemFlow.jsx`

* **Purpose:**  
  Provides a visual representation of the ML pipeline.

* **Flow Representation:**  
  Input → XGBoost → Fusion → Action  

* **Benefit:**  
  Helps users understand how data flows through the system and how decisions are derived.