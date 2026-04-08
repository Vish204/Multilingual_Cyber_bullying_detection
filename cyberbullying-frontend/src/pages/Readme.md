# 📂 Pages Directory: `src/pages/`

## 📌 Overview
The `pages/` directory houses the high-level React components associated with specific routes defined in `routes.jsx`. While the **Layouts** folder defines the structural skeleton, the **Pages** folder acts as the consumer of those layouts or as a container for page-specific business logic and route-level headers.

At this stage (Phase 4), the application is divided into a functional **Dashboard** and **Moderation** workspace, with future audit and insight tools currently in the "UI Shell" phase.

---

## 📄 Component Registry

### 1. `Dashboard.jsx`
The primary landing page of the application. It provides a "Bird's-eye View" of the entire system's health and acts as the entry portal for the moderator.

* **Orchestration:** Composes specialized components from `src/components/dashboard/` to create an information-dense landing.

* **Key Features:**
  * **Analytics Integration:** Renders `StatsOverview` to show current bullying trends and severe alert counts.
  * **Alert Visibility:** Hosts the `RecentAlerts` feed for immediate situational awareness of high-severity content.
  * **Navigation:** Acts as the launchpad for moderators through `NavigationCards`.

* **Logic:** Applies the `dashboard-container` class to manage global spacing and the dashboard-specific background.

---

### 2. `Moderation.jsx`
The functional heart of the "Human-in-the-Loop" system.

* **Layout Consumption:** Serves as a thin wrapper around **`ModerationLayout`**.
* **Intent:** By separating the page component from the layout logic, the 3-panel moderation view remains a reusable structure that can be repurposed for "Audit" or "Manager" views without duplicating core moderation logic.

---

### 3. `History.jsx` (Placeholder)

* **Current Status:** Renders a simple `<h1>History</h1>` tag.
* **Future Intent:** Will connect to the `/history` API to allow moderators to search, view, and potentially audit or reverse previous actions.

---

### 4. `Analysis.jsx` (Placeholder)

* **Current Status:** Renders a simple `<h1>Analysis</h1>` tag.
* **Future Intent:** Will host data visualizations (Charts/Graphs) regarding model accuracy, common toxicity categories, and platform-specific metrics.

---

## 🔄 Interaction Flow

1. **Dashboard Entry:** The user arrives at `Dashboard.jsx`, views high-level stats, and identifies "Severe" alerts.
2. **Transition:** The user navigates to the Moderation tab via the `NavigationCards`.
3. **Workspace Activation:** `ModerationLayout` mounts and prepares the workspace in an **idle state** (no data loaded).
4. **Data Ingestion:** The moderator clicks the **"Start Stream"** toggle. This activates the `isLive` state effect, triggering the sequential call to `collectData()` (backend scraping) and `fetchPosts()` (local UI population).
5. **Active Moderation:** Once the feed is populated, the user begins the click-to-analyze-to-action cycle.

---

## 🎨 Page-Specific Styling

* **Dashboard:** Uses `dashboard.css` for a card-based, centralized layout intended for analytical consumption.
* **Moderation:** Uses `moderation.css` (via the layout) for a flex-based, edge-to-edge layout designed for high-speed productivity and 3-panel multitasking.