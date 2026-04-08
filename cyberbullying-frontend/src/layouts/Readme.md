# 📂 Layouts Directory: `src/layouts/`

## 📌 Overview
The `layouts/` directory acts as the **structural and logical backbone** of the application. Unlike atomic components, layouts define the spatial organization of the UI and serve as the **primary orchestrator of state** for the moderation workflow.

The architecture follows a **Three-Panel Design**, facilitating a seamless transition from data ingestion (Left) to deep analysis (Middle) and policy reference (Right).

---

## 🏗️ Core Files

### `ModerationLayout.jsx`
This is the "Command Center" of the moderation workspace. It is a monolithic state controller that manages the lifecycle of social media content.

**Key Responsibilities:**

* **Master State Management:**
  * `feed`: The "Single Source of Truth" for all posts currently loaded in the UI.
  * `selectedPost`: Tracks which post is being audited in the middle panel.
  * `filters`: An object managing the criteria for real-time feed pruning.

* **Integrated Alert System:**
  * The **High-Priority Alert Banner** is implemented **inline** within this file.
  * It uses a `useEffect` hook to monitor the feed for `severe` posts and displays a sticky, interactive notification at the top of the viewport.

* **Action Orchestration (Immediate UI Updates):**
  * Implements `handleModeration` which triggers API calls via the service layer.
  * **Delete Action:** Removes the post from the local `feed` immediately (Optimistic removal).
  * **Ignore/Report/Save Actions:** Updates the post object's `reviewed`, `moderator_action`, or `saved` status in the local state immediately to provide instant visual feedback to the moderator.

* **Stream & Loading States:**
  * Manages the logic of "Live" simulation, handling the transition from `collectData()` to `fetchPosts()`.

---

### `moderation.css`
The architectural stylesheet. It defines the dashboard's personality using a **Flexbox-first approach** to create a non-scrolling, app-like experience.

**Structural Breakdown:**

* **`.main-content`**: The master flex container managing the 1:2:1 panel ratio.
* **`.panel`**: Shared styling for consistency (rounded corners, white backgrounds, subtle shadows).

* **Animations:**
  * `slideDown`: Handles the entry of the inline Alert Banner.
  * `fadeInOut`: Manages the lifecycle of the "Action Toasts" that appear after moderation decisions.

---

## 🔄 The 3-Panel Logic Flow

| Panel | Component Mapping | Primary Function |
| :--- | :--- | :--- |
| **Left** | `FeedList` + `FilterBar` | Ingestion, filtering, and selection of content. |
| **Middle** | `PostDetails` | AI Signal analysis, XAI visualization, and action execution. |
| **Right** | `ContextPanel` | Renders the **Moderation Guide** (Severity scales and emotion definitions) to ensure objective decision-making. |

---

## 🎨 Design Logic: Severity Borders

The layout uses a **Severity-Border System** defined in CSS. Each post in the feed is assigned a dynamic border color based on its normalized `severity` string:

* `none`: Green (`#22c55e`) — No action required.
* `mild`: Yellow (`#eab308`) — Low-level concern.
* `moderate`: Orange (`#f97316`) — Requires review.
* `severe`: Red (`#ef4444`) — High-priority/Dangerous content.

---

## 🛠️ Data Handling Nuances

* **Export Logic:**
  * The layout contains a specialized `handleExport` function that generates a CSV Blob locally from the `reviewed` posts, allowing moderators to download audit logs directly from the browser.

* **Context Components:**
  * While `RightPanel` exists as a component, the layout specifically utilizes **`ContextPanel`** to provide the detailed Moderation Guide used during active review.


One small detail: RightPanel is imported but not actually rendered; ContextPanel is the right-hand component in use.