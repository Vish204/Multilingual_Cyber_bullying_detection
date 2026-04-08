# 📂 Source Directory: `src/`

## 📌 Overview
The `src/` folder is the command center of the application. It follows a modular architecture that separates **routing**, **layout structure**, **business logic (services)**, and **UI components**. The architecture is designed to support a unidirectional data flow, primarily managed through the `layouts` layer.

---

## 🏗️ Core Architectural Files

### `main.jsx`
The entry point of the application. It initializes the React DOM and mounts the `App` component into the `index.html` root.

* **Wrapper:** Uses `StrictMode` to catch potential problems during development.
* **Global Styles:** Imports `index.css` to ensure consistent resets and scrollbar styling across the SPA.

### `App.jsx`
A clean entry component that acts as a wrapper for the routing logic. By keeping this file minimal, we ensure that global providers (like Theme or Auth) can be added easily in future phases.

### `routes.jsx`
The navigation map of the system using **React Router DOM 7.x**.

* **Paths:** Defines the endpoints for the `Dashboard`, `Moderation`, `History`, and `Analysis` pages.
* **Logic:** Uses `BrowserRouter` to enable clean URLs for the Single Page Application.

### `index.css`
The global stylesheet.

* **Resets:** Implements `box-sizing: border-box` and removes default margins/paddings.
* **Theming:** Sets the primary background color (`#eef2f7`) and defines the custom "Global Scrollbar" to match the modern dashboard aesthetic.

---

## 📁 Sub-Directory Breakdown

### 📂 `layouts/`
The "Brain" of the application. Unlike standard components, files here define the structural skeleton of the page.

* **`ModerationLayout.jsx`**: The most complex file in the project. It manages the **Master State**, including the feed data, the currently selected post, and the filtering logic.
* **`moderation.css`**: Contains the Flexbox logic for the 3-panel system and the color variables for severity (Safe, Mild, Moderate, Severe).

### 📂 `pages/`
Route-level components that represent full views.

* **`Dashboard.jsx`**: Orchestrates the landing experience using multiple dashboard-specific components.
* **`Moderation.jsx`**: A functional wrapper that injects the logic from `layouts`.
* **`History.jsx` & `Analysis.jsx`**: Currently placeholders, serving as the UI shell for future data visualization and audit logs.

### 📂 `services/`
The communication layer between the React UI and the FastAPI/Backend.

* **`api.js`**: Contains all `fetch` logic. It handles endpoints for fetching posts, triggering data collection, exporting CSVs, and sending moderation actions.
* **`transform.js`**: A critical **Data Normalization** layer. It maps inconsistent backend data (e.g., "high" vs "severe") into a standardized frontend schema and calculates relative time strings.

### 📂 `components/`
The atomic building blocks of the UI. This is subdivided by feature:

* **`common/`**: Reusable elements like the `FilterBar`.
* **`feed/`**: Components for the left panel (post list).
* **`post/`**: Components for the middle panel (AI analysis & actions).
* **`explainability/`**: XAI-specific components (SHAP highlights, Feature lists).
* **`dashboard/`**: Specialized widgets for the landing page (Stats, Flow diagrams).

### 📂 `hooks/` & `utils/`

* **Status:** Future-ready placeholders.
* **Intent:**  
  * `hooks/` will store reusable logic (e.g., `useAuth` or `useStream`)  
  * `utils/` will store pure functions (e.g., math helpers for AI scores)

---

## 🔄 Core Data Flow (The "src" Loop)

1. **Request:** A page (e.g., `Moderation`) mounts and calls a function in `services/api.js`.
2. **Transform:** The raw response is passed to `services/transform.js` to ensure the UI doesn't break due to missing or weirdly formatted backend data.
3. **State Update:** The transformed data is set into the state of a `layout` component.
4. **Distribution:** The layout passes this data down via **Props** to individual components in the `components/` folder for rendering.

---

## 🎨 Design System & Colors

The `src/` directory adheres to a strict color-coded system for AI verdicts:

* **Bullying (Severe):** `#ef4444` (Red)
* **Bullying (Moderate):** `#f97316` (Orange)
* **Bullying (Mild):** `#eab308` (Yellow)
* **Non-Bullying (None):** `#22c55e` (Green)