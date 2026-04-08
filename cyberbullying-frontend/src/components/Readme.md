# 📂 Components Directory: `src/components/`

## 📌 Overview
The `components/` directory follows a **Feature-Based Modular Architecture**. Instead of a flat list of files, components are grouped by their functional domain (e.g., everything related to the "Feed" stays in the `feed/` folder).

This structure ensures that the project remains maintainable as the UI grows, allowing developers to locate and modify specific UI logic without navigating through unrelated code.

---

## 📁 Sub-Folder Registry

| Folder | Responsibility | Key Component Example |
| :--- | :--- | :--- |
| **`common/`** | Global, reusable UI utilities used across multiple pages. | `FilterBar.jsx` |
| **`dashboard/`** | High-level widgets for system health and navigation. | `StatsOverview.jsx` |
| **`feed/`** | Logic for the list-based ingestion of social content. | `FeedItem.jsx` |
| **`post/`** | Deep analysis and moderation controls for a single post. | `PostDetails.jsx` |
| **`explainability/`** | XAI visualizations (SHAP highlights and feature lists). | `ShapHighlights.jsx` |
| **`context/`** | Supplemental info and the Moderator's Guidebook. | `ContextPanel.jsx` |

---

## 🧬 Component Communication (Prop Patterns)

The components in this directory follow a **Stateless/Presentational** pattern wherever possible:

* **Inputs:** They receive data (posts, filters, stats) via **Props**.
* **Outputs:** They communicate user interactions (clicks, input changes) back to the Layout layer via **Callback Functions** (e.g., `onAction`, `onSelectPost`).

---

## 🎨 Design Philosophy

Every component in this directory adheres to the **Phase 4 Design System**:

1. **Visual Hierarchy:** Use of cards (`.card`) with subtle shadows to separate concerns.  
2. **Color Semantics:** Uniform use of the severity color palette (Red, Orange, Yellow, Green).  
3. **Responsiveness:** Use of Flexbox and CSS Grid within components to ensure they adapt to different panel widths.