# 📁 Sub-Folder: `src/components/common/`

## 📌 Purpose
The `common/` folder contains components that are "utility players"—reusable UI elements that provide global functionality across the application. Currently, it houses the `FilterBar`, which is the primary tool for data management in the dashboard.

---

## 🧩 Key Components

### `FilterBar.jsx`

* **Logic:**  
  Manages a hybrid state. It receives the global `filters` object as a prop but maintains its own local visibility states (`showAdvanced`, `showSearch`) to keep the UI clean.

* **Search Interaction:**  
  Uses `onBlur` and `onKeyDown` (Escape) listeners to provide a smooth, keyboard-friendly search experience.

* **Multi-Select:**  
  Provides dropdowns for Platform, Severity, Language, and "Reviewed" status.

---

## 🔄 Data Flow

The `FilterBar` does not filter the data itself. It acts as an input collector.

* When a user changes a dropdown, it calls `setFilters` (passed from the Layout).
* This triggers a re-render and filtering logic at the **Layout level**, ensuring a centralized and consistent filtering mechanism.