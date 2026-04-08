# 📁 Sub-Folder: `src/components/feed/`

## 📌 Purpose
The `feed/` folder is responsible for the **Left Panel** of the moderation dashboard. Its job is to render a high-performance list of incoming content and indicate which items require the most urgent attention.

---

## 🧩 Key Components

### `FeedList.jsx`

* **Logic:**  
  Acts as a wrapper for the list. It includes a **"Safety Check"** (conditional rendering) to display an "empty state" message if no posts match the current filters.

* **Mapping:**  
  Iterates through the `feed` array to produce `FeedItem` components.

---

### `FeedItem.jsx`

* **Visual Priority:**  
  Uses the **Severity-Border System** (4px left border) to visually flag content without the user needing to read the text first.

* **Dynamic Icons:**  
  Uses `react-icons` to identify the source platform (YouTube, Reddit, Twitter) and emojis for quick emotion sensing:  
  * 😡 Aggression  
  * 😢 Distress  

* **Status Indicators:**  
  Displays small badges for:  
  * ✔ Reviewed  
  * 💾 Saved  
  This helps prevent redundant work by moderators.

---

## 🎨 Visual State Mapping

| Severity  | Border Color | CSS Class |
| :-------- | :----------- | :-------- |
| Severe    | `#ef4444`    | `.severe` |
| Moderate  | `#f97316`    | `.moderate` |
| Mild      | `#eab308`    | `.mild` |
| None      | `#22c55e`    | `.none` |