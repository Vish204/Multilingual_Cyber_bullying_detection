# 📁 Sub-Folder: `src/components/context/`

## 📌 Purpose
The `context/` folder provides the **Right Panel** content. It serves as a "Policy Reference" to ensure that human moderation remains consistent and aligned with established safety guidelines.

---

## 🧩 Key Components

### `ContextPanel.jsx`

* **The "Moderator's Handbook":**  
  This is the primary component rendered in the right panel of the `ModerationLayout`.

* **Guidance Logic:**  
  Provides static definitions for:
  * Severity levels (None → Severe)
  * Emotion categories (Aggression, Distress)

  This helps reduce subjective bias and ensures consistency during the moderation process.

---

### `RightPanel.jsx`

* **Status:**  
  This component exists in the file structure but is **not currently rendered** in the main application flow.

* **Future Intent:**  
  Serves as a placeholder for future features such as:
  * Quick Stats
  * Session-specific metadata