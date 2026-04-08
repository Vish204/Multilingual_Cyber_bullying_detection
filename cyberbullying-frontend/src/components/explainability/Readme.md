# 📁 Sub-Folder: `src/components/explainability/`

## 📌 Purpose
The `explainability/` folder houses the **XAI (Explainable AI)** suite. This is a critical component of Phase 4, designed to build trust between the AI and the human moderator by revealing the "black box" of model decision-making.

---

## 🧩 Key Components

### `ShapHighlights.jsx`

* **Logic:**  
  Implements a token-based highlighting system. It splits the post text and applies a background color to specific **"High-Impact"** words (e.g., "stupid", "hate").

* **Note:**  
  Currently uses demo word-matching logic as a placeholder for real backend SHAP/Attention weights.

---

### `FeatureList.jsx`

* **Purpose:**  
  Renders the **"Top Influential Signals"** as a series of tags.

* **Functionality:**  
  Displays which specific words had the highest numerical impact on the bullying score.

---

### `ContextSummary.jsx`

* **Purpose:**  
  Provides a plain-language explanation of the AI's reasoning.

* **Example Output:**  
  _"Detected aggressive tone and direct insults"_  