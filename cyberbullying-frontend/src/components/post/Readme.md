# 📁 Sub-Folder: `src/components/post/`

## 📌 Purpose
The `post/` folder handles the **Middle Panel**, which is the primary workspace for a moderator. This module is responsible for synthesizing all AI signals (Verdict, Confidence, Emotion, Sarcasm) and providing the interface for taking action.

---

## 🧩 Key Components

### `PostDetails.jsx`
The orchestrator of the middle panel.

* **Conditional Rendering:**  
  Dynamically renders different sections (XAI, AI Signals, Actions) based on the post's verdict.

* **Empty State Handling:**  
  Displays a placeholder UI when no post is selected.

---

### `PredictionCard.jsx`
A visual summary of the AI's certainty.

* **Confidence Bar:** Displays the overall model confidence.
* **Signals Row:** Groups:
  * `EmotionTag`
  * `SarcasmIndicator`  
  into a unified, easy-to-read section.

---

### `ActionButtons.jsx`
The interface for moderation decisions.

* **Logic:**  
  Implements an interactive **"Report" flow**:
  * Reveals a reason input field only when the moderator clicks **Report**
  * Keeps the UI clean and minimal until action is required

---

### `ConfidenceBreakdown.jsx`
Provides a granular explanation of the final score.

* Shows how:
  * Base Model
  * Emotion Signal
  * Sarcasm Signal  
  contributed to the final confidence percentage.

---

### `EmotionTag.jsx` & `SarcasmIndicator.jsx`
Compact, color-coded badges for quick interpretation.

* **Visual Cues:** Use emojis and percentages for at-a-glance understanding.
* **Purpose:** Reduce cognitive load and enable faster moderation decisions.