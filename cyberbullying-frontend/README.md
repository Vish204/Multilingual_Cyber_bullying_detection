# 🛡️ Cyberbullying Detection System - Frontend

## 📌 Project Overview
This is the **Frontend User Interface** for the AI-powered Cyberbullying Detection & Moderation System. It is designed as a specialized dashboard that bridges the gap between complex Machine Learning models and human moderators.

The application currently supports a **"Human-in-the-Loop"** workflow, providing a UI framework for moderators to interpret AI decisions through **Explainable AI (XAI)** signals and execute platform safety actions.

### 🎯 Key Objectives
* **Transparency:** A UI framework to show moderators *why* a post was flagged using SHAP-inspired token highlighting.
* **Signal Integration:** Visualization of multi-model outputs including Bullying Verdicts, Emotion detection (Aggression/Distress), and Sarcasm.
* **Workflow Efficiency:** A specialized 3-panel layout designed for rapid content review.
* **Modular Design:** A clean directory structure allowing for the future integration of various NLP backends and social platform data.

---

## 🧱 Tech Stack
* **Framework:** `React 19.x` (Functional Components & Hooks)
* **Build Tool:** `Vite 8.x`
* **Routing:** `React Router DOM 7.x`
* **Icons:** `react-icons` (Social platforms & UI) & `lucide-react`
* **Styling:** Custom CSS3 with flex-based architecture and severity-coded color palettes.
* **Language:** JavaScript (ES6+)

---

## ⚙️ Project Setup

### 1. Prerequisites
* **Node.js:** `>= 20.19.0` (Vite requirement)
* **Backend:** Ensure the Cyberbullying API is accessible (default `http://localhost:8000`)

### 2. Installation
```bash
# Enter the project directory
cd cyberbullying-frontend

# Install dependencies
npm install
````

### 3. Run Development Server

```bash
npm run dev
```

The application runs by default at: `http://localhost:5173/`

---

## ⚠️ Node.js Versioning Note

If you encounter errors during `npm run dev`, ensure your Node version is updated.

* **Requirement:** `Node >= 20.19`
* **Recommended:** `v24.x` (Latest)

---

## 📁 Project Architecture

### 📂 Root Directory

| File/Folder      | Description                                                        |
| :--------------- | :----------------------------------------------------------------- |
| `index.html`     | The Single Page Application (SPA) entry point.                     |
| `package.json`   | Project metadata, scripts, and `React 19` / `Vite 8` dependencies. |
| `vite.config.js` | Vite-specific build and plugin configuration.                      |
| `src/`           | The primary application source code.                               |

### 📂 The `src/` Directory Breakdown

```text
src/
├── App.jsx           # Entry component (Renders AppRoutes)
├── main.jsx          # React DOM entry point
├── routes.jsx        # Route definitions for the SPA
│
├── pages/            # View-level components (Dashboard, Moderation)
├── layouts/          # Structural UI skeletons (Moderation 3-panel layout)
├── components/       # UI Molecules (Common, Post, Feed, Explainability)
├── services/         # API logic (api.js) and Data Normalization (transform.js)
├── hooks/            # [Future-Ready] Placeholder for custom hooks
└── utils/            # [Future-Ready] Placeholder for utility helpers
```

---

## 🧠 Core Feature: Moderation Workflow (Phase 4)

The **Moderation Page** utilizes a **3-panel architecture** to manage the content review cycle:

### 1. [LEFT] Feed Panel (`components/feed`)

* **Manual Fetch/Toggle:** Features a "Start Stream" toggle state intended to trigger batch data collection.
* **Filtering:** A robust system to sort content by Platform, Severity, Language, and Keyword Search.
* **Selection:** Contextual selection logic where clicking a post populates the detail view.

### 2. [MIDDLE] Post Details (`components/post` & `explainability`)

* **AI Analysis:** Displays the Verdict (BULLYING/SAFE) and Confidence bars derived from the Student XGBoost model.
* **XAI UI:** Visual implementation of **SHAP-inspired highlights** and feature influence lists (currently utilizing demo data for visualization).
* **Action Center:** Moderators can **Ignore**, **Delete**, or **Report** (Report includes an optional input field for reasoning).

### 3. [RIGHT] Context Panel (`components/context`)

* **Moderation Guide:** Static reference system explaining severity scales and emotion labels to ensure moderator consistency.

---

## 🔄 Data Handling & API Integration

* **State Ownership:** `ModerationLayout.jsx` acts as the primary controller, holding the master `feed` state.
* **Transformation Layer:** `services/transform.js` normalizes backend responses (e.g., converting 0-100 scores to 0-1 percentages) before they reach UI components.
* **API Communication:** `services/api.js` manages asynchronous requests to backend endpoints including `/history`, `/moderate`, and `/collect`.

---

## 🎯 Current Project Status

* [x] **Core Dashboard:** Operational high-level overview and navigation cards.
* [x] **3-Panel Layout:** Completed flex-based structural implementation.
* [x] **AI Signal UI:** Finished visualization for confidence, emotion, and sarcasm.
* [!] **History & Analysis:** Currently **Placeholder pages** (UI headers only).
* [!] **Live Ingestion:** Stream toggle is implemented in the UI; backend-wide live ingestion integration is aspirational for future phases.

---

## 🧾 Summary

This frontend is a **Modular** and **Research-aligned** dashboard. It is intentionally designed with future-ready folders (`hooks`, `utils`) and a scalable component library, making it a robust foundation for product-level AI moderation.







# React + Vite

This template provides a minimal setup to get React working in Vite with HMR and some ESLint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Oxc](https://oxc.rs)
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/)

## React Compiler

The React Compiler is not enabled on this template because of its impact on dev & build performances. To add it, see [this documentation](https://react.dev/learn/react-compiler/installation).

## Expanding the ESLint configuration

If you are developing a production application, we recommend using TypeScript with type-aware lint rules enabled. Check out the [TS template](https://github.com/vitejs/vite/tree/main/packages/create-vite/template-react-ts) for information on how to integrate TypeScript and [`typescript-eslint`](https://typescript-eslint.io) in your project.
