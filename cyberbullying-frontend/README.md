# 🚀 Cyberbullying Detection Frontend

## 📌 Overview

This is the **frontend UI** for the Cyberbullying Detection System.

It is designed as a **real-time moderation dashboard** that allows moderators to:

* View incoming social media content
* Analyze AI predictions (bullying, emotion, sarcasm)
* Understand model decisions (Explainable AI - SHAP)
* Take moderation actions (ignore, delete, report)
* Filter and manage content efficiently

---

## 🧱 Tech Stack

* React (UI Framework)
* Vite (Fast build tool)
* JavaScript (ES6+)
* CSS (custom styling)

---

## ⚙️ Project Setup

### 1. Create frontend using Vite

```bash
npm create vite@latest cyberbullying-frontend
```

Select:

* React
* JavaScript

---

### 2. Install dependencies

```bash
cd cyberbullying-frontend
npm install
```

---

### 3. Run development server

```bash
npm run dev
```

App runs at:

```
http://localhost:5173/
```

---

## ⚠️ Node.js Issue & Fix

### Problem:

Vite requires:

```
Node >= 20.19
```

### Your version:

```
v20.17.0 ❌
```

### Fix:

Update Node.js to latest (e.g. v24.x)

Then reinstall:

```bash
rm -rf node_modules package-lock.json
npm install
npm run dev
```

---

## 📁 Project Structure

### Root

```
cyberbullying-frontend/
│
├── index.html        # Entry HTML
├── package.json      # Dependencies & scripts
├── vite.config.js    # Vite config
├── public/           # Static assets
├── src/              # Main application code
```

---

## 📁 src/

```
src/
│
├── App.jsx           # Root component
├── main.jsx          # Entry point (React mount)
├── routes.jsx        # Page routing
│
├── pages/            # Page-level components
├── layouts/          # Layout structure (UI skeleton)
├── components/       # Reusable UI components
├── services/         # API calls (future)
├── hooks/            # Custom hooks
├── utils/            # Helper functions
```

---

## 📁 pages/

```
pages/
│
├── Dashboard.jsx
├── Moderation.jsx
├── History.jsx
├── Analysis.jsx
```

👉 Each file represents a **full page**

---

## 📁 layouts/

```
layouts/
│
├── ModerationLayout.jsx   # Main 3-panel layout
├── moderation.css         # Styling
```

👉 Layout defines **structure of the page**

---

## 📁 components/

### 🔹 common/

Reusable UI

* FilterBar.jsx → filtering system

---

### 🔹 feed/

Left panel

* FeedList.jsx → list of posts
* FeedItem.jsx → single post

---

### 🔹 post/

Middle panel

* PostDetails.jsx → main content
* PredictionCard.jsx → AI output
* EmotionTag.jsx → emotion display
* SarcasmIndicator.jsx → sarcasm score
* SeverityBadge.jsx → severity color
* ActionButtons.jsx → moderation actions

---

### 🔹 explainability/

Explainable AI

* ShapHighlights.jsx → word highlighting
* FeatureList.jsx → top features
* ContextSummary.jsx → reasoning

---

### 🔹 context/

Right panel

* RightPanel.jsx → extra info

---

## 🧠 Core Feature: Live Moderation Page

This is the **main working page**.

### Layout:

```
[ Feed ] | [ Post Details ] | [ Context ]
```

---

## 🔹 LEFT PANEL (Feed)

* Shows posts
* Click → selects post
* Uses filters

---

## 🔹 MIDDLE PANEL (PostDetails)

Displays:

* Full text
* AI prediction
* Emotion & sarcasm
* Explainability (SHAP)
* Moderation buttons

---

## 🔹 RIGHT PANEL (Context)

* Additional info (lightweight)

---

## ⚙️ Moderation Actions

* Ignore
* Delete
* Report

👉 Updates feed dynamically

---

## 🔍 Filtering System

### Basic Filters

* Platform
* Severity
* Language
* Search

---

### Advanced Filters (Toggle)

* Reviewed
* Alert
* Content Type
* Moderator Action

---

## 🧠 Key React Concepts Used

### 1. Lifting State Up

State stored in:

```
ModerationLayout.jsx
```

Shared between:

* FeedList
* PostDetails

---

### 2. Props

Data passed via:

```
onSelectPost
onAction
```

---

### 3. Conditional Rendering

```jsx
{showAdvanced && <AdvancedFilters />}
```

---

### 4. Event-driven UI

* Click → update state → UI updates

---

## 🚀 Future Integration

Frontend will connect to backend APIs:

* `/collect` → fetch data
* `/moderate` → actions
* `/history` → filtering
* `/analysis` → analytics

---

## 🎯 Goal

This UI is designed to demonstrate:

* Real-time moderation workflow
* Multilingual detection
* Emotion-aware AI
* Explainable AI (SHAP)
* Efficient inference (distilled models)

---

## 🧾 Summary

This frontend is:

* Modular
* Scalable
* Research-aligned
* Product-level design

---





# React + Vite

This template provides a minimal setup to get React working in Vite with HMR and some ESLint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Oxc](https://oxc.rs)
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/)

## React Compiler

The React Compiler is not enabled on this template because of its impact on dev & build performances. To add it, see [this documentation](https://react.dev/learn/react-compiler/installation).

## Expanding the ESLint configuration

If you are developing a production application, we recommend using TypeScript with type-aware lint rules enabled. Check out the [TS template](https://github.com/vitejs/vite/tree/main/packages/create-vite/template-react-ts) for information on how to integrate TypeScript and [`typescript-eslint`](https://typescript-eslint.io) in your project.
