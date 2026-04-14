const BASE_URL = "http://localhost:8000";

// 🔹 Get posts
export async function fetchPosts() {
  const res = await fetch(`${BASE_URL}/history?limit=15`);
  const data = await res.json();
  return data.data;
}

// 🔹 Moderate post
export async function moderatePost(id, action, reason = "", saved = false) {
  const res = await fetch(`${BASE_URL}/moderate`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      id,
      action,
      reason, 
      saved // 🔥 NEW: Tell backend if we are saving it for retraining
    }),
  }); return res.json();
}

export async function fetchSeverityStats() {
  const res = await fetch(`${BASE_URL}/analytics/severity`);
  return res.json();
}

export async function fetchAlerts() {
  const res = await fetch(`${BASE_URL}/history?alert=true`);
  return res.json();
}


export async function exportPosts(filters = {}) {
  const cleanedFilters = {};

  Object.keys(filters).forEach((key) => {
    const value = filters[key];
    if (value === null || value === undefined || value === "" || value === "all") {
      return;
    }
    cleanedFilters[key] = value;
  });

  const query = new URLSearchParams(cleanedFilters).toString();

  // 🔥 FIX: Point to the new /export/view endpoint
  const res = await fetch(`${BASE_URL}/export/view?${query}`);

  const blob = await res.blob();
  return blob;
}


export async function collectData() {
  const res = await fetch(`${BASE_URL}/collect`);
  return res.json();
}