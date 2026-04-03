const BASE_URL = "http://localhost:8000";

// 🔹 Get posts
export async function fetchPosts() {
  const res = await fetch(`${BASE_URL}/history`);
  const data = await res.json();
  return data.data;
}

// 🔹 Moderate post
export async function moderatePost(id, action, reason = "") {
  const res = await fetch(`${BASE_URL}/moderate`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      id,
      action,
      reason, // 🔥 ADD
    }),
  });

  return res.json();
}