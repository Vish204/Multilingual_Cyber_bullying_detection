export function transformPost(apiPost) {
  return {
    id: apiPost._id,

    text: apiPost.text,
    platform: apiPost.platform || "Unknown",

    time: apiPost.timestamp, // keep raw for now

    severity: apiPost.severity || "none",

    language: apiPost.language?.name || "Unknown",

    reviewed: apiPost.reviewed || false,
    alert: apiPost.alert || false,
    content_type: apiPost.content_type || "text",
    moderator_action: apiPost.moderator_action || null,

    verdict:
      apiPost.label === "normal"
        ? "NON-BULLYING"
        : "BULLYING",

    confidence: apiPost.confidence / 100,

    // 🔥 Top emotion
    emotion: apiPost.emotions?.[0]?.label || "neutral",
    emotion_score: (apiPost.emotions?.[0]?.score || 0) / 100,

    sarcasm: apiPost.sarcasm, // 🔥 KEEP 0–100

    saved: false,
  };
}