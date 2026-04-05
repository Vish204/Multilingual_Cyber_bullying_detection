// export function transformPost(apiPost) {
//   return {
//     id: apiPost._id,

//     text: apiPost.text,
//     platform: apiPost.platform || "Unknown",

//     time: apiPost.timestamp, // keep raw for now

//     severity: apiPost.severity || "none",

//     language: apiPost.language?.name || "Unknown",

//     reviewed: apiPost.reviewed || false,
//     alert: apiPost.alert || false,
//     content_type: apiPost.content_type || "text",
//     moderator_action: apiPost.moderator_action || null,

//     verdict:
//       apiPost.label === "normal"
//         ? "NON-BULLYING"
//         : "BULLYING",

//     confidence: apiPost.confidence / 100,

//     // 🔥 Top emotion
//     emotion: apiPost.emotions?.[0]?.label || "neutral",
//     emotion_score: (apiPost.emotions?.[0]?.score || 0) / 100,

//     sarcasm: apiPost.sarcasm, // 🔥 KEEP 0–100

//     saved: false,
//   };
// }


export function transformPost(post) {
  return {
    id: post._id,

    text: post.text,

    // ✅ PLATFORM FIX
    platform: normalizePlatform(post.platform),

    // ✅ TIME FORMAT
    time: formatTime(post.timestamp),

    // ✅ SEVERITY FIX
    severity: normalizeSeverity(post.severity),

    // ✅ LANGUAGE FIX
    language: post.language?.name
      ? capitalize(post.language.name)
      : "Unknown",

    reviewed: post.reviewed || false,
    alert: post.alert || false,
    content_type: post.content_type || "text",
    moderator_action: post.moderator_action || null,

    // ✅ VERDICT FIX
    // verdict:
    //   post.label === "bullying"
    //     ? "BULLYING"
    //     : "NON-BULLYING",
    verdict:
      normalizeSeverity(post.severity) === "none"
        ? "NON-BULLYING"
        : "BULLYING",

    confidence: (post.confidence || 0) / 100,

    // 🔥 TOP EMOTION
    emotion: getTopEmotion(post.emotions),
    emotion_score: getTopEmotionScore(post.emotions),

    // ✅ KEEP 0–1 (frontend expects this)
    sarcasm: (post.sarcasm || 0) / 100,

    saved: false,
  };
}

const normalizePlatform = (platform) => {
  if (!platform) return "Unknown";

  const p = platform.toLowerCase();

  if (p.includes("twitter")) return "Twitter";
  if (p.includes("youtube")) return "YouTube";
  if (p.includes("reddit")) return "Reddit";

  return "Manual";
};

const normalizeSeverity = (severity) => {
  if (!severity) return "none";

  const s = severity.toLowerCase();

  if (s === "none") return "none";
  if (s === "low") return "mild";
  if (s === "medium") return "moderate";
  if (s === "high") return "severe";

  return s;
};

const capitalize = (str) =>
  str.charAt(0).toUpperCase() + str.slice(1);

const formatTime = (timestamp) => {
  if (!timestamp) return "";

  const date = new Date(timestamp);
  const now = new Date();

  const diff = Math.floor((now - date) / 1000);

  if (diff < 60) return `${diff}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;

  return date.toLocaleDateString();
};

const getTopEmotion = (emotions) => {
  if (!emotions || emotions.length === 0) return "neutral";
  return emotions[0].label;
};

const getTopEmotionScore = (emotions) => {
  if (!emotions || emotions.length === 0) return 0;
  return emotions[0].score / 100;
};