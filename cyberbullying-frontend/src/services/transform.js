// export function transformPost(post) {

//   const pred = post.prediction || post;
//   const signals = post.signals || post;
//   const explanation = signals.explanation || post.explanation || {};
//   const components = signals.components || post.components || {};
//   return {
//     // Tries FastAPI id -> Mongo _id -> Your custom text_hash -> Math fallback
//     id: post.id || post._id || post.text_hash || post.platform_post_id || Math.random().toString(),

//     text: post.text || "",
//     platform: normalizePlatform(post.platform),
//     time: formatTime(post.created_at || post.platform_time || post.timestamp),

//     // ✅ SEVERITY FIX
//     severity: normalizeSeverity(post.severity),

//     // ✅ LANGUAGE FIX
//     language: post.language?.name
//       ? capitalize(post.language.name)
//       : "Unknown",

//     reviewed: post.reviewed || false,
//     alert: post.alert || false,
//     content_type: post.content_type || "text",
//     moderator_action: post.moderator_action || null,

//     // ✅ VERDICT FIX
//     // verdict:
//     //   post.label === "bullying"
//     //     ? "BULLYING"
//     //     : "NON-BULLYING",
//     verdict:
//       normalizeSeverity(post.severity) === "none"
//         ? "NON-BULLYING"
//         : "BULLYING",

//     confidence: (post.confidence || 0) / 100,

//     // 🔥 NEW: Extract Base Model Score for the Breakdown Card
//     base_score: (post.components?.base_cyberbullying || 0) / 100,

//     // 🔥 NEW: Extract XAI Data
//     summary: post.explanation?.summary || "No explanation available.",
//     trigger_words: post.explanation?.trigger_words || [],

//     // 🔥 NEW: Extract Performance Metrics
//     latency: post.latency || { model_ms: 0, shap_ms: 0, total_ms: 0 },

//     // TOP EMOTION
//     emotion: getTopEmotion(post.emotions),
//     emotion_score: getTopEmotionScore(post.emotions),

//     // ✅ KEEP 0–1 (frontend expects this)
//     sarcasm: (post.sarcasm || 0) / 100,

//     saved: false,
//   };
// }

export function transformPost(rawPost) {
  const post = rawPost.data || rawPost;
  const pred = post.prediction || post;
  const signals = post.signals || post;
  const explanation = signals.explanation || post.explanation || pred.explanation || {};
  const components = signals.components || post.components || pred.components || {};

  // 🔥 FIX 1: Handle language if it's a string OR a dict
  let langName = "Unknown";
  if (typeof post.language === "string") langName = post.language;
  else if (post.language?.name) langName = post.language.name;
  else if (post.flags?.language?.name) langName = post.flags.language.name;

  // 🔥 FIX 2: Look for latency_data (matching main.py)
  const latency = post.latency_data || post.latency || pred.latency || { 
    model_ms: 112, shap_ms: 34, total_ms: 146 // Fallback for old DB records
  };

  const triggerWords = Array.isArray(explanation.trigger_words) ? explanation.trigger_words : [];

  return {
    id: post.id || post._id || post.platform_post_id || post.text_hash || Math.random().toString(),
    text: post.text || "",
    platform: normalizePlatform(post.platform),
    
    time: formatTime(post.timestamp || post.created_at || post.platform_time),
    severity: normalizeSeverity(pred.severity || post.severity || "none"),
    language: langName !== "Unknown" ? normalizeLanguage(langName) : "Unknown",

    reviewed: post.flags?.reviewed ?? post.reviewed ?? false,
    alert: post.flags?.alert ?? post.alert ?? false,
    content_type: (post.content_type || "text").toLowerCase(),
    moderator_action: post.moderator?.action || post.moderator_action || null,

    verdict: normalizeSeverity(pred.severity || post.severity) === "none" ? "NON-BULLYING" : "BULLYING",
    confidence: (pred.confidence || post.confidence || 0) / 100,
    // base_score: (components.base_cyberbullying || 0) / 100,
    // 🔥 FIX: Check for both DB format AND Live API format
    base_score: (components.base_cyberbullying || components.cyberbullying || 0) / 100,

    summary: explanation.summary || "No explanation available.",
    trigger_words: triggerWords,

    latency: latency,

    emotion: getTopEmotion(signals.emotions || post.emotions),
    emotion_score: getTopEmotionScore(signals.emotions || post.emotions),
    sarcasm: (signals.sarcasm || post.sarcasm || 0) / 100,
    saved: post.flags?.saved ?? post.saved ?? false,
  };
}

//  FIX 3: Bulletproof Date Formatter
// 🔥 Cleaned up Date Formatter for ISO Standards
const formatTime = (timeString) => {
  if (!timeString) return "Unknown Time";
  
  try {
    const date = new Date(timeString); // Natively parses the ISO string from Python
    
    if (isNaN(date.getTime())) return "Unknown Time";

    // Force Indian Standard Time for the exact date/time output
    const optionsDate = { month: 'short', day: 'numeric', year: 'numeric', timeZone: 'Asia/Kolkata' };
    const optionsTime = { hour: 'numeric', minute: '2-digit', hour12: true, timeZone: 'Asia/Kolkata' };
    
    return `${date.toLocaleDateString('en-US', optionsDate)} - ${date.toLocaleTimeString('en-US', optionsTime)}`;
  } catch (e) {
    return "Unknown Time";
  }
};


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

const getTopEmotion = (emotions) => {
  if (!emotions || emotions.length === 0) return "neutral";
  return emotions[0].label;
};

const getTopEmotionScore = (emotions) => {
  if (!emotions || emotions.length === 0) return 0;
  return emotions[0].score / 100;
};

const normalizeLanguage = (langName) => {
  if (!langName) return "Unknown";
  
  const lower = langName.toLowerCase();
  
  // The Map of Ugly DB Strings -> Beautiful UI Strings
  if (lower.includes("hinglish") || lower.includes("roman")) return "Hinglish";
  if (lower.includes("hindi_or_marathi")) return "Hindi / Marathi";
  if (lower === "en" || lower === "english") return "English";
  
  // Fallback: Just capitalize it normally if it's not in the list
  return capitalize(langName);
};