import PredictionCard from "./PredictionCard";
import ShapHighlights from "../explainability/ShapHighlights";
import FeatureList from "../explainability/FeatureList";
import ContextSummary from "../explainability/ContextSummary";
import ActionButtons from "./ActionButtons";
import ConfidenceBreakdown from "./ConfidenceBreakdown";
// 🔥 NEW: Import icons for the middle panel!
import { FaTwitter, FaYoutube, FaReddit, FaClock, FaGlobe } from "react-icons/fa";

export default function PostDetails({ post, onAction }) {
  if (!post) {
    return <p className="empty-state">Select a post to view details</p>;
  }

  const handleAction = (action, reason = "") => {
    onAction(action, post.id, reason);
  };

  // Helper to grab the right icon and color for the metadata pill
  const getPlatformIcon = () => {
    if (post.platform.toLowerCase() === "twitter") return <FaTwitter color="#1DA1F2" size={14} />;
    if (post.platform.toLowerCase() === "youtube") return <FaYoutube color="#FF0000" size={14} />;
    if (post.platform.toLowerCase() === "reddit") return <FaReddit color="#FF4500" size={14} />;
    return null;
  };

  return (
    <div className="post-container">

      {/* 🔥 1. HEADER (USING YOUR GLOBAL CSS CLASSES!) */}
      <div className={`card ${(post.severity || "none").toLowerCase()}`}>
        <div className="decision-header">
          <div className={`verdict-box ${post.verdict === "BULLYING" ? "bad" : "good"}`}>
            <div className="verdict-text">{post.verdict}</div>
          </div>

          <div style={{ fontSize: "12px", color: "#6b7280", fontWeight: "600", background: "#e5e7eb", padding: "6px 12px", borderRadius: "999px", display: "flex", gap: "8px" }}>
            <span>⚡ Total: {post.latency.total_ms}ms</span>
            <span style={{opacity: 0.5}}>|</span>
            <span>SHAP: {post.latency.shap_ms}ms</span>
          </div>
        </div>

        <div className="post-meta">
          <span className="meta-pill">
            {getPlatformIcon()}
            <span style={{ textTransform: "capitalize" }}>{post.platform}</span>
          </span>

          <span className="meta-pill">
            <FaClock color="#94a3b8" size={12} />
            {post.time}
          </span>

          <span className="meta-pill">
            <FaGlobe color="#3974c6" size={12} />
            {post.language}
          </span>

          {/* 🔥 Uses your existing .severity-severe, .severity-mild CSS classes! */}
          <span className={`severity-badge severity-${(post.severity || "none").toLowerCase()}`} style={{ marginLeft: "0" }}>
            {post.severity || "NONE"}
          </span>
        </div>
      </div>

      {/* 🔥 2. TEXT WITH XAI HIGHLIGHT */}
      <div className="card">
        <h3 className="section-title">{post.content_type || "Content"}</h3>
        <div className="highlight-text">
          <ShapHighlights text={post.text} triggerWords={post.trigger_words} />
        </div>
        
        {post.verdict === "BULLYING" && (
          <p className="xai-hint">Highlighted words indicate AI attention</p>
        )}
        {post.verdict !== "BULLYING" && (
          <p className="safe-text">No harmful patterns detected</p>
        )}
      </div>

      {/* 🔥 3. AI SIGNALS */}
      <div className="card">
        <h3 className="section-title">AI Signals</h3>
        <PredictionCard data={post} />
        <div className="inner-divider"></div>
        <ConfidenceBreakdown data={post} />
      </div>

      {/* 🔥 4. EXPLAINABILITY (ONLY FOR BULLYING) */}
      {post.verdict === "BULLYING" && (
        <div className="card">
          <h3 className="section-title">Why Flagged</h3>
          <div className="xai-section">
            <FeatureList triggerWords={post.trigger_words} />
            <ContextSummary summary={post.summary} />
          </div>
        </div>
      )}

      {/* 🔥 5. ACTIONS */}
      <div className="card">
        {/* <h3 className="section-title">Moderator Actions</h3> */}
        <ActionButtons onAction={handleAction} />
      </div>

    </div>
  );
}


































// import PredictionCard from "./PredictionCard";
// import ShapHighlights from "../explainability/ShapHighlights";
// import FeatureList from "../explainability/FeatureList";
// import ContextSummary from "../explainability/ContextSummary";
// import ActionButtons from "./ActionButtons";
// import ConfidenceBreakdown from "./ConfidenceBreakdown";

// export default function PostDetails({ post, onAction }) {
//   if (!post) {
//     return <p className="empty-state">Select a post to view details</p>;
//   }

//     const handleAction = (action, reason = "") => {
//       onAction(action, post.id, reason);
//     };

//   return (
//     <div className="post-container">

//       {/* 🔥 1. HEADER (COMPACT DECISION BAR) */}
//       {/* <div className="card">
//         <div className="decision-header">

//           <div className={`verdict-box ${post.verdict === "BULLYING" ? "bad" : "good"}`}>
//             <div className="verdict-text">{post.verdict}</div>
//             {/* <div className="verdict-confidence">
//               {(post.confidence * 100).toFixed(0)}%
//             </div> */}
//           {/* </div>

//         </div>

//         <div className="post-meta">
//           <span>{post.platform}</span>
//           <span>{post.time}</span>
//           <span>{post.language}</span>
//           <span className={`severity ${post.severity}`}>
//             {post.severity.toUpperCase()}
//           </span>
//         </div>
//       </div> */}

//       {/* 🔥 2. TEXT WITH XAI HIGHLIGHT */}
//       {/* <div className="card">
//       <h3 className="section-title">
//         {post.content_type}
//       </h3>
//         <div className="highlight-text">
//           <ShapHighlights text={post.text} />
//         </div>
//         {post.verdict === "BULLYING" && (
//           <p className="xai-hint">
//             Highlighted words indicate AI attention
//           </p>
//         )}
//         {post.verdict !== "BULLYING" && (
//           <p className="safe-text">
//             No harmful patterns detected
//           </p>
//         )}
//       </div> */} 

//       {/* 🔥 1. HEADER (COMPACT DECISION BAR) */}
//       <div className="card">
//         <div className="decision-header">
//           <div className={`verdict-box ${post.verdict === "BULLYING" ? "bad" : "good"}`}>
//             <div className="verdict-text">{post.verdict}</div>
//           </div>

//           {/* ⚡ NEW: The Performance Flex Badge */}
//           <div style={{ fontSize: "12px", color: "#6b7280", fontWeight: "600", background: "#e5e7eb", padding: "6px 12px", borderRadius: "999px", display: "flex", gap: "8px" }}>
//             <span>⚡ Total: {post.latency.total_ms}ms</span>
//             <span style={{opacity: 0.5}}>|</span>
//             <span>SHAP: {post.latency.shap_ms}ms</span>
//           </div>
//         </div>

//         <div className="post-meta">
//           <span>{post.platform}</span>
//           <span>{post.time}</span>
//           <span>{post.language}</span>
//           <span className={`severity ${post.severity}`}>
//             {post.severity.toUpperCase()}
//           </span>
//         </div>
//       </div>

//       {/* 🔥 2. TEXT WITH XAI HIGHLIGHT */}
//       <div className="card">
//         <h3 className="section-title">{post.content_type}</h3>
//         <div className="highlight-text">
//           {/* 🔥 FIX: We must pass triggerWords down here! */}
//           <ShapHighlights text={post.text} triggerWords={post.trigger_words} />
//         </div>
        
//         {post.verdict === "BULLYING" && (
//           <p className="xai-hint">Highlighted words indicate AI attention</p>
//         )}
//         {post.verdict !== "BULLYING" && (
//           <p className="safe-text">No harmful patterns detected</p>
//         )}
//       </div>

//       {/* 🔥 3. AI SIGNALS (CLEAN VIEW) */}
//       <div className="card">
//         <h3 className="section-title">AI Signals</h3>

//         <PredictionCard data={post} />

//           <div className="inner-divider"></div>
//             <ConfidenceBreakdown data={post} />
//           </div>


//         {/* 🔥 4. EXPLAINABILITY (ONLY FOR BULLYING) */}
//         {post.verdict === "BULLYING" && (
//           <div className="card">
//             <h3 className="section-title">Why flagged</h3>

//             <div className="xai-section">
//               <FeatureList triggerWords={post.trigger_words} />
//               <ContextSummary summary={post.summary} />
//             </div>
//           </div>
//         )}

//       {/* 🔥 5. ACTIONS */}
//       <div className="card">
//         <ActionButtons onAction={handleAction} />
//       </div>

//     </div>
//   );
// }