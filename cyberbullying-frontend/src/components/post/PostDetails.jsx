import PredictionCard from "./PredictionCard";
import ShapHighlights from "../explainability/ShapHighlights";
import FeatureList from "../explainability/FeatureList";
import ContextSummary from "../explainability/ContextSummary";
import ActionButtons from "./ActionButtons";
import ConfidenceBreakdown from "./ConfidenceBreakdown";

export default function PostDetails({ post, onAction }) {
  if (!post) {
    return <p className="empty-state">Select a post to view details</p>;
  }

    const handleAction = (action, reason = "") => {
      onAction(action, post.id, reason);
    };

  return (
    <div className="post-container">

      {/* 🔥 1. HEADER (COMPACT DECISION BAR) */}
      {/* <div className="card">
        <div className="decision-header">

          <div className={`verdict-box ${post.verdict === "BULLYING" ? "bad" : "good"}`}>
            <div className="verdict-text">{post.verdict}</div>
            {/* <div className="verdict-confidence">
              {(post.confidence * 100).toFixed(0)}%
            </div> */}
          {/* </div>

        </div>

        <div className="post-meta">
          <span>{post.platform}</span>
          <span>{post.time}</span>
          <span>{post.language}</span>
          <span className={`severity ${post.severity}`}>
            {post.severity.toUpperCase()}
          </span>
        </div>
      </div> */}

      {/* 🔥 2. TEXT WITH XAI HIGHLIGHT */}
      {/* <div className="card">
      <h3 className="section-title">
        {post.content_type}
      </h3>
        <div className="highlight-text">
          <ShapHighlights text={post.text} />
        </div>
        {post.verdict === "BULLYING" && (
          <p className="xai-hint">
            Highlighted words indicate AI attention
          </p>
        )}
        {post.verdict !== "BULLYING" && (
          <p className="safe-text">
            No harmful patterns detected
          </p>
        )}
      </div> */} 

      {/* 🔥 1. HEADER (COMPACT DECISION BAR) */}
      <div className="card">
        <div className="decision-header">
          <div className={`verdict-box ${post.verdict === "BULLYING" ? "bad" : "good"}`}>
            <div className="verdict-text">{post.verdict}</div>
          </div>

          {/* ⚡ NEW: The Performance Flex Badge */}
          <div style={{ fontSize: "12px", color: "#6b7280", fontWeight: "600", background: "#e5e7eb", padding: "6px 12px", borderRadius: "999px", display: "flex", gap: "8px" }}>
            <span>⚡ Total: {post.latency.total_ms}ms</span>
            <span style={{opacity: 0.5}}>|</span>
            <span>SHAP: {post.latency.shap_ms}ms</span>
          </div>
        </div>

        <div className="post-meta">
          <span>{post.platform}</span>
          <span>{post.time}</span>
          <span>{post.language}</span>
          <span className={`severity ${post.severity}`}>
            {post.severity.toUpperCase()}
          </span>
        </div>
      </div>

      {/* 🔥 2. TEXT WITH XAI HIGHLIGHT */}
      <div className="card">
        <h3 className="section-title">{post.content_type}</h3>
        <div className="highlight-text">
          {/* 🔥 FIX: We must pass triggerWords down here! */}
          <ShapHighlights text={post.text} triggerWords={post.trigger_words} />
        </div>
        
        {post.verdict === "BULLYING" && (
          <p className="xai-hint">Highlighted words indicate AI attention</p>
        )}
        {post.verdict !== "BULLYING" && (
          <p className="safe-text">No harmful patterns detected</p>
        )}
      </div>

      {/* 🔥 3. AI SIGNALS (CLEAN VIEW) */}
      <div className="card">
        <h3 className="section-title">AI Signals</h3>

        <PredictionCard data={post} />

          <div className="inner-divider"></div>
            <ConfidenceBreakdown data={post} />
          </div>


        {/* 🔥 4. EXPLAINABILITY (ONLY FOR BULLYING) */}
        {post.verdict === "BULLYING" && (
          <div className="card">
            <h3 className="section-title">Why flagged</h3>

            <div className="xai-section">
              <FeatureList triggerWords={post.trigger_words} />
              <ContextSummary summary={post.summary} />
            </div>
          </div>
        )}

      {/* 🔥 5. ACTIONS */}
      <div className="card">
        <ActionButtons onAction={handleAction} />
      </div>

    </div>
  );
}