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

  const handleAction = (action) => {
    onAction(action, post.id);
  };

  return (
    <div className="post-container">

      {/* 🔥 1. HEADER (COMPACT DECISION BAR) */}
      <div className="card">
        <div className="decision-header">

          <div className={`verdict-box ${post.verdict === "BULLYING" ? "bad" : "good"}`}>
            <div className="verdict-text">{post.verdict}</div>
            {/* <div className="verdict-confidence">
              {(post.confidence * 100).toFixed(0)}%
            </div> */}
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
              <FeatureList />
              <ContextSummary />
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