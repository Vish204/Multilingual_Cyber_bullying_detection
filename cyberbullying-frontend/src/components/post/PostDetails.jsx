import PredictionCard from "./PredictionCard";
import ShapHighlights from "../explainability/ShapHighlights";
import FeatureList from "../explainability/FeatureList";
import ContextSummary from "../explainability/ContextSummary";
import ActionButtons from "./ActionButtons";

export default function PostDetails({ post, onAction }) {
  if (!post) {
    return <p className="empty-state">Select a post to view details</p>;
  }

  const aiData = {
    label: "bullying",
    confidence: 0.87,
    severity: "high",
    language: "English",
    emotion: "aggression",
    sarcasm: 0.2
  };

  const handleAction = (action) => {
    onAction(action, post.id);
  };

  return (
    <div className="post-container">

      {/* 🔹 HEADER */}
      <div className="post-header">
        <h2 className="post-text">{post.text}</h2>

        <div className="post-meta">
          <span>{post.platform}</span>
          <span>{post.time}</span>
        </div>
      </div>

      {/* 🔹 AI CARD */}
      <div className="card">
        <PredictionCard data={aiData} />
      </div>

      {/* 🔹 EXPLAINABILITY */}
      <div className="card">
        <h3 className="section-title">Explainability</h3>

        <ShapHighlights text={post.text} />
        <FeatureList />
        <ContextSummary />
      </div>

      {/* 🔹 ACTIONS */}
      <div className="card">
        <h3 className="section-title">Moderation Actions</h3>
        <ActionButtons onAction={handleAction} />
      </div>

    </div>
  );
}