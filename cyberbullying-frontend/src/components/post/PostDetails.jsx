import PredictionCard from "./PredictionCard";
import ShapHighlights from "../explainability/ShapHighlights";
import FeatureList from "../explainability/FeatureList";
import ContextSummary from "../explainability/ContextSummary";
import ActionButtons from "./ActionButtons";

export default function PostDetails({ post, onAction }) {
  if (!post) {
    return <p>Select a post to view details</p>;
  }

  // 🔥 Dummy AI data (temporary)
  const aiData = {
    label: "bullying",
    confidence: 0.87,
    severity: "high",
    language: "English",
    emotion: "aggression",
    sarcasm: 0.2
  };

  // Handle moderation actions
  const handleAction = (action) => {
    onAction(action, post.id);
  };

  return (
    <div>
      {/* Text */}
      <h3>Post</h3>
      <p>{post.text}</p>

      {/* Meta */}
      <div style={{ marginTop: "10px", color: "gray" }}>
        <small>Platform: {post.platform}</small><br />
        <small>Time: {post.time}</small>
      </div>

      {/* AI Output */}
      <PredictionCard data={aiData} />
    

      {/* Explainability */}
      <div style={{ marginTop: "20px" }}>
        <h3>Explainability</h3>

        <ShapHighlights text={post.text} />
        <FeatureList />
        <ContextSummary />
      </div>

      <ActionButtons onAction={handleAction} />
    </div>
  );
}