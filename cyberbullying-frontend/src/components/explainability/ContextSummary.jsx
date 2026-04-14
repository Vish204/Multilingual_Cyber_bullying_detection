export default function ContextSummary({ summary }) {
  return (
    <div className="xai-summary">

      <p className="xai-subtitle">Reasoning</p>

      <p className="xai-text">
        { summary }
      </p>

    </div>
  );
}