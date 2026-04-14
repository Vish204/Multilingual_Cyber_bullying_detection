export default function FeatureList({ triggerWords = [] }) {
  if (triggerWords.length === 0) return null;

  return (
    <div className="xai-features">
      <p className="xai-subtitle">Top Influential Signals</p>
      <div className="xai-tags">
        {triggerWords.map((f, index) => (
          <span key={index} className="xai-tag">
            {f.word}
          </span>
        ))}
      </div>
    </div>
  );
}