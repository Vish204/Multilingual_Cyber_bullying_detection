export default function FeatureList() {
  // 🔥 dummy for now (later from backend SHAP/attention)
  const features = [
    { word: "stupid", impact: 0.42 },
    { word: "hate", impact: 0.31 },
    { word: "loser", impact: 0.25 },
  ];

  return (
    <div className="xai-features">

      <p className="xai-subtitle">Top Influential Signals</p>

      <div className="xai-tags">
        {features.map((f, index) => (
          <span key={index} className="xai-tag">
            {f.word}
          </span>
        ))}
      </div>

    </div>
  );
}