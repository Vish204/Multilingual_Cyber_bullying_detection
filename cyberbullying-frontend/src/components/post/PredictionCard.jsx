import EmotionTag from "./EmotionTag";
import SarcasmIndicator from "./SarcasmIndicator";

export default function PredictionCard({ data }) {
  const confidencePercent = (data.confidence * 100).toFixed(0);

  return (
    <div>

      {/* ✅ CONFIDENCE LABEL */}
      <div className="confidence-header">
        <span className="confidence-title confidence-theme">Model Confidence</span>
        <span className="confidence-value confidence-theme">{confidencePercent}%</span>
      </div>

      {/* ✅ CONFIDENCE BAR */}
      <div className="confidence-bar">
        <div
          className="confidence-fill"
          style={{ width: `${confidencePercent}%` }}
        ></div>
      </div>

      {/* ✅ AI SIGNALS (SEPARATE MEANING) */}
      <div className="ai-tags">

        <EmotionTag emotion={data.emotion} />

        {/* Show sarcasm ONLY if >= 50% */}
        {data.sarcasm >= 0.5 && (
          <SarcasmIndicator value={data.sarcasm} />
        )}

      </div>

    </div>
  );
}