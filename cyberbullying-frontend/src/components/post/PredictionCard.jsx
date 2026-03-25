import SeverityBadge from "./SeverityBadge";
import EmotionTag from "./EmotionTag";
import SarcasmIndicator from "./SarcasmIndicator";

export default function PredictionCard({ data }) {
  return (
    <div className="card">
      <h3>AI Prediction</h3>

      <p><strong>Label:</strong> {data.label}</p>
      <p><strong>Confidence:</strong> {(data.confidence * 100).toFixed(2)}%</p>

      <p>
        <strong>Severity:</strong>{" "}
        <SeverityBadge severity={data.severity} />
      </p>

      <p><strong>Language:</strong> {data.language}</p>

      <EmotionTag emotion={data.emotion} />
      <SarcasmIndicator value={data.sarcasm} />
    </div>
  );
}