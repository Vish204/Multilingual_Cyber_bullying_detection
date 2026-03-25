import SeverityBadge from "./SeverityBadge";
import EmotionTag from "./EmotionTag";
import SarcasmIndicator from "./SarcasmIndicator";

export default function PredictionCard({ data }) {
  return (
    <div style={{
      border: "1px solid #ccc",
      padding: "10px",
      marginTop: "15px",
      borderRadius: "8px"
    }}>
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