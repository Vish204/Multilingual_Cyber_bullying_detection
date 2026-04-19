export default function ConfidenceBreakdown({ data }) {
  // console.log("🚨 RAW DATA IN BREAKDOWN:", data);
  
  //const base = ((data.signals?.components?.base_cyberbullying || 0) * 100).toFixed(0); // TEMP (replace later with p_cb)
  const base = ((data?.base_score || 0) * 100).toFixed(0);
  const emotion = (data.emotion_score * 100).toFixed(0);
  const sarcasm = (data.sarcasm * 100).toFixed(0);
  const final = (data.confidence * 100).toFixed(0);

  return (
    <div className="breakdown-card">

      <h3 className="breakdown-title">Model Contributions</h3>

      <div className="breakdown-row">
        <span>Base Model</span>
        <span>{base}%</span>
      </div>

      <div className="breakdown-row">
        <span>Emotion Signal</span>
        <span>{emotion}%</span>
      </div>

      {/* Show sarcasm */}
        <div className="breakdown-row">
          <span>Sarcasm Signal</span>
          <span>{sarcasm}%</span>
        </div>

      <div className="divider"></div>

      <div className="breakdown-row highlight">
        <span>Final Confidence</span>
        <span>{final}%</span>
      </div>

    </div>
  );
}