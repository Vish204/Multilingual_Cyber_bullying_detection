export default function ContextSummary() {
  return (
    <div style={{ marginTop: "10px" }}>
      <h4>Why flagged:</h4>
      <ul>
        <li>Aggressive tone detected</li>
        <li>Direct insult keywords present</li>
        <li>Low sarcasm → likely genuine</li>
      </ul>
    </div>
  );
}