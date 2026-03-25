export default function RightPanel() {
  return (
    <div style={{ padding: "10px" }}>
      <h3>Context Panel</h3>

      {/* Alert Section */}
      <div style={{ marginBottom: "20px", color: "red" }}>
        🚨 High severity alert detected
      </div>

      {/* Mini Stats */}
      <div>
        <h4>Quick Stats</h4>
        <p>High: 2</p>
        <p>Medium: 1</p>
        <p>Low: 1</p>
      </div>
    </div>
  );
}