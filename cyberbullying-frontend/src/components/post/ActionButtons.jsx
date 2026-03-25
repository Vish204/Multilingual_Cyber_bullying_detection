export default function ActionButtons({ onAction }) {
  return (
    <div style={{ marginTop: "20px" }}>
      <h3>Moderation Actions</h3>

      <button onClick={() => onAction("ignore")}>Ignore</button>
      <button onClick={() => onAction("delete")}>Delete</button>
      <button onClick={() => onAction("report")}>Report</button>
    </div>
  );
}