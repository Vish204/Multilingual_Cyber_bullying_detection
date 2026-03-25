export default function ActionButtons({ onAction }) {
  return (
    <div className="actions-container">
      <h3 className="section-title">Moderation Actions</h3>

      <div className="action-buttons">
        <button className="btn ignore" onClick={() => onAction("ignore")}>
          Ignore
        </button>
        <button className="btn delete" onClick={() => onAction("delete")}>
          Delete
        </button>
        <button className="btn report" onClick={() => onAction("report")}>
          Report
        </button>
      </div>
    </div>
  );
}