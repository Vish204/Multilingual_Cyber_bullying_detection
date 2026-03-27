import { useState } from "react";


export default function ActionButtons({ onAction }) {

  const [showReportInput, setShowReportInput] = useState(false);
  const [reportReason, setReportReason] = useState("");

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

        {/* <button className="btn report" onClick={() => onAction("report")}>
          Report
        </button> */}
        {/* 🔴 REPORT */}

        {showReportInput ? (
          <div className="report-box">
            <input
              type="text"
              placeholder="Reason (optional)"
              value={reportReason}
              onChange={(e) => setReportReason(e.target.value)}
            />

            <button
              onClick={() => {
                onAction("report", reportReason);
                setReportReason("");
                setShowReportInput(false);
              }}
            >
              Confirm
            </button>
          </div>
        ) : (
          <button className="btn report" onClick={() => setShowReportInput(true)}>
             Report
          </button>
        )}
        <button className="btn save" onClick={() => onAction("save")}>
          Save
        </button>
      </div>
    </div>
  );
}