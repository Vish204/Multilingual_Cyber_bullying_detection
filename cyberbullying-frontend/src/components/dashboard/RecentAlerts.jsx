import { useNavigate } from "react-router-dom";
import "./dashboard.css";

export default function RecentAlerts() {
  const navigate = useNavigate();

  // Dummy alerts (later from backend)
  const alerts = [
    {
      id: 1,
      text: "You are so useless, nobody wants you here.",
      severity: "high",
      platform: "Twitter",
    },
    {
      id: 2,
      text: "This is why people like you should stay quiet.",
      severity: "high",
      platform: "Reddit",
    },
    {
      id: 3,
      text: "Such a dumb comment, can't believe this.",
      severity: "medium",
      platform: "YouTube",
    },
  ];

  return (
    <div className="alerts-section">
      <h2 className="section-title">Recent Alerts</h2>

      <div className="alerts-list">
        {alerts.map((alert) => (
          <div
            key={alert.id}
            className={`alert-card ${alert.severity}`}
            onClick={() => navigate("/moderation")}
          >
            <p className="alert-text">{alert.text}</p>

            <div className="alert-meta">
              <span>{alert.platform}</span>
              <span className="severity">{alert.severity}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}