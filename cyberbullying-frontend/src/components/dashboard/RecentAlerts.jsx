import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { fetchAlerts } from "../../services/api";
import { transformPost } from "../../services/transform";
import "./dashboard.css";

export default function RecentAlerts() {
  const navigate = useNavigate();
  const [alerts, setAlerts] = useState([]);

  useEffect(() => {
    async function loadAlerts() {
      try {
        const data = await fetchAlerts();
        // console.log("Alerts API FULL:", JSON.stringify(data, null, 2)); // 🔍 debug

        const transformed = (data.data || []).map(transformPost);
        setAlerts(transformed);
      } catch (err) {
        console.error("Error fetching alerts:", err);
      }
    }

    loadAlerts();
  }, []);

  return (
    <div className="alerts-section">
      <h2 className="section-title">Recent Alerts</h2>

      <div className="alerts-list">
        {alerts.length === 0 ? (
          <p>No alerts found</p>
        ) : (
          alerts.map((alert) => (
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
          ))
        )}
      </div>
    </div>
  );
}