import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { fetchDashboardAlerts } from "../../services/api";
import { transformPost } from "../../services/transform";
import "./dashboard.css";

export default function RecentAlerts() {
  const navigate = useNavigate();
  const [alerts, setAlerts] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadAlerts() {
      try {
        setLoading(true);
        const data = await fetchDashboardAlerts();
        
        const rawPosts = data.data || []; 
        const transformed = rawPosts.map(transformPost);
        
        setAlerts(transformed);
      } catch (err) {
        console.error("Error fetching Dashboard alerts:", err);
      } finally {
        setLoading(false);
      }
    }
    loadAlerts();
  }, []);

  if (loading) return <div className="alerts-section">Gathering priority alerts...</div>;

  return (
    <div className="alerts-section">
      <div className="section-header-flex">
        <h2 className="section-title">Critical Alerts, Not Reviewed Yet</h2>
        {/* <span className="view-all-link" onClick={() => { console.log("🚨 DASHBOARD CLICK FIRED! Alert ID:", alert.id); 
          navigate(`/moderation?alertId=${alert.id}`)}}>
           Showing latest 5
        </span> */}
      </div>

      <div className="alerts-list">
        {alerts.length === 0 ? (
          <div className="no-alerts-card">
            <p className="no-alerts-text">✅ System clear. No pending alerts.</p>
          </div>
        ) : (
          alerts.map((alert) => (
            // <div
            //   key={alert.id}
            //   className={`alert-card ${alert.severity}`}
            //   onClick={() => navigate("/moderation")}
            // >
            <div
              key={alert.id}
              className={`alert-card ${alert.severity}`}
              // 👇 1. Force it to the front, and give it a visible magenta border
              // style={{ position: "relative", zIndex: 9999, border: "2px solid magenta", cursor: "pointer" }}
              
              onClick={(e) => {
                // 👇 2. Stop the click from bubbling up to any invisible parents
                e.preventDefault();
                e.stopPropagation();
                
                //console.log("🚨 NUKED CLICK FIRED! Alert ID:", alert.id);
                navigate('/moderation', { state: { alertPost: alert } });
              }}
            >
              <p className="alert-text">
                {alert.text ? alert.text : "Content missing from record"}
              </p>

              <div className="alert-meta">
                <span>{alert.platform || "Direct API"}</span>
                <span className="severity-badge">{alert.severity}</span>
                <span>{alert.time}</span>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}






// import { useEffect, useState } from "react";
// import { useNavigate } from "react-router-dom";
// import { fetchAlerts } from "../../services/api";
// import { transformPost } from "../../services/transform";
// import "./dashboard.css";

// export default function RecentAlerts() {
//   const navigate = useNavigate();
//   const [alerts, setAlerts] = useState([]);

//   useEffect(() => {
//     async function loadAlerts() {
//       try {
//         const data = await fetchAlerts();
//         // console.log("Alerts API FULL:", JSON.stringify(data, null, 2)); // 🔍 debug

//         const transformed = (data.data || []).map(transformPost);
//         setAlerts(transformed);
//       } catch (err) {
//         console.error("Error fetching alerts:", err);
//       }
//     }

//     loadAlerts();
//   }, []);

//   return (
//     <div className="alerts-section">
//       <h2 className="section-title">Recent Alerts</h2>

//       <div className="alerts-list">
//         {alerts.length === 0 ? (
//           <p>No alerts found</p>
//         ) : (
//           alerts.map((alert) => (
//             <div
//               key={alert.id}
//               className={`alert-card ${alert.severity}`}
//               onClick={() => navigate("/moderation")}
//             >
//               <p className="alert-text">{alert.text}</p>

//               <div className="alert-meta">
//                 <span>{alert.platform}</span>
//                 <span className="severity">{alert.severity}</span>
//               </div>
//             </div>
//           ))
//         )}
//       </div>
//     </div>
//   );
// }