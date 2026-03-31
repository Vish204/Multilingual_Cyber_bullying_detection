import "./dashboard.css";

export default function StatsOverview() {
  // Dummy data for now (later from backend)
const stats = [
  { label: "Total Posts", value: 1280, color: "blue" },
  { label: "Bullying Detected", value: "32%", color: "red" },
  { label: "Active Alerts", value: 14, color: "orange" },
  { label: "Reviewed", value: 860, color: "green" },
];

  return (
    <div className="stats-container">
      {stats.map((stat, index) => (
        <div key={index} className={`stat-card stat-${stat.color}`}> 
          <h2>{stat.value}</h2>
          <p>{stat.label}</p>
        </div>
      ))}
    </div>
  );
}