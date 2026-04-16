import { useEffect, useState } from "react";
import { fetchDashboardSummary } from "../../services/api";
import "./dashboard.css";

export default function StatsOverview() {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadStats() {
      try {
        const data = await fetchDashboardSummary();
        setStats(data);
      } catch (err) {
        console.error("Error loading dashboard summary:", err);
      } finally {
        setLoading(false);
      }
    }
    loadStats();
  }, []);

  if (loading) return <div className="stats-container">Loading metrics...</div>;

  const statCards = [
    { label: "Total (15d)", value: stats.total_posts, color: "blue" },
    { label: "Bullying %", value: `${stats.bullying_percentage}%`, color: "red" },
    { label: "Pending Priority", value: stats.pending_priority, color: "orange" },
    { label: "System Speed", value: `⚡ ${stats.avg_latency_ms}ms`, color: "green", sub: "Avg last 50 UGC" },
  ];

  return (
    <div className="stats-container">
      {statCards.map((item, index) => (
        <div key={index} className={`stat-card stat-${item.color}`}>
          <h2>{item.value}</h2>
          <p>{item.label}</p>
          {item.sub && (
            <span className="stat-sub-label">
              {item.sub}
            </span>
          )}
        </div>
      ))}
    </div>
  );
}





  // import { useEffect, useState } from "react";
  // import { fetchSeverityStats } from "../../services/api";
  // import { fetchPosts } from "../../services/api";
  // import "./dashboard.css";

  // export default function StatsOverview() {
  //   const [statsData, setStatsData] = useState(null);
  //   const [reviewedCount, setReviewedCount] = useState(0);


  //   async function loadReviewed() {
  //     try {
  //       const data = await fetchPosts();

  //       // handle both formats
  //       const posts = data.data || data;

  //       const count = posts.filter(p => p.reviewed === true).length;

  //       setReviewedCount(count);
  //     } catch (err) {
  //       console.error("Error fetching reviewed:", err);
  //     }
  //   }

  //   useEffect(() => {
  //     async function loadStats() {
  //       try {
  //         const data = await fetchSeverityStats();
  //         console.log("Stats API:", data);
  //         setStatsData(data);
  //       } catch (err) {
  //         console.error("Error fetching stats:", err);
  //       }
  //     }

  //     loadStats();
  //     loadReviewed();
  //   }, []);

  //   if (!statsData) {
  //     return <div className="stats-container">Loading stats...</div>;
  //   }

  //   // ✅ TOTAL POSTS
  //   const total =
  //     (statsData.none || 0) +
  //     (statsData.mild || 0) +
  //     (statsData.moderate || 0) +
  //     (statsData.severe || 0);

  //   // ✅ BULLYING = moderate + severe
  //   const bullyingCount =
  //     (statsData.moderate || 0) + (statsData.severe || 0);

  //   const bullyingPercent =
  //     total > 0 ? Math.round((bullyingCount / total) * 100) : 0;

  //   // ✅ ALERTS = severe
  //   const alerts = statsData.severe || 0;

  //   // ⚠️ reviewed check only 50 data since we using /history. later we do separate api
  //   const reviewed = reviewedCount;

  //   const stats = [
  //     { label: "Total Posts", value: total, color: "blue" },
  //     { label: "Bullying Detected", value: `${bullyingPercent}%`, color: "red" },
  //     { label: "Active Alerts", value: alerts, color: "orange" },
  //     { label: "Reviewed", value: reviewed, color: "green" },
  //   ];

  //   return (
  //     <div className="stats-container">
  //       {stats.map((stat, index) => (
  //         <div key={index} className={`stat-card stat-${stat.color}`}>
  //           <h2>{stat.value}</h2>
  //           <p>{stat.label}</p>
  //         </div>
  //       ))}
  //     </div>
  //   );
  // }