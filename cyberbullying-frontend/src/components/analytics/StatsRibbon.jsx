import React from "react";
import MiniStats from "./MiniStats";

export default function StatsRibbon({ data }) {
  // 1. Get counts from the severity object specifically
  const severityCounts = data.severity || {};

  // 2. The REAL total of processed records
  const totalProcessed = Object.values(severityCounts).reduce((a, b) => a + b, 0);

  // 3. The total flagged (Severe + Moderate + Mild)
  const totalFlagged = (severityCounts.severe || 0) + 
                       (severityCounts.moderate || 0) + 
                       (severityCounts.mild || 0);

  // 4. Calculate Flag Rate
  const flagRate = totalProcessed > 0 
    ? ((totalFlagged / totalProcessed) * 100).toFixed(1) 
    : 0;

  // 5. Calculate active platforms
  const platforms = Object.keys(data.platforms || {}).filter(p => p.toLowerCase() !== 'manual');
  const platformCount = platforms.length;

  // 6. Safely grab the live latency from the backend (default to 0 if loading)
  const latency = data.system_latency_ms || 0;

  return (
    <>
      <MiniStats 
        title="Total Analyzed" 
        value={data.total_analyzed_posts} 
        sub="Last 15 days" 
        colorClass="text-blue"
      />
      <MiniStats 
        title="Flag Rate" 
        value={`${flagRate}%`} 
        sub="Avg. across platforms" 
        colorClass="text-orange"
      />
      <MiniStats 
        title="Active Platforms" 
        value={platformCount} 
        sub="Live data scrapers" 
        colorClass="text-green"
      />
      <MiniStats 
        title="System Latency" 
        value={`${latency}ms`}   
        sub="Avg. last 50 inferences" 
        colorClass="text-purple"
      />
    </>
  );
}


// import React from "react";
// import MiniStats from "./MiniStats";

// export default function StatsRibbon({ data }) {
//   const platformCount = data.platforms ? Object.keys(data.platforms).length : 0;
//   // const severityCounts = data.severity || {};

  
//   // const totalFlagged = (severityCounts.severe || 0) + 
//   //                      (severityCounts.moderate || 0) + 
//   //                      (severityCounts.mild || 0);

//   // const flagRate = data.total_analyzed_posts > 0 
//   //   ? ((totalFlagged / data.total_analyzed_posts) * 100).toFixed(1) 
//   //   : 0;
//   // 1. Get counts from the severity object specifically
// const severityCounts = data.severity || {};

// // 2. The REAL total of processed records
// const totalProcessed = Object.values(severityCounts).reduce((a, b) => a + b, 0);

// // 3. The total flagged (Severe + Moderate + Mild)
// const totalFlagged = (severityCounts.severe || 0) + 
//                      (severityCounts.moderate || 0) + 
//                      (severityCounts.mild || 0);

// // 4. Calculate Flag Rate using the processed total
// const flagRate = totalProcessed > 0 
//   ? ((totalFlagged / totalProcessed) * 100).toFixed(1) 
//   : 0;

//   return (
//     <>
//       <MiniStats 
//         title="Total Analyzed" 
//         value={data.total_analyzed_posts} 
//         sub="Processed UGC records" 
//         colorClass="text-blue"
//       />
//       <MiniStats 
//         title="Flag Rate" 
//         value={`${flagRate}%`} 
//         sub="Flagged as Cyberbullying" 
//         colorClass="text-orange"
//       />
//       <MiniStats 
//         title="Active Platforms" 
//         value={platformCount} 
//         sub="Monitored sources" 
//         colorClass="text-green"
//       />
//       <MiniStats 
//         title="Unique Languages" 
//         value="14" 
//         sub="Multilingual coverage" 
//         colorClass="text-purple"
//       />
//     </>
//   );
// }