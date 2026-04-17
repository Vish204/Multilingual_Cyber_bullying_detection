import React, { useEffect, useState } from "react";
import { fetchAnalyticsOverview } from "../services/api";

import AnalyticsHeader from "../components/analytics/AnalyticsHeader";
import StatsRibbon from "../components/analytics/StatsRibbon";
import TrendChart from "../components/analytics/TrendChart";
import DistributionRow from "../components/analytics/DistributionRow";
import LanguageChart from "../components/analytics/LanguageChart";
import ConfidenceChart from "../components/analytics/ConfidenceChart";

import "../components/analytics/analysis.css";

export default function Analysis() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  // useEffect(() => {
  //   fetchAnalyticsOverview().then(res => {
  //     setData(res);
  //     setLoading(false);
  //   }).catch(() => setLoading(false));
  // }, []);


  

  useEffect(() => {
    // 1. Define the fetching function
    const loadData = async () => {
      try {
        const res = await fetchAnalyticsOverview();
        setData(res);
        setLoading(false);
      } catch (err) {
        console.error("Fetch error:", err);
        setLoading(false);
      }
    };

    // 2. Run it immediately on load
    loadData();

    // 3. Set a timer to run it every 30 seconds
    const interval = setInterval(loadData, 30000); 

    // 4. CLEANUP: If the user leaves this page, stop the timer
    // This is crucial to prevent memory leaks and background crashes!
    return () => clearInterval(interval);
  }, []); // Still empty brackets because we only set up the timer ONCE


  if (loading) return <div className="analysis-container">Loading...</div>;

  return (
    <div className="analysis-container">
      <AnalyticsHeader />
      <div className="analysis-grid">
        <StatsRibbon data={data} />
        <TrendChart trendData={data.trends} />
        <DistributionRow 
          severityData={data.severity} 
          platformData={data.platforms} 
        />
      </div>

      <div className="analysis-grid" style={{ marginTop: '24px' }}>
        {data.languages && <LanguageChart langData={data.languages} />}
        {data.trust_levels && <ConfidenceChart alignmentData={data.alignment} />}
      </div>
    </div>
  );
}