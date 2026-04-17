// src/components/analytics/MiniStats.jsx
import React from "react";

export default function MiniStats({ title, value, sub, colorClass }) {
  return (
    <div className="analysis-card col-3">
      <p className="card-title">{title}</p>
      <div>
        <h2 className={`stat-value ${colorClass}`}>{value}</h2>
        <p className="stat-sub">{sub}</p>
      </div>
    </div>
  );
}