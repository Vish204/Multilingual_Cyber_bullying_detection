import React from 'react';

export default function HistoryHeader({ showDisagreements, setShowDisagreements, onExport, totalCount }) {
  return (
    <div className="history-header">
      <div>
        <h2 className="history-title">System Audit Log</h2>
        <span className="history-subtext">Showing {totalCount} reviewed records</span>
      </div>
      
      <div className="history-controls">
        <button 
          className={`toggle-btn ${showDisagreements ? 'active' : ''}`}
          onClick={() => setShowDisagreements(!showDisagreements)}
        >
          <i className="fas fa-filter"></i> 
          {showDisagreements ? "View All Records" : "Show AI Disagreements Only"}
        </button>
        
        <button className="export-btn" onClick={onExport}>
          <i className="fas fa-file-csv"></i> Export CSV
        </button>
      </div>
    </div>
  );
}