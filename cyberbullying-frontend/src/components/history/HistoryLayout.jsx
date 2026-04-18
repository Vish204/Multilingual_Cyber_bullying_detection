import React, { useState, useEffect } from 'react';
import HistoryHeader from './HistoryHeader';
import AuditTable from './AuditTable';
import './history.css';

export default function HistoryLayout() {
  const [logs, setLogs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [showDisagreements, setShowDisagreements] = useState(false);

  useEffect(() => {
    async function fetchHistory() {
      try {
        // Adjust the URL to match your backend port/setup
        const response = await fetch('http://127.0.0.1:8000/history/reviewed');
        const result = await response.json();
        if (result.status === 'success') {
          setLogs(result.data);
        }
      } catch (error) {
        console.error("Failed to fetch history:", error);
      } finally {
        setLoading(false);
      }
    }
    fetchHistory();
  }, []);

  // ⚡ The "Smart Filter" - Instant frontend filtering
  const displayedLogs = showDisagreements 
    ? logs.filter(log => log.alignment_status === 'Overruled')
    : logs;

  // 📥 The Export Engine (Mic Drop Feature)
  const exportToCSV = () => {
    if (displayedLogs.length === 0) return alert("No data to export.");

    // Define columns
    const headers = ["Timestamp", "Platform", "Text", "AI_Severity", "AI_Confidence", "Moderator_Action", "Alignment"];
    
    // Map rows to CSV format
    const csvRows = displayedLogs.map(log => {
      // Escape quotes in text to prevent CSV breaking
      const safeText = `"${log.text.replace(/"/g, '""')}"`;
      return `${log.timestamp},${log.platform},${safeText},${log.ai_severity},${log.ai_confidence},${log.moderator_action},${log.alignment_status}`;
    });

    const csvContent = [headers.join(","), ...csvRows].join("\n");
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);

    const baseName = showDisagreements ? 'ML_Retraining_Dataset' : 'System_Audit_Log';
    const fileName = `${baseName}_${new Date().toISOString().split('T')[0]}.csv`;

    link.setAttribute('download', fileName);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="history-container">
      <HistoryHeader 
        showDisagreements={showDisagreements} 
        setShowDisagreements={setShowDisagreements} 
        onExport={exportToCSV}
        totalCount={displayedLogs.length}
      />
      <div className="table-wrapper">
        <AuditTable logs={displayedLogs} loading={loading} />
      </div>
    </div>
  );
}