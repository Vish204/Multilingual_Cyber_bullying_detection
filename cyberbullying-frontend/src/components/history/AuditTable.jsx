import React from 'react';

export default function AuditTable({ logs, loading }) {
  if (loading) return <div className="table-loading">Loading Audit Logs...</div>;
  if (logs.length === 0) return <div className="table-empty">No records match the current filter.</div>;

  // 👇 1. Add this tiny formatter
  const formatPlatform = (platform) => {
    const names = {
      youtube: "YouTube",
      reddit: "Reddit",
      twitter: "Twitter"
    };
    const key = platform.toLowerCase();
    return names[key] || platform.charAt(0).toUpperCase() + platform.slice(1);
  };

  const formatDate = (isoString) => {
    const date = new Date(isoString);
    return date.toLocaleDateString() + ' - ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  return (
    <table className="audit-table">
      <thead>
        <tr>
          <th>Timestamp</th>
          <th>Platform</th>
          <th>Content Snippet</th>
          <th>AI Prediction</th>
          <th>Moderator Action</th>
          <th>Alignment</th>
        </tr>
      </thead>
      <tbody>
        {logs.map(log => (
          <tr key={log.id}>
            <td className="timestamp-cell">{formatDate(log.timestamp)}</td>
            <td className="platform-cell">
              <span className={`plat-badge ${log.platform.toLowerCase()}`}>
                {formatPlatform(log.platform)}
              </span>
            </td>
            <td className="content-cell" title={log.text}>
              {log.text}
            </td>
            <td>
              <span className={`ai-badge ${log.ai_severity.toLowerCase()}`}>
                {log.ai_severity.toUpperCase()} ({Math.round(log.ai_confidence)}%)
              </span>
            </td>
            <td>
              <span className={`mod-badge ${log.moderator_action.toLowerCase()}`}>
                {log.moderator_action.toUpperCase()}
              </span>
            </td>
            <td className="alignment-cell">
              {log.alignment_status === 'Agreed' && (
                <span className="align-agreed">✅ Agreed</span>
              )}
              {log.alignment_status === 'Overruled' && (
                <span className="align-overruled">⚠️ Overruled</span>
              )}
              {log.alignment_status === 'Pending Review' && (
                <span className="align-pending" style={{ color: '#d97706', fontWeight: '600' }}>
                  ⏳ Pending
                </span>
              )}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}