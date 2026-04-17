import React, { useEffect, useRef } from 'react';
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, Tooltip } from 'chart.js';
import ChartDataLabels from 'chartjs-plugin-datalabels';

ChartJS.register(CategoryScale, LinearScale, BarElement, Tooltip, ChartDataLabels);

export default function ConfidenceChart({ alignmentData }) {
  const canvasRef = useRef(null);
  const chartRef = useRef(null);

  useEffect(() => {
    // If no data yet (threshold not met), don't attempt to draw the chart
    if (!canvasRef.current || !alignmentData) return;
    if (chartRef.current) chartRef.current.destroy();

    const mapping = [
      { label: 'AI & Human Agreed', value: alignmentData.agreed || 0, color: '#10b981' },
      { label: 'Moderator Re-evaluated', value: alignmentData.reevaluated || 0, color: '#f59e0b' }
    ];

    // Add 20% headroom so the longest bar doesn't touch the right edge
    const maxVal = Math.max(alignmentData.agreed, alignmentData.reevaluated);
    const suggestedMax = maxVal + (maxVal * 0.2);

    const ctx = canvasRef.current.getContext('2d');
    chartRef.current = new ChartJS(ctx, {
      type: 'bar',
      data: {
        labels: mapping.map(m => m.label),
        datasets: [{
          data: mapping.map(m => m.value),
          backgroundColor: mapping.map(m => m.color),
          borderRadius: 4,
          barThickness: 30
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          datalabels: {
            color: '#4b5563',
            anchor: 'end',
            align: 'right',
            offset: 8,
            font: { weight: 'bold', size: 12 },
            formatter: (v) => v
          }
        },
        scales: {
          x: { display: false, suggestedMax: suggestedMax },
          y: { grid: { display: false }, ticks: { font: { weight: 'bold' } } }
        },
        layout: { padding: { right: 50 } }
      }
    });

    return () => { if (chartRef.current) chartRef.current.destroy(); };
  }, [alignmentData]);

  // 🛡️ THE EMPTY STATE FALLBACK
  if (!alignmentData) {
    return (
      <div className="analysis-card col-6">
        <p className="card-title">Decision Alignment</p>
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '280px', color: '#6b7280' }}>
          <i className="fas fa-balance-scale" style={{ fontSize: '2.5rem', marginBottom: '12px', color: '#d1d5db' }}></i>
          <p style={{ margin: 0, fontWeight: 'bold', fontSize: '1.1rem' }}>Calibrating Alignment...</p>
          <span style={{ fontSize: '0.875rem', marginTop: '4px' }}>Waiting for moderator actions (20 minimum)</span>
        </div>
      </div>
    );
  }

  // 📊 THE RENDERED CHART
  return (
    <div className="analysis-card col-6">
      <div className="trend-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '15px' }}>
        <p className="card-title" style={{ margin: 0 }}>Decision Alignment</p>
        {/* Shows the Accuracy % calculated from your backend */}
        <span className="stat-sub" style={{ color: '#10b981', fontWeight: 'bold' }}>
          {alignmentData.accuracy_rate}% Accuracy
        </span>
      </div>
      <div className="chart-container" style={{ height: '260px' }}>
        <canvas ref={canvasRef}></canvas>
      </div>
    </div>
  );
}