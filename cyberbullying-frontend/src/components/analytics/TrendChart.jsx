import React, { useEffect, useRef } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
  LineController
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  LineController,
  Title,
  Tooltip,
  Legend,
  Filler
);

export default function TrendChart({ trendData }) {
  const canvasRef = useRef(null);
  const chartRef = useRef(null);

  useEffect(() => {
    // Safety checks
    if (!canvasRef.current || !trendData || trendData.length === 0) return;

    if (chartRef.current) {
      chartRef.current.destroy();
    }

    const ctx = canvasRef.current.getContext('2d');
    
    chartRef.current = new ChartJS(ctx, {
      type: 'line',
      data: {
        labels: trendData.map(item => item.date || item._id),
        // Inside TrendChart.jsx -> datasets array

      datasets: [
        {
          label: 'Total Analyzed',
          data: trendData.map(item => item.total), // 👈 Maps to the backend 'total'
          fill: true,
          borderColor: '#3b82f6',
          backgroundColor: 'rgba(59, 130, 246, 0.05)',
          tension: 0.4,
          borderWidth: 3,
          pointRadius: 3, 
        },
        {
          label: 'Cyberbullying Detected',
          data: trendData.map(item => item.flagged), // 👈 Maps to the backend 'flagged'
          fill: false,
          borderColor: '#ef4444',
          borderDash: [4, 4],
          tension: 0.4,
          borderWidth: 1.5,
          pointRadius: 2,
        }
      ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { 
          // 👈 Turn this ON now that we have two datasets
          legend: { 
            display: true, 
            position: 'top',
            align: 'end',
            labels: { boxWidth: 10, font: { size: 12, weight: 'bold' } }
          },
          datalabels: { display: false } //  Suggest turning off for line charts to avoid clutter
        },
        scales: {
          x: { display: true, grid: { display: false } },
          y: { 
            display: true, 
            beginAtZero: true, 
            suggestedMax: 100, // Using suggestedMax so it can grow if data exceeds 150
            ticks: { stepSize: 25 } 
          }
        }
      }
    });

    return () => {
      if (chartRef.current) chartRef.current.destroy();
    };
  }, [trendData]);


  // 🛡️ THE EMPTY STATE FALLBACK
  if (!trendData || trendData.length === 0) {
    return (
      <div className="analysis-card col-12">
        <div className="trend-header" style={{ display: 'flex', justifyContent: 'space-between' }}>
          <p className="card-title" style={{ margin: 0 }}>Activity Trend</p>
          <span className="stat-sub">Last 15 Days</span>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '250px', color: '#6b7280' }}>
          <i className="fas fa-chart-line" style={{ fontSize: '2.5rem', marginBottom: '12px', color: '#d1d5db' }}></i>
          <p style={{ margin: 0, fontWeight: 'bold', fontSize: '1.1rem' }}>Collecting Trend Data...</p>
          <span style={{ fontSize: '0.875rem', marginTop: '4px' }}>Awaiting active data stream for the last 15 days</span>
        </div>
      </div>
    );
  }



  return (
    <div className="analysis-card col-12">
      <div className="trend-header">
        <p className="card-title" style={{ margin: 0 }}>Activity Trend</p>
        <span className="stat-sub">Last 15 Days</span>
      </div>
      
      {/* ⚠️ Notice: No className on the canvas. Just the container! */}
      <div className="chart-container">
        <canvas ref={canvasRef}></canvas>
      </div>
    </div>
  );
}