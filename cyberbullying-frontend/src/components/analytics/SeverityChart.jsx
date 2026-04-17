// src/components/analytics/SeverityChart.jsx
import React, { useEffect, useRef } from 'react';
import { Chart as ChartJS, ArcElement, Tooltip, Legend, DoughnutController } from 'chart.js';
import ChartDataLabels from 'chartjs-plugin-datalabels'; // 👈 1. Import Plugin

ChartJS.register(ArcElement, Tooltip, Legend, DoughnutController, ChartDataLabels); // 👈 2. Register it

export default function SeverityChart({ severityData }) {
  const canvasRef = useRef(null);
  const chartRef = useRef(null);

  useEffect(() => {
    if (!canvasRef.current || !severityData) return;
    if (chartRef.current) chartRef.current.destroy();

    const order = ['none', 'mild', 'moderate', 'severe']; // Defined logical order

    // Create a sorted array of keys based on our preferred order
    const sortedKeys = Object.keys(severityData).sort((a, b) => {
    return order.indexOf(a) - order.indexOf(b);
    });

    const labels = sortedKeys.map(k => k.charAt(0).toUpperCase() + k.slice(1));
    const dataValues = sortedKeys.map(k => severityData[k]);
    const total = dataValues.reduce((a, b) => a + b, 0); // Need total for % math

    const colors = sortedKeys.map(key => {
      if (key === 'mild') return '#eab308';
      if (key === 'moderate') return '#f97316';
      if (key === 'severe') return '#ef4444';

      return '#10b981';
    });

    const ctx = canvasRef.current.getContext('2d');
    chartRef.current = new ChartJS(ctx, {
      type: 'doughnut',
      data: {
        labels: labels,
        datasets: [{
          data: dataValues,
          backgroundColor: colors,
          borderWidth: 2,
          borderColor: '#ffffff',
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        cutout: '65%',
        plugins: {
          legend: { 
            position: 'right', 
            labels: { usePointStyle: true, padding: 20, font: { size: 12 } } 
          },
          // 👈 3. DATALABELS CONFIGURATION
          datalabels: {
            color: '#fff',
            font: { weight: 'bold', size: 11 },
            // formatter: (value) => {
            //   const percentage = ((value / total) * 100).toFixed(1);
            //   return percentage > 5 ? `${percentage}%` : null; // Only show if > 5% to avoid crowding
            // },
            // inside SeverityChart.jsx options -> plugins -> datalabels
// Inside SeverityChart.jsx options -> plugins -> datalabels
          formatter: (value, ctx) => {
  const data = ctx.chart.data.datasets[0].data;
  const total = data.reduce((a, b) => a + b, 0);
  
  // 1. Calculate floor percentages for all
  const percentages = data.map(v => Math.floor((v / total) * 100));
  const sum = percentages.reduce((a, b) => a + b, 0);
  const diff = 100 - sum;

  // 2. Find the index of the largest value (usually 'None')
  const maxIndex = data.indexOf(Math.max(...data));

  // 3. Current value's index
  const currentIndex = ctx.dataIndex;

  // 4. If this is the largest slice, add the difference to it
  let finalPercent = percentages[currentIndex];
  if (currentIndex === maxIndex) {
    finalPercent += diff;
  }

  return finalPercent > 4 ? `${finalPercent}%` : null;
},
            display: true,
            align: 'center',
          }
        }
      }
    });

    return () => { if (chartRef.current) chartRef.current.destroy(); };
  }, [severityData]);


  // 🛡️ THE EMPTY STATE FALLBACK
  if (!severityData || Object.keys(severityData).length === 0) {
    return (
      <div className="analysis-card col-6">
        <p className="card-title">Threat Severity</p>
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '280px', color: '#6b7280' }}>
          <i className="fas fa-chart-pie" style={{ fontSize: '2.5rem', marginBottom: '12px', color: '#d1d5db' }}></i>
          <p style={{ margin: 0, fontWeight: 'bold', fontSize: '1.1rem' }}>Analyzing Threats...</p>
          <span style={{ fontSize: '0.875rem', marginTop: '4px' }}>Awaiting severity classifications</span>
        </div>
      </div>
    );
  }

  return (
    <div className="analysis-card col-6">
      <p className="card-title">Threat Severity</p>
      <div className="chart-container">
        <canvas ref={canvasRef}></canvas>
      </div>
    </div>
  );
}