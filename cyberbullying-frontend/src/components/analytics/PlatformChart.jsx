import React, { useEffect, useRef } from 'react';
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, Tooltip, BarController, Legend } from 'chart.js';
import ChartDataLabels from 'chartjs-plugin-datalabels'; 

// 👈 Don't forget to register Legend so the user knows what the colors mean!
ChartJS.register(CategoryScale, LinearScale, BarElement, Tooltip, BarController, Legend, ChartDataLabels); 

export default function PlatformChart({ platformData }) {
  const canvasRef = useRef(null);
  const chartRef = useRef(null);

  // 🛡️ THE EMPTY STATE FALLBACK
  if (!platformData || Object.keys(platformData).length === 0) {
    return (
      <div className="analysis-card col-6">
        <p className="card-title">Platform Safety Comparison</p>
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '280px', color: '#6b7280' }}>
          <i className="fas fa-network-wired" style={{ fontSize: '2.5rem', marginBottom: '12px', color: '#d1d5db' }}></i>
          <p style={{ margin: 0, fontWeight: 'bold', fontSize: '1.1rem' }}>No Platform Data...</p>
          <span style={{ fontSize: '0.875rem', marginTop: '4px' }}>Connect data scrapers to populate</span>
        </div>
      </div>
    );
  }

  useEffect(() => {
    if (!canvasRef.current || !platformData) return;
    if (chartRef.current) chartRef.current.destroy();

    // Sort and filter platforms
    const sortedPlatforms = Object.entries(platformData)
      .filter(([key]) => key.toLowerCase() !== 'manual')
      .sort(([a], [b]) => a.localeCompare(b));

    const labels = sortedPlatforms.map(([k]) => k.charAt(0).toUpperCase() + k.slice(1));
    
    // Map our two new data streams
    const totalValues = sortedPlatforms.map(([, v]) => v.total || 0);
    const flaggedValues = sortedPlatforms.map(([, v]) => v.flagged || 0);

    const ctx = canvasRef.current.getContext('2d');
    chartRef.current = new ChartJS(ctx, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [
          {
            label: 'Total Posts',
            data: totalValues,
            backgroundColor: '#e5e7eb', // Light Gray (Noise)
            borderRadius: 4
          },
          {
            label: 'Flagged Bullying',
            data: flaggedValues,
            backgroundColor: '#ef4444', // Red (Signal)
            borderRadius: 4
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        layout: {
          padding: { top: 30 }
        },
        plugins: { 
          // Turn legend ON so examiner knows what Gray vs Red means
          legend: { 
            display: true, 
            position: 'top', 
            labels: { boxWidth: 12, font: { size: 11 } } 
          },
          datalabels: {
            anchor: 'end',      
            align: 'top',       
            color: '#4b5563',   
            font: { weight: 'bold', size: 10 },
            offset: 2,
            formatter: (value) => value > 0 ? value : null  
          }
        },
        scales: {
          y: { 
            beginAtZero: true, 
            grid: { color: '#f3f4f6' },
            ticks: { precision: 0 } 
          },
          x: { grid: { display: false } }
        }
      }
    });

    return () => { if (chartRef.current) chartRef.current.destroy(); };
  }, [platformData]);

  return (
    <div className="analysis-card col-6">
      <p className="card-title">Platform Toxicity</p>
      <div className="chart-container" style={{ height: '260px' }}>
        <canvas ref={canvasRef}></canvas>
      </div>
    </div>
  );
}



//To show manual as well


// import React, { useEffect, useRef } from 'react';
// import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, Tooltip, BarController } from 'chart.js';

// ChartJS.register(CategoryScale, LinearScale, BarElement, Tooltip, BarController);

// export default function PlatformChart({ platformData }) {
//   const canvasRef = useRef(null);
//   const chartRef = useRef(null);

//   useEffect(() => {
//     if (!canvasRef.current || !platformData) return;
//     if (chartRef.current) chartRef.current.destroy();

//     const ctx = canvasRef.current.getContext('2d');
//     chartRef.current = new ChartJS(ctx, {
//       type: 'bar',
//       data: {
//         labels: Object.keys(platformData).map(k => k.charAt(0).toUpperCase() + k.slice(1)),
//         datasets: [{
//           data: Object.values(platformData),
//           backgroundColor: '#6366f1',
//           borderRadius: 4
//         }]
//       },
//      options: {
//         responsive: true,
//         maintainAspectRatio: false,
//         plugins: { 
//           legend: { display: false },
//           // 👈 3. Updated DataLabels Config
//           datalabels: {
//             color: '#ffffff',     // White text for visibility
//             anchor: 'center',    // Positioned in the middle of the bar
//             align: 'center',
//             font: { weight: 'bold', size: 12 },
//             formatter: (value) => value > 0 ? value : null // Hide '0' labels
//           }
//         },
//         scales: {
//           y: { 
//             beginAtZero: true, 
//             grid: { color: '#f3f4f6' },
//             ticks: { precision: 0 } 
//           },
//           x: { grid: { display: false } }
//         }
//       }
//     });
//     return () => { if (chartRef.current) chartRef.current.destroy(); };
//   }, [platformData]);

//   return (
//     <div className="analysis-card col-6">
//       <p className="card-title">Platform Distribution</p>
//       <div className="chart-container">
//         <canvas ref={canvasRef}></canvas>
//       </div>
//     </div>
//   );
// }