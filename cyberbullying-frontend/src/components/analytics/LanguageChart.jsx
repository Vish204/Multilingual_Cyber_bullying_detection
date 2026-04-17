import React, { useState, useEffect, useRef } from 'react';
import { 
  Chart as ChartJS, 
  CategoryScale, 
  LinearScale, 
  BarElement, 
  Tooltip 
} from 'chart.js';
import ChartDataLabels from 'chartjs-plugin-datalabels';

ChartJS.register(CategoryScale, LinearScale, BarElement, Tooltip, ChartDataLabels);

export default function LanguageChart({ langData }) {
  const [showAll, setShowAll] = useState(false);
  const canvasRef = useRef(null);
  const chartRef = useRef(null);

  useEffect(() => {
    // 🛡️ Safety check (will trigger empty state below if data is missing)
    if (!canvasRef.current || !langData || langData.length === 0) return;
    if (chartRef.current) chartRef.current.destroy();

    // ✂️ The Toggle Logic: Slice the first 5, or use the whole array
    const displayData = showAll ? langData : langData.slice(0, 5);

    // Clean up long technical names for the UI
    const labels = displayData.map(item => {
      let label = item.language;
      if (label === 'Keywords_hinglish_romanized') return 'HINGLISH';
      if (label === 'Hindi_or_marathi') return 'HINDI/MARATHI';
      return String(label).toUpperCase();
    });

    const counts = displayData.map(item => item.count);

    const ctx = canvasRef.current.getContext('2d');
    chartRef.current = new ChartJS(ctx, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [{
          data: counts,
          backgroundColor: '#8b5cf6', // Purple
          borderRadius: 4,
          // Dynamic thickness: thinner when showing all 14, thicker for top 5
          barThickness: showAll ? 12 : 20 
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          datalabels: {
            anchor: 'end',
            align: 'right',
            color: '#4b5563',
            offset: 5,
            font: { weight: 'bold', size: 11 }
          }
        },
        scales: {
          x: { 
            display: false,
            // Add padding so the longest number doesn't get cut off
            suggestedMax: Math.max(...counts) * 1.2 
          },
          y: { 
            grid: { display: false },
            ticks: { font: { weight: 'bold' } }
          }
        },
        layout: { padding: { right: 40 } }
      }
    });

    return () => { if (chartRef.current) chartRef.current.destroy(); };
  }, [langData, showAll]);

  // 🛡️ THE EMPTY STATE FALLBACK
  if (!langData || langData.length === 0) {
    return (
      <div className="analysis-card col-6">
        <p className="card-title">Language Distribution</p>
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '280px', color: '#6b7280' }}>
          <i className="fas fa-language" style={{ fontSize: '2.5rem', marginBottom: '12px', color: '#d1d5db' }}></i>
          <p style={{ margin: 0, fontWeight: 'bold', fontSize: '1.1rem' }}>Collecting Multilingual Data...</p>
          <span style={{ fontSize: '0.875rem', marginTop: '4px' }}>Awaiting valid language inputs</span>
        </div>
      </div>
    );
  }

  // 📊 THE RENDERED CHART
  return (
    <div className="analysis-card col-6">
      <div className="trend-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '15px' }}>
        <p className="card-title" style={{ margin: 0 }}>Language Distribution</p>
        {/* Only show the button if there are actually more than 5 languages */}
        {langData.length > 5 && (
          <button 
            onClick={() => setShowAll(!showAll)}
            style={{ padding: '4px 12px', fontSize: '0.8rem', borderRadius: '4px', border: '1px solid #e5e7eb', background: '#f9fafb', cursor: 'pointer', fontWeight: 'bold', color: '#4b5563' }}
          >
            {showAll ? "Show Top 5" : `See All (${langData.length})`}
          </button>
        )}
      </div>
      <div className="chart-container" style={{ height: '260px' }}>
        <canvas ref={canvasRef}></canvas>
      </div>
    </div>
  );
}





// import React, { useState, useEffect, useRef } from 'react';
// import { 
//   Chart as ChartJS, // 👈 This was missing or incorrectly named
//   CategoryScale, 
//   LinearScale, 
//   BarElement, 
//   Tooltip 
// } from 'chart.js';
// import ChartDataLabels from 'chartjs-plugin-datalabels';

// ChartJS.register(CategoryScale, LinearScale, BarElement, Tooltip, ChartDataLabels);

// export default function LanguageChart({ langData }) {
//   const [showAll, setShowAll] = useState(false);
//   const canvasRef = useRef(null);
//   const chartRef = useRef(null);

//   // inside LanguageChart.jsx -> useEffect
// useEffect(() => {
//     if (!canvasRef.current || !langData || !langData.top) return;
//     if (chartRef.current) chartRef.current.destroy();

//     // 1. Map the 'top' array using your exact keys: 'language' and 'count'
//     let entries = langData.top.map(item => {
//       let label = item.language || "Unknown";
      
//       // Clean up long technical names for the UI
//       if (label === 'Keywords_hinglish_romanized') label = 'Hinglish';
//       if (label === 'Hindi_or_marathi') label = 'Hindi/Marathi';
      
//       return [label, item.count || 0];
//     });

//     // 2. Add 'Others' if toggled
//     if (showAll && langData.others > 0) {
//       entries.push(['Others', langData.others]);
//     }

//     const ctx = canvasRef.current.getContext('2d');
//     chartRef.current = new ChartJS(ctx, {
//       type: 'bar',
//       data: {
//         labels: entries.map(item => String(item[0]).toUpperCase()),
//         datasets: [{
//           data: entries.map(item => item[1]),
//           backgroundColor: '#8b5cf6',
//           borderRadius: 4,
//           barThickness: 18
//         }]
//       },
//       options: {
//         indexAxis: 'y',
//         responsive: true,
//         maintainAspectRatio: false,
//         plugins: {
//           legend: { display: false },
//           datalabels: {
//             anchor: 'end',
//             align: 'right',
//             color: '#4b5563',
//             offset: 5,
//             font: { weight: 'bold', size: 11 }
//           }
//         },
//         scales: {
//           x: { display: false },
//           y: { 
//             grid: { display: false },
//             ticks: { font: { weight: 'bold' } }
//           }
//         },
//         layout: { padding: { right: 40 } }
//       }
//     });

//     return () => { if (chartRef.current) chartRef.current.destroy(); };
//   }, [langData, showAll]);

//   return (
//     <div className="analysis-card col-6">
//       <div className="trend-header">
//         <p className="card-title">Language Distribution</p>
//         <button 
//           onClick={() => setShowAll(!showAll)}
//           className="toggle-btn"
//         >
//           {showAll ? "Top 5" : "See All"}
//         </button>
//       </div>
//       <div className="chart-container" style={{ height: '300px' }}>
//         <canvas ref={canvasRef}></canvas>
//       </div>
//     </div>
//   );
// }