import { useState } from "react";

export default function FilterBar({ filters, setFilters }) {
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [showSearch, setShowSearch] = useState(false);

  // 🔥 BUG 1 FIXED: Checks ALL 9 filters to see if any are active
  const hasActiveFilters = 
    filters.platform !== "all" || 
    filters.label !== "all" || 
    filters.severity !== "all" || 
    filters.language !== "all" ||
    filters.search !== "" ||
    filters.reviewed !== null ||
    filters.alert !== null ||
    filters.content_type !== null ||
    filters.moderator_action !== null;

  // 🔥 BUG 2 FIXED: Clears advanced filters too!
  const handleClear = () => {
    setFilters({ 
      platform: "all", label: "all", severity: "all", language: "all", search: "",
      reviewed: null, alert: null, content_type: null, moderator_action: null
    });
  };

  return (
    <div className="filter-container">
      
      {/* 🔥 FIX 3: Perfect 2x2 Grid (No empty spaces!) */}
      <div className="filter-grid-2x2">
        <select value={filters.platform ?? "all"} onChange={(e) => setFilters({ ...filters, platform: e.target.value })}>
          <option value="all">All Platforms</option>
          <option value="Reddit">Reddit</option>
          <option value="YouTube">YouTube</option>
        </select>

        <select value={filters.label ?? "all"} onChange={(e) => setFilters({ ...filters, label: e.target.value })}>
          <option value="all">All Verdicts</option>
          <option value="cyberbullying">Bullying</option>
          <option value="non-cyberbullying">Safe</option>
        </select>
          
        <select value={filters.severity ?? "all"} onChange={(e) => setFilters({ ...filters, severity: e.target.value })}>
          <option value="all">All Severity</option>
          <option value="severe">Severe</option>
          <option value="moderate">Moderate</option>
          <option value="mild">Mild</option>
        </select>

        <select value={filters.language ?? "all"} onChange={(e) => setFilters({ ...filters, language: e.target.value })}>
          <option value="all">All Languages</option>
          <option value="English">English</option>
          <option value="Hindi_or_marathi">Hindi/Marathi</option>
          <option value="Keywords_hinglish_romanized">Hinglish</option>
          <option value="Bengali">Bengali</option>
          <option value="Marathi">Marathi</option>
          <option value="Telugu">Telugu</option>
          <option value="Tamil">Tamil</option>
          <option value="Gujarati">Gujarati</option>
          <option value="Urdu">Urdu</option>
          <option value="Kannada">Kannada</option>
          <option value="Odia">Oriya</option>
          <option value="Malayalam">Malayalam</option>
          <option value="Punjabi">Punjabi</option>
          <option value="Sanskrit">Sanskrit</option>
        </select>
      </div>

      {/* Row 2: Search Bar + Buttons */}
      <div className="filter-controls-row">
        {showSearch ? (
          <input
            type="text"
            className="filter-search-input"
            placeholder="Search posts..."
            autoFocus
            value={filters.search}
            onChange={(e) => setFilters({ ...filters, search: e.target.value })}
            onBlur={() => { if (!filters.search) setShowSearch(false); }}
            onKeyDown={(e) => { if (e.key === "Escape") setShowSearch(false); }}
          />
        ) : (
          <button className="search-btn" onClick={() => setShowSearch(true)}>🔍</button>
        )}

        <button className={`filter-toggle-btn ${showAdvanced ? "active" : ""}`} 
          onClick={() => setShowAdvanced(!showAdvanced)}
        >
          Filters
        </button>

        <button 
          className={`clear-filters-btn ${hasActiveFilters ? "active" : "disabled"}`}
          onClick={handleClear} 
          disabled={!hasActiveFilters}
        >
          Clear Filters ✖
        </button>
      </div>

      {/* Advanced Filters */}
      {showAdvanced && (
        <div className="advanced-filters">
          <select value={filters.reviewed ?? "all"} onChange={(e) => setFilters({ ...filters, reviewed: e.target.value === "all" ? null : e.target.value === "true" })}>
            <option value="all">Reviewed</option>
            <option value="true">Yes</option>
            <option value="false">No</option>
          </select>
          <select value={filters.alert ?? "all"} onChange={(e) => setFilters({ ...filters, alert: e.target.value === "all" ? null : e.target.value === "true" })}>
            <option value="all">Alert</option>
            <option value="true">Yes</option>
            <option value="false">No</option>
          </select>
          <select value={filters.content_type ?? "all"} onChange={(e) => setFilters({ ...filters, content_type: e.target.value === "all" ? null : e.target.value })}>
            <option value="all">Content Type</option>
            <option value="post">Post</option>
            <option value="comment">Comment</option>
            <option value="tweet">Tweet</option>
          </select>
          <select value={filters.moderator_action ?? "all"} onChange={(e) => setFilters({ ...filters, moderator_action: e.target.value === "all" ? null : e.target.value })}>
            <option value="all">Action</option>
            <option value="ignore">Ignore</option>
            <option value="delete">Delete</option>
            <option value="report">Report</option>
          </select>
        </div>
      )}
    </div>
  );
}

















































// import { useState } from "react";

// export default function FilterBar({ filters, setFilters }) {

//   // ✅ MUST be inside component
//   const [showAdvanced, setShowAdvanced] = useState(false);
//   const [showSearch, setShowSearch] = useState(false);

//   // Check if any filter is actually active
//   const hasActiveFilters = 
//     filters.platform !== "all" || 
//     filters.label !== "all" || 
//     filters.severity !== "all" || 
//     filters.language !== "all" ||
//     filters.search !== "";

//   // Helper to reset everything
//   const handleClear = () => {
//     setFilters({ ...filters, platform: "all", label: "all", severity: "all", language: "all", search: "" });
//   };

//   return (
//   // <div className="filter-container">

//   //   {/* 🔹 Row 1: Main Filters */}
//   //   <div className="filter-row">

//   //     <select
//   //       value={filters.platform}
//   //       onChange={(e) =>
//   //         setFilters({ ...filters, platform: e.target.value })
//   //       }
//   //     >
//   //       <option value="all">All Platforms</option>
//   //       <option value="Reddit">Reddit</option>
//   //       <option value="Twitter">Twitter</option>
//   //       <option value="YouTube">YouTube</option>
//   //     </select>


//   //     {/* 1. NEW: Label Filter */}
//   //       <select
//   //         value={filters.label ?? "all"}
//   //         onChange={(e) => setFilters({ ...filters, label: e.target.value === "all" ? null : e.target.value })}
//   //       >
//   //         <option value="all">All Verdicts</option>
//   //         <option value="cyberbullying">Bullying</option>
//   //         <option value="non-cyberbullying">Safe</option>
//   //       </select>

//   //     <select
//   //       value={filters.severity}
//   //       onChange={(e) =>
//   //         setFilters({ ...filters, severity: e.target.value })
//   //       }
//   //     >
//   //       <option value="all">All Severity</option>
//   //       <option value="severe">Severe</option>
//   //       <option value="moderate">Moderate</option>
//   //       <option value="mild">Mild</option>
//   //       <option value="none">None</option>
//   //     </select>

//   //     {/* <select
//   //       value={filters.language}
//   //       onChange={(e) =>
//   //         setFilters({ ...filters, language: e.target.value })
//   //       }
//   //     >
//   //       <option value="all">All Languages</option>
//   //       <option value="English">English</option>
//   //     </select> */}
//   //     {/* 2. UPDATED: 14 Languages */}
//   //       <select
//   //         value={filters.language ?? "all"}
//   //         onChange={(e) => setFilters({ ...filters, language: e.target.value === "all" ? null : e.target.value })}
//   //       >
//   //         <option value="all">All Languages</option>
//   //         <option value="English">English</option>
//   //         <option value="Hindi">Hindi</option>
//   //         <option value="Bengali">Bengali</option>
//   //         <option value="Marathi">Marathi</option>
//   //         <option value="Telugu">Telugu</option>
//   //         <option value="Tamil">Tamil</option>
//   //         <option value="Gujarati">Gujarati</option>
//   //         <option value="Urdu">Urdu</option>
//   //         <option value="Kannada">Kannada</option>
//   //         <option value="Odia">Odia</option>
//   //         <option value="Malayalam">Malayalam</option>
//   //         <option value="Punjabi">Punjabi</option>
//   //         <option value="Assamese">Assamese</option>
//   //         <option value="Hinglish">Hinglish</option>
//   //       </select>

//   //   </div>


//     <div className="filter-container">
//       <div className="filter-grid">
//         <select value={filters.platform ?? "all"} onChange={(e) => setFilters({ ...filters, platform: e.target.value })}>
//           <option value="all">All Platforms</option>
//           <option value="Reddit">Reddit</option>
//           <option value="YouTube">YouTube</option>
//         </select>

//         <select value={filters.label ?? "all"} onChange={(e) => setFilters({ ...filters, label: e.target.value })}>
//           <option value="all">All Verdicts</option>
//           <option value="cyberbullying">Bullying</option>
//           <option value="non-cyberbullying">Safe</option>
//         </select>
          
//         <select value={filters.severity ?? "all"} onChange={(e) => setFilters({ ...filters, severity: e.target.value })}>
//           <option value="all">All Severity</option>
//           <option value="severe">Severe</option>
//           <option value="moderate">Moderate</option>
//           <option value="mild">Mild</option>
//         </select>

//         <select value={filters.language ?? "all"} onChange={(e) => setFilters({ ...filters, language: e.target.value })}>
//           <option value="all">All Languages</option>
//           <option value="English">English</option>
//           <option value="Hindi">Hindi</option>
//           <option value="Bengali">Bengali</option>
//           <option value="Marathi">Marathi</option>
//           <option value="Telugu">Telugu</option>
//           <option value="Tamil">Tamil</option>
//           <option value="Gujarati">Gujarati</option>
//           <option value="Urdu">Urdu</option>
//           <option value="Kannada">Kannada</option>
//           <option value="Odia">Odia</option>
//           <option value="Malayalam">Malayalam</option>
//           <option value="Punjabi">Punjabi</option>
//           <option value="Assamese">Assamese</option>
//           <option value="Hinglish">Hinglish</option>
//         </select>
//       </div>


















//     {/* 🔹 Row 2: Search + Toggle */}
//     <div className="filter-row">

//       {/* 🔍 Search Toggle */}
//             {showSearch ? (
//               <input
//                 type="text"
//                 placeholder="Search..."
//                 autoFocus
//                 value={filters.search}
//                     onChange={(e) =>
//                       setFilters({ ...filters, search: e.target.value })
//                     }
//                     onBlur={() => {
//                       if (!filters.search) setShowSearch(false);
//                     }}
//                     onKeyDown={(e) => {
//                       if (e.key === "Escape") setShowSearch(false);
//                     }}
  
//               />
//             ) : (
//               <button
//                 className="search-btn"
//                 onClick={() => setShowSearch(true)}
//               >
//                 🔍
//               </button>
              
//             )}


//       <button
//         className="filter-toggle-btn"
//         onClick={() => setShowAdvanced(!showAdvanced)}
//       >
//         <span>Filters</span>
//       </button>

//     </div>

//     {/* 🔹 Advanced Filters */}
//     {showAdvanced && (
//       <div className="advanced-filters">

//         <select
//           value={filters.reviewed ?? "all"}
//           onChange={(e) =>
//             setFilters({
//               ...filters,
//               reviewed:
//                 e.target.value === "all"
//                   ? null
//                   : e.target.value === "true",
//             })
//           }
//         >
//           <option value="all">Reviewed</option>
//           <option value="true">Yes</option>
//           <option value="false">No</option>
//         </select>

//         <select
//           value={filters.alert ?? "all"}
//           onChange={(e) =>
//             setFilters({
//               ...filters,
//               alert:
//                 e.target.value === "all"
//                   ? null
//                   : e.target.value === "true",
//             })
//           }
//         >
//           <option value="all">Alert</option>
//           <option value="true">Yes</option>
//           <option value="false">No</option>
//         </select>

//         <select
//           value={filters.content_type ?? "all"}
//           onChange={(e) =>
//             setFilters({
//               ...filters,
//               content_type:
//                 e.target.value === "all" ? null : e.target.value,
//             })
//           }
//         >
//           <option value="all">Content Type</option>
//           <option value="post">Post</option>
//           <option value="comment">Comment</option>
//         </select>

//         <select
//           value={filters.moderator_action ?? "all"}
//           onChange={(e) =>
//             setFilters({
//               ...filters,
//               moderator_action:
//                 e.target.value === "all" ? null : e.target.value,
//             })
//           }
//         >
//           <option value="all">Action</option>
//           <option value="ignore">Ignore</option>
//           <option value="delete">Delete</option>
//           <option value="report">Report</option>
//         </select>

//       </div>

      
//     )}

//     {/* 3. NEW: Clear Filters Button */}
        
//           <button 
//             className={`clear-filters-btn ${hasActiveFilters ? "active" : "disabled"}`}
//             onClick={handleClear} 
//             disabled={!hasActiveFilters}
//           >
//             Clear Filters ✖
//           </button>
        
//   </div>
// );
// }