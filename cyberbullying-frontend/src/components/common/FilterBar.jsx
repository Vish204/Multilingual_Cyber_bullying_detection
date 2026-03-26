import { useState } from "react";

export default function FilterBar({ filters, setFilters }) {

  // ✅ MUST be inside component
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [showSearch, setShowSearch] = useState(false);

  return (
  <div className="filter-container">

    {/* 🔹 Row 1: Main Filters */}
    <div className="filter-row">

      <select
        value={filters.platform}
        onChange={(e) =>
          setFilters({ ...filters, platform: e.target.value })
        }
      >
        <option value="all">All Platforms</option>
        <option value="Reddit">Reddit</option>
        <option value="Twitter">Twitter</option>
        <option value="YouTube">YouTube</option>
      </select>

      <select
        value={filters.severity}
        onChange={(e) =>
          setFilters({ ...filters, severity: e.target.value })
        }
      >
        <option value="all">All Severity</option>
        <option value="high">High</option>
        <option value="medium">Medium</option>
        <option value="low">Low</option>
      </select>

      <select
        value={filters.language}
        onChange={(e) =>
          setFilters({ ...filters, language: e.target.value })
        }
      >
        <option value="all">All Languages</option>
        <option value="English">English</option>
      </select>

    </div>

    {/* 🔹 Row 2: Search + Toggle */}
    <div className="filter-row">

      {/* 🔍 Search Toggle */}
            {showSearch ? (
              <input
                type="text"
                placeholder="Search..."
                autoFocus
                value={filters.search}
                    onChange={(e) =>
                      setFilters({ ...filters, search: e.target.value })
                    }
                    onBlur={() => {
                      if (!filters.search) setShowSearch(false);
                    }}
                    onKeyDown={(e) => {
                      if (e.key === "Escape") setShowSearch(false);
                    }}
  
              />
            ) : (
              <button
                className="search-btn"
                onClick={() => setShowSearch(true)}
              >
                🔍
              </button>
              
            )}


      <button
        className="filter-toggle-btn"
        onClick={() => setShowAdvanced(!showAdvanced)}
      >
        <span>Filters</span>
      </button>

    </div>

    {/* 🔹 Advanced Filters */}
    {showAdvanced && (
      <div className="advanced-filters">

        <select
          value={filters.reviewed ?? "all"}
          onChange={(e) =>
            setFilters({
              ...filters,
              reviewed:
                e.target.value === "all"
                  ? null
                  : e.target.value === "true",
            })
          }
        >
          <option value="all">Reviewed</option>
          <option value="true">Yes</option>
          <option value="false">No</option>
        </select>

        <select
          value={filters.alert ?? "all"}
          onChange={(e) =>
            setFilters({
              ...filters,
              alert:
                e.target.value === "all"
                  ? null
                  : e.target.value === "true",
            })
          }
        >
          <option value="all">Alert</option>
          <option value="true">Yes</option>
          <option value="false">No</option>
        </select>

        <select
          value={filters.content_type ?? "all"}
          onChange={(e) =>
            setFilters({
              ...filters,
              content_type:
                e.target.value === "all" ? null : e.target.value,
            })
          }
        >
          <option value="all">Content Type</option>
          <option value="post">Post</option>
          <option value="comment">Comment</option>
        </select>

        <select
          value={filters.moderator_action ?? "all"}
          onChange={(e) =>
            setFilters({
              ...filters,
              moderator_action:
                e.target.value === "all" ? null : e.target.value,
            })
          }
        >
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