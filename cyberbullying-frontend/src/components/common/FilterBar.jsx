import { useState } from "react";

export default function FilterBar({ filters, setFilters }) {

  // ✅ MUST be inside component
  const [showAdvanced, setShowAdvanced] = useState(false);

  return (
    <div style={{ marginBottom: "10px" }}>

      {/* BASIC FILTERS */}
      <div style={{ display: "flex", gap: "10px" }}>

        {/* Platform */}
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

        {/* Severity */}
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

        {/* Language */}
        <select
          value={filters.language}
          onChange={(e) =>
            setFilters({ ...filters, language: e.target.value })
          }
        >
          <option value="all">All Languages</option>
          <option value="English">English</option>
        </select>

        {/* Search */}
        <input
          type="text"
          placeholder="Search..."
          value={filters.search}
          onChange={(e) =>
            setFilters({ ...filters, search: e.target.value })
          }
        />

        {/* Toggle Button */}
        <button onClick={() => setShowAdvanced(!showAdvanced)}>
          ⚙️ More Filters
        </button>

      </div>

      {/* ✅ ADVANCED FILTERS */}
      {showAdvanced && (
        <div
          style={{
            marginTop: "10px",
            display: "flex",
            gap: "10px",
            flexWrap: "wrap",
          }}
        >

          {/* Reviewed */}
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
            <option value="all">Reviewed: All</option>
            <option value="true">Reviewed</option>
            <option value="false">Unreviewed</option>
          </select>

          {/* Alert */}
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
            <option value="all">Alert: All</option>
            <option value="true">Alert</option>
            <option value="false">No Alert</option>
          </select>

          {/* Content Type */}
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

          {/* Moderator Action */}
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