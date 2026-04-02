import { useState } from "react";
import "./moderation.css";
import FeedList from "../components/feed/FeedList";
import PostDetails from "../components/post/PostDetails";
import RightPanel from "../components/context/RightPanel";
import FilterBar from "../components/common/FilterBar";
import ContextPanel from "../components/context/ContextPanel";
import { FaDownload } from "react-icons/fa";

import { fetchPosts, moderatePost } from "../services/api";
import { transformPost } from "../services/transform";

import { useEffect } from "react";

export default function ModerationLayout() {

  // ✅ Feed FIRST
  // const [feed, setFeed] = useState([
  //   {
  //     id: 1,
  //     text: "You are so stupid",
  //     platform: "Reddit",
  //     time: "2m ago",
  //     severity: "high",
  //     language: "English",
  //     reviewed: false,
  //     alert: true,
  //     content_type: "post",
  //     moderator_action: null,

  //     verdict: "BULLYING",
  //     confidence: 0.87,
  //     emotion: "aggression",
  //     emotion_score: 0.8,
  //     sarcasm: 0.6,
  //     saved: false,
  //   },
  //   {
  //     id: 2,
  //     text: "I hate this person",
  //     platform: "Twitter",
  //     time: "5m ago",
  //     severity: "medium",
  //     language: "English",
  //     reviewed: false,
  //     alert: true,
  //     content_type: "tweet",
  //     moderator_action: null,
    
  //     verdict: "BULLYING",
  //     confidence: 0.87,
  //     emotion: "distress",
  //     emotion_score: 0.8,
  //     sarcasm: 0.6,
  //     saved: false,
  //   },
  //   {
  //     id: 3,
  //     text: "This is normal text",
  //     platform: "YouTube",
  //     time: "10m ago",
  //     severity: "low",
  //     language: "English",
  //     reviewed: true,
  //     alert: false,
  //     content_type: "comment",
  //     moderator_action: "ignore",

  //     verdict: "NON-BULLYING",
  //     confidence: 0.12,
  //     emotion: "neutral",
  //     emotion_score: 0.7,
  //     sarcasm: 0.2,
  //     saved: false,
  //   },
  // ]);

  const [feed, setFeed] = useState([]);

  useEffect(() => {
  async function loadFeed() {
    try {
      const data = await fetchPosts();
      const transformed = data.map(transformPost);
      setFeed(transformed);
    } catch (err) {
      console.error("Error fetching posts:", err);
    }
  }

  loadFeed();
}, []);

  const [selectedPost, setSelectedPost] = useState(null);
  const [actionMessage, setActionMessage] = useState(null); //toast for letting moderator know their action worked
  const [isLive, setIsLive] = useState(false);
  const [alert, setAlert] = useState(null);
  const [alertedIds, setAlertedIds] = useState(new Set());

  // ✅ Filters
  const [filters, setFilters] = useState({
    platform: "all",
    severity: "all",
    language: "all",
    search: "",

    reviewed: null,
    alert: null,
    moderator_action: null,
    content_type: null,
  });

  // ✅ Filtering AFTER feed is defined
  const filteredFeed = feed.filter((post) => {

    if (filters.platform !== "all" && post.platform !== filters.platform) {
      return false;
    }

    if (filters.severity !== "all" && post.severity !== filters.severity) {
      return false;
    }

    if (filters.language !== "all" && post.language !== filters.language) {
      return false;
    }

    if (
      filters.search &&
      !post.text.toLowerCase().includes(filters.search.toLowerCase())
    ) {
      return false;
    }
    //hidden filters for future use
    if (filters.reviewed !== null && post.reviewed !== filters.reviewed) {
      return false;
    }

    if (filters.alert !== null && post.alert !== filters.alert) {
      return false;
    }

    if (filters.content_type && post.content_type !== filters.content_type) {
      return false;
    }

    if (
      filters.moderator_action &&
      post.moderator_action !== filters.moderator_action
    ) {
      return false;
    }

    return true;
  });


const handleExport = () => {
  const dataToExport = filteredFeed;

  if (!dataToExport.length) {
    alert("No data to export");
    return;
  }

  const csvRows = [];

  const headers = Object.keys(dataToExport[0]);
  csvRows.push(headers.join(","));

  dataToExport.forEach((row) => {
    const values = headers.map((header) =>
      `"${(row[header] ?? "").toString().replace(/"/g, '""')}"`
    );
    csvRows.push(values.join(","));
  });

  const blob = new Blob([csvRows.join("\n")], { type: "text/csv" });
  const url = URL.createObjectURL(blob);

  const a = document.createElement("a");
  a.href = url;
  a.download = "moderation_export.csv";
  a.click();

  URL.revokeObjectURL(url);
};


  // ✅ Moderation
const handleModeration = async (action, postId, reason = "") => {
  
  console.log("Moderating:", action, postId);

  try {
    await moderatePost(postId, action);
  } catch (err) {
    console.error("API error:", err);
  }

//   if (action === "export") {
//   const reviewedPosts = feed.filter(p => p.reviewed);

//   if (reviewedPosts.length === 0) {
//     alert("No reviewed posts to export");
//     return;
//   }

//   const dataStr = JSON.stringify(reviewedPosts, null, 2);
//   const blob = new Blob([dataStr], { type: "application/json" });

//   const url = URL.createObjectURL(blob);
//   const a = document.createElement("a");
//   a.href = url;
//   a.download = "reviewed_posts.json";
//   a.click();

//   URL.revokeObjectURL(url);
//   return;
// }

    if (action === "export") {
      const reviewedPosts = feed.filter(p => p.reviewed);

      if (reviewedPosts.length === 0) {
        alert("No reviewed posts to export");
        return;
      }

      // 🔹 Define CSV headers
      const headers = [
        "id",
        "text",
        "platform",
        "severity",
        "language",
        "verdict",
        "confidence",
        "emotion",
        "sarcasm",
        "reviewed",
        "saved",
        "moderator_action"
      ];

      // 🔹 Convert to CSV rows
      const rows = reviewedPosts.map(post =>
        headers.map(field => `"${post[field] ?? ""}"`).join(",")
      );

      // 🔹 Combine header + rows
      const csvContent = [
        headers.join(","), 
        ...rows
      ].join("\n");

      // 🔹 Create file + download
      const blob = new Blob([csvContent], { type: "text/csv" });
      const url = URL.createObjectURL(blob);

      const a = document.createElement("a");
      a.href = url;
      a.download = "reviewed_posts.csv";
      a.click();

      URL.revokeObjectURL(url);
    }

    if (action === "delete") {
      // 🔥 ONLY delete removes from feed
      const updatedFeed = feed.filter((item) => item.id !== postId);
      setFeed(updatedFeed);
      //so that it doesnt jump to next post after deleting
      // if (updatedFeed.length > 0) {
      //   setSelectedPost(updatedFeed[0]);
      // } else {
      //   setSelectedPost(null);
      // }
      setSelectedPost(null);

    } else if (action === "ignore" || action === "report" || action === "save") {
      // 🔥 update post instead of removing
      const updatedFeed = feed.map((item) => {
        if (item.id === postId) {
          return {
            ...item,
            reviewed: true,
            saved: action === "save",
            moderator_action: action,
            reason: reason || "", 
          };
        }
        return item;
      });

      setFeed(updatedFeed);

      let message = "";

      if (action === "ignore") message = "Post ignored";
      if (action === "report") message = "Post reported";
      if (action === "save") message = "Post saved";

      setActionMessage(message);

      setTimeout(() => setActionMessage(null), 2000);

      // keep same post selected (important UX)
      const updatedPost = updatedFeed.find(p => p.id === postId);
      setSelectedPost(updatedPost);
    }
};

    const triggerAlert = (post) => {
      if (post.severity === "high") {
        setAlert((prev) => {
          // 🔥 prevent overwriting existing alert
          if (prev && prev.post.id === post.id) return prev;

          return {
            message: `${post.platform} | ${post.verdict} | ${Math.round(post.confidence * 100)}%`,
            post: post,
          };
        });
      }
    };
    useEffect(() => {
      // 🔥 Get latest high severity post (from bottom = newest)
      const latestHigh = [...feed].reverse().find(
        (post) => post.severity === "high" && !alertedIds.has(post.id)
      );

      if (latestHigh) {
        triggerAlert(latestHigh);

        setAlertedIds((prev) => {
          const updated = new Set(prev);
          updated.add(latestHigh.id);
          return updated;
        });
      }
    }, [feed, alertedIds]);

    // 🔥 Alert cleanup (IMPORTANT FIX)
    useEffect(() => {
      if (!alert) return;

      const exists = feed.some(p => p.id === alert.post.id);

      if (!exists) {
        setAlert(null);
      }

    }, [feed, alert]);


  return (
    <div className="moderation-container">

      {alert && (
        <div
          className="alert-banner"
            onClick={() => {
              const postExists = feed.find(p => p.id === alert.post.id);

              if (postExists) {
                setSelectedPost(postExists);
              } else {
                // setSelectedPost(null); // prevent ghost UI
                    setAlert(null);   // 🔥 remove invalid alert
                    return;
              }

              setAlert(null);
            }}
                  >
          🚨 {alert.message}

          <button
            onClick={(e) => {
              e.stopPropagation(); // prevent triggering click
              setAlert(null);
            }}
          >
            ✖
          </button>
        </div>
      )}


      {actionMessage && (
        <div className="action-toast">
          {actionMessage}
        </div>
      )}
      
      {/* Top Bar */}
      <div className="top-bar">

      </div>

      {/* Main Layout */}
      <div className="main-content">

        {/* ✅ LEFT PANEL (ONLY ONE) */}
        <div className="panel left-panel">
          <div className="live-indicator">
            <span className="live-dot"></span>
            Live Feed
          </div>
        <h2>Live Moderation</h2>
        <button
        className="fetch-btn"
        onClick={() => setIsLive(!isLive)}
        >
          {isLive ? "Stop Stream" : "Start Stream"}
        </button>
        <div className="feed-header">
          <h3>Feed</h3>
        <button className="btn export" onClick={handleExport}>
          <FaDownload size={14} />
           Export CSV
        </button>
        </div>
          <FilterBar filters={filters} setFilters={setFilters} />

          <FeedList
            feed={filteredFeed}
            selectedPost={selectedPost}
            onSelectPost={(post) => {
              if (selectedPost?.id === post.id) {
                setSelectedPost(null);   // 🔥 deselect on second click
              } else {
                setSelectedPost(post);
              }
            }}
          />

        </div>

        {/* MIDDLE PANEL */}
        <div className="panel middle-panel">
          <PostDetails post={selectedPost} onAction={handleModeration} />
        </div>

        {/* RIGHT PANEL */}
        <div className="panel right-panel">
          <ContextPanel />
        </div>

      </div>
    </div>
  );
}