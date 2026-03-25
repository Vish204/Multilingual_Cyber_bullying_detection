import { useState } from "react";
import "./moderation.css";
import FeedList from "../components/feed/FeedList";
import PostDetails from "../components/post/PostDetails";
import RightPanel from "../components/context/RightPanel";
import FilterBar from "../components/common/FilterBar";

export default function ModerationLayout() {

  // ✅ Feed FIRST
  const [feed, setFeed] = useState([
    {
      id: 1,
      text: "You are so stupid",
      platform: "Reddit",
      time: "2m ago",
      severity: "high",
      language: "English",
      reviewed: false,
      alert: true,
      content_type: "post",
      moderator_action: null,

      verdict: "BULLYING",
      confidence: 0.87,
      emotion: "aggression",
      emotion_score: 0.8,
      sarcasm: 0.2,
    },
    {
      id: 2,
      text: "I hate this person",
      platform: "Twitter",
      time: "5m ago",
      severity: "medium",
      language: "English",
      reviewed: true,
      alert: true,
      content_type: "comment",
      moderator_action: "delete",
    
      verdict: "BULLYING",
      confidence: 0.87,
      emotion: "distress",
      emotion_score: 0.8,
      sarcasm: 0.6,
    },
    {
      id: 3,
      text: "This is normal text",
      platform: "YouTube",
      time: "10m ago",
      severity: "low",
      language: "English",
      reviewed: false,
      alert: false,
      content_type: "comment",
      moderator_action: "ignore",

      verdict: "NON-BULLYING",
      confidence: 0.12,
      emotion: "NEUTRAL",
      emotion_score: 0.7,
      sarcasm: 0.2,
    },
  ]);

  const [selectedPost, setSelectedPost] = useState(null);

  const [isLive, setIsLive] = useState(false);
  const [alert, setAlert] = useState(null);

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

  // ✅ Moderation
  const handleModeration = (action, postId) => {
    console.log("Moderating:", action, postId);

    const updatedFeed = feed.filter((item) => item.id !== postId);
    setFeed(updatedFeed);

    if (updatedFeed.length > 0) {
      setSelectedPost(updatedFeed[0]);
    } else {
      setSelectedPost(null);
    }
  };

  const triggerAlert = (post) => {
  if (post.severity === "high") {
    setAlert({
      message: `${post.platform} | ${post.verdict} | ${Math.round(post.confidence * 100)}%`,
      post: post,
    });

    // Auto hide after 5 sec
    setTimeout(() => setAlert(null), 5000);
  }
};


  return (
    <div className="moderation-container">

        {alert && (
              <div className="alert-banner">
                🚨 {alert.message}
                <button onClick={() => setAlert(null)}>✖</button>
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
          <h3>Feed</h3>

          <FilterBar filters={filters} setFilters={setFilters} />

          <FeedList
            feed={filteredFeed}
            onSelectPost={(post) => {
              setSelectedPost(post);
              triggerAlert(post);
            }}
          />

        </div>

        {/* MIDDLE PANEL */}
        <div className="panel middle-panel">
          <PostDetails post={selectedPost} onAction={handleModeration} />
        </div>

        {/* RIGHT PANEL */}
        <div className="panel right-panel">
          <RightPanel />
        </div>

      </div>
    </div>
  );
}