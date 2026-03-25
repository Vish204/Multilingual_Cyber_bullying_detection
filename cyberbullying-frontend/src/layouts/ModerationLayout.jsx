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
    },
  ]);

  const [selectedPost, setSelectedPost] = useState(null);

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

  return (
    <div className="moderation-container">

      {/* Top Bar */}
      <div className="top-bar">
        <h2>Live Moderation</h2>
        <button className="fetch-btn">Fetch Data</button>
      </div>

      {/* Main Layout */}
      <div className="main-content">

        {/* ✅ LEFT PANEL (ONLY ONE) */}
        <div className="left-panel">
          <h3>Feed</h3>

          <FilterBar filters={filters} setFilters={setFilters} />

          <FeedList
            feed={filteredFeed}
            onSelectPost={setSelectedPost}
          />
        </div>

        {/* MIDDLE PANEL */}
        <div className="middle-panel">
          <PostDetails post={selectedPost} onAction={handleModeration} />
        </div>

        {/* RIGHT PANEL */}
        <div style={{ flex: 1, borderLeft: "1px solid gray" }}>
          <RightPanel />
        </div>

      </div>
    </div>
  );
}