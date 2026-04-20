import { useState } from "react";
import { useLocation } from "react-router-dom";

import "./moderation.css";
import FeedList from "../components/feed/FeedList";
import PostDetails from "../components/post/PostDetails";
import RightPanel from "../components/context/RightPanel";
import FilterBar from "../components/common/FilterBar";
import ContextPanel from "../components/context/ContextPanel";
import { FaDownload } from "react-icons/fa";


import { fetchPosts, moderatePost, collectData } from "../services/api";
import { exportPosts } from "../services/api";
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
  

  const [selectedPost, setSelectedPost] = useState(null);
  const [actionMessage, setActionMessage] = useState(null); //toast for letting moderator know their action worked
  const [isLive, setIsLive] = useState(false);
  // const [alert, setAlert] = useState(null);
  // const [alertedIds, setAlertedIds] = useState(new Set());
  const [isLoading, setIsLoading] = useState(false);
  const [loadingText, setLoadingText] = useState("Start Stream");

//For Dashboard Page alert navigation
 const location = useLocation();

  useEffect(() => {
    // 👇 1. Check if the router handed us a full post object
    if (location.state && location.state.alertPost) {
      
      const targetPost = location.state.alertPost;
      console.log("🎯 ALERT INTERCEPTED FROM ROUTER:", targetPost);

      // 👇 2. Instantly drop it into the UI (No API fetch needed!)
      setFeed([targetPost]);
      setSelectedPost(targetPost);
      
      // 👇 3. Turn off the live stream so it doesn't get buried
      setIsLive(false); 
      setLoadingText("Start Stream");
      
      // 👇 4. Clean up the router state so a refresh doesn't trigger it again
      window.history.replaceState({}, document.title)
    }
  }, [location.state]);



  const loadFeed = async () => {
    try {
      const data = await fetchPosts();
      const transformed = data.map(transformPost);
      setFeed(transformed);
    } catch (err) {
      console.error("Error fetching posts:", err);
    }
  };

  // const handleStreamToggle = async () => {
  //   if (!isLive) {
  //     // START stream
  //     setIsLive(true);
  //     await loadFeed();
  //   } else {
  //     // STOP stream
  //     setIsLive(false);
  //     setFeed([]); // optional: clear feed
  //   }
  // };
  const handleStreamToggle = () => {
    if (!isLive) {
      console.log("🧹 STREAM STARTING: Aggressively nuking old UI...");
      setSelectedPost(null);  // Instantly kill middle panel
      setFeed([]);            // Instantly kill left panel & alert banner
    }
    setIsLive(prev => !prev);
  };

  // ✅ Filters
  const [filters, setFilters] = useState({
    platform: "all",
    label: "all",
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

    // 🔥 NEW: Check the Verdict/Label filter
    if (filters.label !== "all") {
      // Safely check post.label (or map verdict back to backend standard just in case)
      const postLabel = post.label || (post.verdict === "BULLYING" ? "cyberbullying" : "non-cyberbullying");
      
      if (postLabel !== filters.label) {
        return false;
      }
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
  // console.log("FEED:", feed);
  // console.log("FILTERED:", filteredFeed);

  // 🔥 THE SMART QUEUE: Automatically finds unreviewed high-priority alerts
  const pendingAlerts = feed.filter(post => post.alert === true && !post.reviewed);
  const currentAlert = pendingAlerts.length > 0 ? pendingAlerts[0] : null;



    const selectedIndex = filteredFeed.findIndex(
      (p) => p.id === selectedPost?.id
    );

    // const activeAlerts = feed.filter(post => post.alert === true && !post.reviewed);

  const handleExport = async () => {
    try {
      // Calls the python backend directly for a perfectly formatted CSV!
      const exportFilters = { ...filters, limit: 15 };

      const blob = await exportPosts(exportFilters); 
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "cyberbullying_export.csv";
      a.click();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      console.error("Export failed:", err);
    }
  };


  // ✅ Moderation
// const handleModeration = async (action, postId, reason = "") => {
  
//   console.log("Moderating:", action, postId);

//   try {
//     if (action !== "save") {
//     await moderatePost(postId, action, reason);
//     }
//   } catch (err) {
//     console.error("Backend error:", err);
//   }

//     if (action === "export") {
//       const reviewedPosts = feed.filter(p => p.reviewed);

//       if (reviewedPosts.length === 0) {
//         alert("No reviewed posts to export");
//         return;
//       }

//       // 🔹 Define CSV headers
//       const headers = [
//         "id",
//         "text",
//         "platform",
//         "severity",
//         "language",
//         "verdict",
//         "confidence",
//         "emotion",
//         "sarcasm",
//         "reviewed",
//         "saved",
//         "moderator_action"
//       ];

//       // 🔹 Convert to CSV rows
//       const rows = reviewedPosts.map(post =>
//         headers.map(field => `"${post[field] ?? ""}"`).join(",")
//       );

//       // 🔹 Combine header + rows
//       const csvContent = [
//         headers.join(","), 
//         ...rows
//       ].join("\n");

//       // 🔹 Create file + download
//       const blob = new Blob([csvContent], { type: "text/csv" });
//       const url = URL.createObjectURL(blob);

//       const a = document.createElement("a");
//       a.href = url;
//       a.download = "reviewed_posts.csv";
//       a.click();

//       URL.revokeObjectURL(url);
//     }

//     if (action === "delete") {
//       // 🔥 ONLY delete removes from feed

//       // const updatedFeed = feed.filter((item) => item.id !== postId);
//       const updatedFeed = feed.filter(post => post.id !== postId);
//       setFeed(updatedFeed);
//   if (selectedPost?.id === postId) {
//     setSelectedPost(null);
//   }

//   return;

//     } 
//     // else if (action === "save" && item.moderator_action === "delete") {
//     //     return item;
//     // } 
//     else if (action === "ignore" || action === "report" || action === "save") {
//       // 🔥 update post instead of removing
//       const updatedFeed = feed.map((item) => {
//         if (item.id === postId) {

//           // 🟢 SAVE → independent toggle
//           if (action === "save") {
//             return {
//               ...item,
//               saved: !item.saved, // toggle save
//             };
//           }

//           // 🔴 DELETE handled separately (already done)

//           // 🔴 IGNORE / REPORT → real moderation
//           return {
//             ...item,
//             reviewed: true,
//             moderator_action: action,
//             reason: reason || "",
//           };
//         }

//         return item;
//       });

//       setFeed(updatedFeed);
//       // await loadFeed();
//       let message = "";

//       if (action === "ignore") message = "Post ignored";
//       if (action === "report") message = "Post reported";
//       if (action === "save") message = "Post saved";

//       setActionMessage(message);

//       setTimeout(() => setActionMessage(null), 2000);

//       // keep same post selected (important UX)
//       const updatedPost = updatedFeed.find(p => p.id === postId);
//       setSelectedPost(updatedPost);
//     }
// };


const handleModeration = async (action, postId, reason = "") => {
    console.log("Moderating:", action, postId);

    // 1. Find current post state
    const postToUpdate = feed.find((p) => p.id === postId);
    if (!postToUpdate) return;

    // 2. Determine new state
    let newAction = postToUpdate.moderator_action;
    let newSaved = postToUpdate.saved;

    // 🔥 1. Add the Overrule check here
    const isOverruled = 
      (postToUpdate.verdict === "BULLYING" && action === "ignore") || 
      (postToUpdate.verdict === "NON-BULLYING" && (action === "delete" || action === "report"));

    if (action === "save") {
      newSaved = !postToUpdate.saved; // Toggle save state
    } else {
      newAction = action; // Ignore, Delete, or Report
      // 🔥 2. Auto-save if they corrected the AI
      if (isOverruled) newSaved = true; 
    }

    // 3. Send to MongoDB (Wait for it to finish)
    try {
      await moderatePost(postId, newAction || "pending", reason, newSaved);
    } catch (err) {
      console.error("Backend DB error:", err);
    }

    // 4. Update UI: If delete, remove it from the screen entirely
    if (action === "delete") {
      setFeed(feed.filter((p) => p.id !== postId));
      if (selectedPost?.id === postId) setSelectedPost(null);
      return;
    }

    // 5. Update UI: For ignore/report/save, update the post in place
    const updatedFeed = feed.map((item) => {
      if (item.id === postId) {
        return {
          ...item,
          reviewed: true,
          moderator_action: newAction,
          reason: reason || item.reason,
          saved: newSaved,
        };
      }
      return item;
    });

    setFeed(updatedFeed);
    setSelectedPost(updatedFeed.find((p) => p.id === postId));

    // Show Toast Message
    let message = "";
    if (action === "ignore") message = isOverruled ? "Ignored & Auto-Saved " : "Post Ignored";
    if (action === "report") message = isOverruled ? "Reported & Auto-Saved " : "Post Reported";
    if (action === "save") message = newSaved ? "Saved to Curated Dataset " : "Removed from Dataset";

    setActionMessage(message);
    setTimeout(() => setActionMessage(null), 2000);
  };

    // const triggerAlert = (post) => {
    //   if (post.severity === "severe") {
    //     setAlert((prev) => {
    //       // 🔥 prevent overwriting existing alert
    //       if (prev && prev.post.id === post.id) return prev;

    //       return {
    //         message: `${post.platform} | ${post.verdict} | ${Math.round(post.confidence * 100)}%`,
    //         post: post,
    //       };
    //     });
    //   }
    // };
    // useEffect(() => {
    //   // 🔥 Get latest high severity post (from bottom = newest)
    //   const latestHigh = [...feed].reverse().find(
    //     (post) => post.severity === "severe" && !alertedIds.has(post.id)
    //   );

    //   if (latestHigh) {
    //     triggerAlert(latestHigh);

    //     setAlertedIds((prev) => {
    //       const updated = new Set(prev);
    //       updated.add(latestHigh.id);
    //       return updated;
    //     });
    //   }
    // }, [feed, alertedIds]);

    // 🔥 Alert cleanup (IMPORTANT FIX)
    // useEffect(() => {
    //   if (!alert) return;

    //   const exists = feed.some(p => p.id === alert.post.id);

    //   if (!exists) {
    //     setAlert(null);
    //   }

    // }, [feed, alert]);
    
      // useEffect(() => {
      //   if (!isLive) return;

      //   async function runStream() {
      //     try {
      //       setIsLoading(true);

      //       console.log("🚀 Collecting 15 posts...");

      //       await collectData();

      //       const data = await fetchPosts();
      //       console.log("RAW DATA:", data);

      //       const transformed = data.map(transformPost);
      //       console.log("TRANSFORMED:", transformed);

      //       setFeed(transformed);
      //       setSelectedPost(null);

      //     } catch (err) {
      //       console.error("Stream error:", err);
      //     } finally {
      //       // 🔥 FIX HERE
      //       setTimeout(() => {
      //         setIsLive(false);
      //         setIsLoading(false);
      //         console.log("⏹ Stream stopped automatically");
      //       }, 100);
      //     }
      //   }

      //   runStream();
      // }, [isLive]);


      // 🔥 AUTO-SCROLL TO SELECTED POST
      useEffect(() => {
        if (selectedPost) {
          const element = document.getElementById(`feed-post-${selectedPost.id}`);
          if (element) {
            element.scrollIntoView({ behavior: "smooth", block: "center" });
          }
        }
      }, [selectedPost]);

      useEffect(() => {
        if (!isLive) {
          setLoadingText("Start Stream");
          return;
        }

        let isMounted = true;

        async function runStream() {
          try {
            console.log(" 1. runStream started");
            setIsLoading(true);

            //  P15: Instantly wipe the old UI (Middle Panel & Alert Banner)
            setSelectedPost(null);
            setFeed([]);


            console.log(" Collecting 15 posts...");

            //  NEW: Dynamic Loading Sequence!
            setLoadingText(" Connecting to Social APIs...");
            const timer1 = setTimeout(() => { if (isMounted) setLoadingText("⚙️ Running XLM-R Pipeline..."); }, 3000);
            const timer2 = setTimeout(() => { if (isMounted) setLoadingText("⚡ Processing Multilingual Content..."); }, 7000);

            // Run collector & fetch
            await collectData();
            const data = await fetchPosts();
            console.log("RAW DATA:", data);

            const transformed = data.map(transformPost);

            if (isMounted) {
              setFeed(transformed);
              setSelectedPost(null);
            }

            // Cleanup timers
            clearTimeout(timer1);
            clearTimeout(timer2);

          } catch (err) {
            console.error("Stream error:", err);
          } finally {
            if (isMounted) {
              setIsLive(false);
              setIsLoading(false);
              console.log("⏹ Stream stopped automatically");
              setLoadingText("Start Stream");
              
            }
          }
        }

        runStream();
        
        return () => { isMounted = false; };
      }, [isLive]);

  return (
    <div className="moderation-container">

      {/* {alert && (
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
      )} */}


      {/* 🔥 THE NEW SMART ALERT BANNER */}
      {/* {currentAlert && (
        <div 
          className="alert-banner" 
          onClick={() => setSelectedPost(currentAlert)}
        >
          <div style={{ display: "flex", alignItems: "center" }}>
            <span style={{ fontSize: "11px", fontWeight: "bold", marginRight: "12px", letterSpacing: "1px", opacity: 0.9 }}>
              HIGH PRIORITY
            </span>
            🚨 {currentAlert.platform} | {currentAlert.verdict} | {Math.round(currentAlert.confidence * 100)}%
            <span className="alert-banner-count">
              {pendingAlerts.length} Pending
            </span>
          </div>

          <button 
            className="alert-banner-btn"
            onClick={(e) => {
              e.stopPropagation(); // Prevents the parent div click from firing twice
              setSelectedPost(currentAlert);
            }}
          >
            Review Next ➔
          </button>
        </div>
      )} */}
      {/*  THE NEW SMART ALERT BANNER */}
      {currentAlert && (
        <div 
          className="alert-banner" 
          onClick={() => setSelectedPost(currentAlert)}
        >
          <div style={{ display: "flex", alignItems: "center" }}>
            <span style={{ fontSize: "11px", fontWeight: "bold", marginRight: "12px", letterSpacing: "1px", opacity: 0.9 }}>
              HIGH PRIORITY
            </span>
            🚨 {currentAlert.platform} | {currentAlert.verdict} | {Math.round(currentAlert.confidence * 100)}%
            <span className="alert-banner-count">
              {pendingAlerts.length} Pending
            </span>
          </div>

          <button 
            className="alert-banner-btn"
            onClick={(e) => {
              e.stopPropagation(); // Prevents the parent div click from firing
              
              if (pendingAlerts.length > 1) {
                // If more left, select it so the user can review it
                setSelectedPost(currentAlert);
              } else {
                // If it's the last one, dismiss the banner by un-flagging it in UI state
                setFeed(feed.map(p => p.id === currentAlert.id ? { ...p, alert: false } : p));
              }
            }}
          >
            {pendingAlerts.length > 1 ? "Review Next ➔" : "✖ Close"}
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
          {/* <div className="live-indicator">
            <span className="live-dot"></span>
            Live Feed
          </div> */}

            {/* 🔥 Dynamic Live Indicator */}
          <div className={`live-indicator ${!isLive ? "paused" : ""}`}>
            {isLive ? <span className="live-dot"></span> : <span className="paused-icon">⏸️</span>}
            {isLive ? "Live Fetching" : "Stream Paused"}
          </div>

        <h2>Live Moderation</h2>
        <button
        className="fetch-btn"
        onClick={handleStreamToggle}
        disabled={isLoading}
        >
          {/* {isLoading ? "Loading..." : "Start Stream"} */}
          {loadingText}
        </button>
        {/* <div className="feed-header">
          <h3>Feed</h3>

            <span className="feed-count">
              {selectedIndex >= 0
                ? `${selectedIndex + 1} / ${filteredFeed.length}`
                : `0 / ${filteredFeed.length}`}
            </span>


        <button className="btn export" onClick={handleExport}>
          <FaDownload size={14} />
           Export CSV
        </button>
        </div> */}
        <div className="feed-header">
          {/* 🔥 FIX 4: Groups "Feed" and the Badge tightly together on the left */}
          <div className="feed-header-left">
            <h3>Feed</h3>
            <span className="feed-count-badge">
              {selectedIndex >= 0
                ? `${selectedIndex + 1} / ${filteredFeed.length}`
                : `0 / ${filteredFeed.length}`}
            </span>
          </div>

          <button className="btn export" onClick={handleExport}>
            <FaDownload size={14} />
             Export CSV
          </button>
        </div>





          <FilterBar filters={filters} setFilters={setFilters} />
        {/* {!isLive ? (
          <div className="empty-state">
            Click "Start Stream" to begin monitoring
          </div>
        ) : (
          <FeedList
            key={filteredFeed.length} 
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
          )} */}
          {/* 🔥 FIX 3: Dynamic Left Panel State */}
          {isLoading ? (
            <div className="loading-container">
              <div className="loading-spinner">⏳</div>
              <div>{loadingText}</div>
            </div>
          ) : feed.length === 0 ? (
            <div className="empty-state">
              Click "Start Stream" to begin monitoring
            </div>
          ) : (
            <FeedList
              key={filteredFeed.length}
              feed={filteredFeed}
              selectedPost={selectedPost}
              onSelectPost={(post) => {
                if (selectedPost?.id === post.id) {
                  setSelectedPost(null);
                } else {
                  setSelectedPost(post);
                }
              }}
            />
          )}
          
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