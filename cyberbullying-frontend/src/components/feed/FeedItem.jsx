import { FaTwitter, FaYoutube, FaReddit } from "react-icons/fa";

export default function FeedItem({ post, isSelected, onClick }) {

  const getPlatformIcon = () => {
    if (post.platform === "Twitter") return <FaTwitter color="#1DA1F2" />;
    if (post.platform === "YouTube") return <FaYoutube color="#FF0000" />;
    return <FaReddit color="#FF4500" />;
  };

  const getEmotionIcon = () => {
      if (post.emotion === "aggression") return "😡";
      if (post.emotion === "distress") return "😢";
      return "😐";
  };

  return (
    <div
      onClick={onClick}
      className={`feed-item ${isSelected ? "selected" : ""} ${post.severity}`}
    >

      {/* 🔹 Top row */}
      <div className="feed-top">
        {/* LEFT: Platform */}
        <div className="platform-info">
          {getPlatformIcon()}
          <span>{post.platform}</span>
        </div>

{/* right side high/low etc  */}
        {/* <span className={`severity ${post.severity}`}>
          {post.severity.toUpperCase()}
        </span> */}

        {/* RIGHT: verdict */}
        <span className={`verdict ${post.verdict === "BULLYING" ? "bad" : "good"}`}>
          {post.verdict}
        </span>

      </div>

      {/* 🔹 Text */}
      <p className="feed-text">
        {post.text.length > 80
          ? post.text.substring(0, 80) + "..."
          : post.text}
        <div className="feed-ai">


          {/* Emotion (only if confident) */}
          {post.emotion_score >= 0.5 && (
            <span className={`tag ${post.emotion}`}>
              {getEmotionIcon()} {post.emotion}
            </span>
          )}

          {/* Sarcasm */}
          {post.sarcasm >= 0.5 && (
            <span className="tag warning">
              😏 Sarcasm
            </span>
          )}

          {/* Confidence */}
          <span className="tag confidence">
            Confidence: {Math.round(post.confidence * 100)}%
          </span>

        </div>
      </p>

      {/* 🔹 Bottom */}
      <div className="feed-meta">
        <span>{post.time}</span>
      </div>

    </div>
  );
}