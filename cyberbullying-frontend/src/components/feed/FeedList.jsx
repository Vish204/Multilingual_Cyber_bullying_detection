import { useState } from "react";
import FeedItem from "./FeedItem";

export default function FeedList({ feed, onSelectPost }) {
  const [selectedId, setSelectedId] = useState(null);

  const handleClick = (post) => {
    setSelectedId(post.id);
    onSelectPost(post);
  };

  // ✅ safety check (VERY IMPORTANT)
  if (!feed || feed.length === 0) {
    return <p>No data available</p>;
  }

  return (
    <div>
      {feed.map((post) => (
        <FeedItem
          key={post.id}
          post={post}
          isSelected={selectedId === post.id}
          onClick={() => handleClick(post)}
        />
      ))}
    </div>
  );
}