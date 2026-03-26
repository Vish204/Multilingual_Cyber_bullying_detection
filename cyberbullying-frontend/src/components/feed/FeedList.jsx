import FeedItem from "./FeedItem";

export default function FeedList({ feed, onSelectPost, selectedPost }) {

  // ✅ safety check
  if (!feed || feed.length === 0) {
    return <p>No data available</p>;
  }

  return (
    <div className="feed-list">
      {feed.map((post) => (
        <FeedItem
          key={post.id}
          post={post}
          isSelected={selectedPost?.id === post.id}
          onClick={() => onSelectPost(post)}
        />
      ))}
    </div>
  );
}