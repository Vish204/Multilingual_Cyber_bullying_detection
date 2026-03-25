export default function FeedItem({ post, isSelected, onClick }) {
  return (
    <div
      onClick={onClick}
      style={{
        padding: "10px",
        marginBottom: "8px",
        cursor: "pointer",
        border: "1px solid #ccc",
        backgroundColor: isSelected ? "#e6f0ff" : "white",
      }}
    >
      <p>{post.text}</p>
      <small>{post.platform} | {post.time}</small>
    </div>
  );
}