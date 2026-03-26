export default function EmotionTag({ emotion }) {
  if (!emotion) return null;

  const normalized = emotion.toLowerCase().trim();

  const emojiMap = {
    aggression: "😡",
    distress: "😢",
    neutral: "😐"
  };

  const display =
    normalized.charAt(0).toUpperCase() + normalized.slice(1);

  return (
    <span className={`tag ${normalized}`}>
      {emojiMap[normalized] || "❓"} {display}
    </span>
  );
}