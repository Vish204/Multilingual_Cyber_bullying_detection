export default function EmotionTag({ emotion }) {
  const emojiMap = {
    aggression: "😡",
    distress: "😢",
    neutral: "😐"
  };

  return (
    <div>
      Emotion: {emojiMap[emotion]} {emotion}
    </div>
  );
}