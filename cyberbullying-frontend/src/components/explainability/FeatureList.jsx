export default function FeatureList() {
  const features = [
    { word: "stupid", score: 0.42 },
    { word: "hate", score: 0.31 }
  ];

  return (
    <div>
      <h4>Top Contributing Words</h4>
      <ul>
        {features.map((f, i) => (
          <li key={i}>
            {f.word} (+{f.score})
          </li>
        ))}
      </ul>
    </div>
  );
}