export default function SarcasmIndicator({ value }) {
  if (value < 0.5) return null;

  return (
    <span className={`tag ${value > 0.5 ? "warning" : ""}`}>
      Sarcasm: {(value * 100).toFixed(0)}%
    </span>
  );
}

