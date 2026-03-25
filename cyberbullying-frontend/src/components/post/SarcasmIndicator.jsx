export default function SarcasmIndicator({ value }) {
  return (
    <div>
      Sarcasm: {(value * 100).toFixed(1)}%
    </div>
  );
}