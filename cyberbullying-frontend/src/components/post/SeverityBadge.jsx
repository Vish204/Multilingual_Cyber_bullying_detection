export default function SeverityBadge({ severity }) {
  let color = "green";

  if (severity === "high") color = "red";
  else if (severity === "medium") color = "orange";

  return (
    <span style={{
      backgroundColor: color,
      color: "white",
      padding: "4px 8px",
      borderRadius: "5px",
      fontSize: "12px"
    }}>
      {severity.toUpperCase()}
    </span>
  );
}