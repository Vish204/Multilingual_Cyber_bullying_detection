export default function ShapHighlights({ text }) {
  // Dummy highlight words
  const highlightWords = ["stupid", "hate"];

  const words = text.split(" ");

  return (
    <p>
      {words.map((word, index) => {
        const cleanWord = word.toLowerCase().replace(/[^a-z]/g, "");

        if (highlightWords.includes(cleanWord)) {
          return (
            <span
              key={index}
              style={{
                backgroundColor: "#ffcccc",
                fontWeight: "bold",
                marginRight: "5px"
              }}
            >
              {word}
            </span>
          );
        }

        return <span key={index}>{word} </span>;
      })}
    </p>
  );
}