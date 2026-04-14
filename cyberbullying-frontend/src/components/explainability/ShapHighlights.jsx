export default function ShapHighlights({ text, triggerWords = [] }) {
  const words = text.split(" ");

  return (
    <p>
      {words.map((word, index) => {
        // Clean punctuation for matching
        const cleanWord = word.toLowerCase().replace(/[^a-z0-9]/g, "");
        
        // Check if this word is in our backend trigger_words array
        const triggerData = triggerWords.find((tw) => tw.word === cleanWord);

        if (triggerData) {
          // Dynamic Opacity: High impact = dark red, low impact = light red
          const opacity = Math.max(0.3, triggerData.impact); 

          return (
            <span
              key={index}
              style={{
                backgroundColor: `rgba(239, 68, 68, ${opacity})`, // Tailwind red-500 with dynamic opacity
                fontWeight: "600",
                marginRight: "4px",
                padding: "0 2px",
                borderRadius: "3px"
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


// export default function ShapHighlights({ text }) {
//   // Dummy highlight words
//   const highlightWords = ["stupid", "hate"];

//   const words = text.split(" ");

//   return (
//     <p>
//       {words.map((word, index) => {
//         const cleanWord = word.toLowerCase().replace(/[^a-z]/g, "");

//         if (highlightWords.includes(cleanWord)) {
//           return (
//             <span
//               key={index}
//               style={{
//                 backgroundColor: "#ffcccc",
//                 fontWeight: "bold",
//                 marginRight: "5px"
//               }}
//             >
//               {word}
//             </span>
//           );
//         }

//         return <span key={index}>{word} </span>;
//       })}
//     </p>
//   );
// }