export default function ContextPanel() {
  return (
    <div className="context-container">
        <h2 className="context-header">Moderation Guide</h2>

        {/* 🔹 Severity Guide */}
        <div className="context-card">
        <h3 className="context-title">Severity Guide</h3>
        <ul>
            <li><span className="low">LOW</span> → Normal / safe</li>
            <li><span className="mild">MILD</span> → Slightly inappropriate</li>
            <li><span className="medium">MEDIUM</span> → Harmful / needs attention</li>
            <li><span className="high">HIGH</span> → Severe / immediate action</li>
        </ul>
        </div>

      {/* 🔹 Emotion Guide */}
      <div className="context-card">
        <h3 className="context-title">Emotion Guide</h3>
        <ul>
          <li><b>Aggression</b> → Attacking / abusive language</li>
          <li><b>Distress</b> → Victim / emotional pain</li>
          <li><b>Neutral</b> → Normal conversation</li>
        </ul>
      </div>

      {/* 🔹 Toxicity Categories */}
      <div className="context-card">
        <h3 className="context-title">Toxicity Categories</h3>
        <ul>
          <li>Insult</li>
          <li>Threat</li>
          <li>Hate Speech</li>
          <li>Harassment</li>
        </ul>
      </div>

      {/* 🔹 Bystander Bullying Scale */}
      <div className="context-card">
        <h3 className="context-title">Bullying Scale</h3>
        <ul>
          <li>Mild → Casual / low harm</li>
          <li>Moderate → Repeated negativity</li>
          <li>Severe → Direct harmful intent</li>
          <li>Critical → Immediate intervention needed</li>
        </ul>
      </div>

    </div>
  );
}