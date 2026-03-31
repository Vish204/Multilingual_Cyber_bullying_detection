import "./dashboard.css";

export default function SystemFlow() {
    const steps = [
    "Input (Social Media Posts)",
    // "Preprocessing & Feature Extraction",
    "Cyberbullying Model (Student XGBoost)",
    "Auxiliary Signals (Emotion + Sarcasm)",
    "Fusion & Decision",
    "Moderator Action",
    ];

  return (
    <div className="flow-section">
      <h2 className="section-title">How the System Works</h2>

      <div className="flow-container">
        {steps.map((step, index) => (
          <div key={index} className="flow-step">
            <span className="flow-text">{step}</span>
            {index !== steps.length - 1 && (
              <span className="flow-arrow">→</span>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}