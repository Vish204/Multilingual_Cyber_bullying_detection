import "./dashboard.css";

export default function DashboardHeader() {
  return (
    <div className="dashboard-header">
      <h1 className="dashboard-title">
        AI Cyberbullying Detection & Moderation System
      </h1>

      <p className="dashboard-subtitle">
        {/* Detect harmful content using AI (emotion, sarcasm, context) and assist
        moderators with explainable decisions. */}
        Detect and moderate harmful content using a machine learning pipeline 
        (Student XGBoost model) enhanced with emotion and sarcasm signals 
        for explainable and context-aware decisions.
      </p>

      <div className="dashboard-badges">
        <span>Explainable AI (XAI)</span>
        <span>Real-time Alerts</span>
        <span>Multi-platform</span>
        <span>Moderator-in-loop</span>
      </div>
    </div>
  );
}