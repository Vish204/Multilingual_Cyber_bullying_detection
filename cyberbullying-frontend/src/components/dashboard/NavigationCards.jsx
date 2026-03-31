import { useNavigate } from "react-router-dom";
import "./dashboard.css";

export default function NavigationCards() {
  const navigate = useNavigate();

  const cards = [
  {
    title: "Moderation",
    description: "Review and take action on flagged posts",
    path: "/moderation",
    icon: "🛡",
  },
  {
    title: "Analysis",
    description: "View insights and AI performance",
    path: "/analysis",
    icon: "📊",
  },
  {
    title: "History",
    description: "Track reviewed posts and decisions",
    path: "/history",
    icon: "🕓",
  },
];

return (
  <div className="nav-cards">
    {cards.map((card, index) => (
      <div
        key={index}
        className={`nav-card nav-${card.title.toLowerCase()}`}
        onClick={() => navigate(card.path)}
      >
        <div className="nav-icon">{card.icon}</div>
        <h3>{card.title}</h3>
        <p>{card.description}</p>
      </div>
    ))}
  </div>
);
}