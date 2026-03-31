import DashboardHeader from "../components/dashboard/DashboardHeader";
import NavigationCards from "../components/dashboard/NavigationCards";
import StatsOverview from "../components/dashboard/StatsOverview";
import RecentAlerts from "../components/dashboard/RecentAlerts";
import SystemFlow from "../components/dashboard/SystemFlow";
import "../components/dashboard/dashboard.css";

export default function Dashboard() {
  return (
    <div className="dashboard-container">
       <DashboardHeader />
       <NavigationCards />
       <StatsOverview />
       <RecentAlerts />
       <SystemFlow />
    </div>
  );
}