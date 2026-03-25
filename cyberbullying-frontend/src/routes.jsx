import { BrowserRouter, Routes, Route } from "react-router-dom";
import Dashboard from "./pages/Dashboard";
import Moderation from "./pages/Moderation";
import History from "./pages/History";
import Analysis from "./pages/Analysis";

export default function AppRoutes() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/moderation" element={<Moderation />} />
        <Route path="/history" element={<History />} />
        <Route path="/analysis" element={<Analysis />} />
      </Routes>
    </BrowserRouter>
  );
}