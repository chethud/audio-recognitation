import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { AuthProvider, useAuth } from "./context/AuthContext.jsx";
import GlassBackground from "./components/GlassBackground.jsx";
import Home from "./pages/Home.jsx";
import Login from "./pages/Login.jsx";
import Signup from "./pages/Signup.jsx";

function AppRoutes() {
  const { ready } = useAuth();
  if (!ready) {
    return (
      <GlassBackground className="flex items-center justify-center">
        <div className="glass-panel px-8 py-6 flex items-center gap-3 text-slate-300 text-sm">
          <span className="h-5 w-5 rounded-full border-2 border-violet-400/30 border-t-violet-300 animate-spin" />
          Loading…
        </div>
      </GlassBackground>
    );
  }
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/login" element={<Login />} />
      <Route path="/signup" element={<Signup />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <AppRoutes />
      </AuthProvider>
    </BrowserRouter>
  );
}
