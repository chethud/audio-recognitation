import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { AuthProvider, useAuth } from "./context/AuthContext.jsx";
import Home from "./pages/Home.jsx";
import Login from "./pages/Login.jsx";
import Signup from "./pages/Signup.jsx";

function AppRoutes() {
  const { ready } = useAuth();
  if (!ready) {
    return (
      <div className="min-h-screen bg-slate-950 flex items-center justify-center text-slate-400 text-sm">
        Loading…
      </div>
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
