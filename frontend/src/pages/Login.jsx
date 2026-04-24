import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import AuthLayout from "../components/AuthLayout.jsx";
import { useAuth } from "../context/AuthContext.jsx";

export default function Login() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const { login } = useAuth();
  const navigate = useNavigate();

  async function onSubmit(e) {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      await login(email, password);
      navigate("/", { replace: true });
    } catch (err) {
      const d = err?.response?.data?.detail;
      let msg =
        (typeof d === "string" && d) ||
        err?.message ||
        "Login failed. Is the API running?";
      if (Array.isArray(d)) {
        msg = d.map((x) => x?.msg || JSON.stringify(x)).join(" ");
      }
      setError(msg);
    } finally {
      setLoading(false);
    }
  }

  return (
    <AuthLayout title="Log in" subtitle="Welcome back">
      <form onSubmit={onSubmit} className="space-y-4">
        {error ? (
          <p
            className="rounded-lg bg-red-950/50 border border-red-900/60 text-red-200 text-sm px-3 py-2"
            role="alert"
          >
            {error}
          </p>
        ) : null}
        <div>
          <label htmlFor="login-email" className="block text-sm text-slate-400 mb-1">
            Email
          </label>
          <input
            id="login-email"
            name="email"
            type="email"
            autoComplete="email"
            required
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="w-full rounded-lg border border-slate-700 bg-slate-950/80 px-3 py-2 text-white placeholder-slate-600 focus:outline-none focus:ring-2 focus:ring-cyan-500/40 focus:border-cyan-500/50"
            placeholder="you@example.com"
          />
        </div>
        <div>
          <label
            htmlFor="login-password"
            className="block text-sm text-slate-400 mb-1"
          >
            Password
          </label>
          <input
            id="login-password"
            name="password"
            type="password"
            autoComplete="current-password"
            required
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="w-full rounded-lg border border-slate-700 bg-slate-950/80 px-3 py-2 text-white placeholder-slate-600 focus:outline-none focus:ring-2 focus:ring-cyan-500/40 focus:border-cyan-500/50"
            placeholder="••••••••"
          />
        </div>
        <button
          type="submit"
          disabled={loading}
          className="w-full rounded-lg bg-cyan-600 hover:bg-cyan-500 disabled:opacity-50 disabled:pointer-events-none text-white font-display font-semibold py-2.5 transition-colors"
        >
          {loading ? "Signing in…" : "Sign in"}
        </button>
      </form>
      <p className="mt-6 text-center text-slate-400 text-sm">
        No account?{" "}
        <Link to="/signup" className="text-cyan-400 hover:underline">
          Sign up
        </Link>
      </p>
    </AuthLayout>
  );
}
