import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import AuthLayout from "../components/AuthLayout.jsx";
import { useAuth } from "../context/AuthContext.jsx";

export default function Signup() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const { signup } = useAuth();
  const navigate = useNavigate();

  async function onSubmit(e) {
    e.preventDefault();
    setError(null);
    if (password.length < 8) {
      setError("Password must be at least 8 characters.");
      return;
    }
    if (password !== confirm) {
      setError("Passwords do not match.");
      return;
    }
    setLoading(true);
    try {
      await signup(email, password);
      navigate("/", { replace: true });
    } catch (err) {
      const d = err?.response?.data?.detail;
      let msg =
        (typeof d === "string" && d) ||
        err?.message ||
        "Sign up failed. Is the API running?";
      if (Array.isArray(d)) {
        msg = d.map((x) => x?.msg || JSON.stringify(x)).join(" ");
      }
      setError(msg);
    } finally {
      setLoading(false);
    }
  }

  return (
    <AuthLayout title="Create account" subtitle="Join ALM-Lite">
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
          <label
            htmlFor="signup-email"
            className="block text-sm text-slate-400 mb-1"
          >
            Email
          </label>
          <input
            id="signup-email"
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
            htmlFor="signup-password"
            className="block text-sm text-slate-400 mb-1"
          >
            Password
          </label>
          <input
            id="signup-password"
            name="password"
            type="password"
            autoComplete="new-password"
            required
            minLength={8}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="w-full rounded-lg border border-slate-700 bg-slate-950/80 px-3 py-2 text-white placeholder-slate-600 focus:outline-none focus:ring-2 focus:ring-cyan-500/40 focus:border-cyan-500/50"
            placeholder="At least 8 characters"
          />
        </div>
        <div>
          <label
            htmlFor="signup-confirm"
            className="block text-sm text-slate-400 mb-1"
          >
            Confirm password
          </label>
          <input
            id="signup-confirm"
            name="confirm"
            type="password"
            autoComplete="new-password"
            required
            value={confirm}
            onChange={(e) => setConfirm(e.target.value)}
            className="w-full rounded-lg border border-slate-700 bg-slate-950/80 px-3 py-2 text-white placeholder-slate-600 focus:outline-none focus:ring-2 focus:ring-cyan-500/40 focus:border-cyan-500/50"
            placeholder="Repeat password"
          />
        </div>
        <button
          type="submit"
          disabled={loading}
          className="w-full rounded-lg bg-cyan-600 hover:bg-cyan-500 disabled:opacity-50 disabled:pointer-events-none text-white font-display font-semibold py-2.5 transition-colors"
        >
          {loading ? "Creating account…" : "Sign up"}
        </button>
      </form>
      <p className="mt-6 text-center text-slate-400 text-sm">
        Already have an account?{" "}
        <Link to="/login" className="text-cyan-400 hover:underline">
          Log in
        </Link>
      </p>
    </AuthLayout>
  );
}
