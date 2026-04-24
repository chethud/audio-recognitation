import { Link } from "react-router-dom";

export default function AuthLayout({ title, subtitle, children }) {
  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-950 via-slate-900 to-slate-950 flex flex-col items-center justify-center px-4 py-12">
      <div className="w-full max-w-md">
        <p className="font-display text-center text-sm uppercase tracking-[0.2em] text-cyan-400/80 mb-3">
          <Link to="/" className="hover:text-cyan-300">
            ALM-Lite
          </Link>
        </p>
        <h1 className="font-display text-2xl sm:text-3xl font-bold text-white text-center tracking-tight">
          {title}
        </h1>
        {subtitle ? (
          <p className="mt-2 text-slate-400 text-center text-sm">{subtitle}</p>
        ) : null}
        <div className="mt-8 rounded-2xl border border-slate-800 bg-slate-900/50 p-6 shadow-xl shadow-black/40 backdrop-blur">
          {children}
        </div>
        <p className="mt-6 text-center">
          <Link
            to="/"
            className="text-sm text-slate-500 hover:text-slate-300 transition-colors"
          >
            ← Back to app
          </Link>
        </p>
      </div>
    </div>
  );
}
