import { Link } from "react-router-dom";
import GlassBackground from "./GlassBackground.jsx";

export default function AuthLayout({ title, subtitle, children }) {
  return (
    <GlassBackground className="flex flex-col items-center justify-center px-4 py-12">
      <div className="w-full max-w-md">
        <p className="font-display text-center text-sm uppercase tracking-[0.2em] text-violet-300/80 mb-3">
          <Link to="/" className="hover:text-violet-200 transition-colors inline-flex items-center gap-2">
            <span className="inline-flex h-7 w-7 items-center justify-center rounded-lg border border-violet-400/25 bg-violet-500/15 text-xs font-bold">
              ALM
            </span>
            ALM-LITE
          </Link>
        </p>
        <h1 className="font-display text-2xl sm:text-3xl font-bold text-gradient text-center tracking-tight">
          {title}
        </h1>
        {subtitle ? (
          <p className="mt-2 text-slate-400/90 text-center text-sm">{subtitle}</p>
        ) : null}
        <div className="mt-8 glass-panel p-6 sm:p-8">{children}</div>
        <p className="mt-6 text-center">
          <Link
            to="/"
            className="text-sm text-slate-500 hover:text-slate-300 transition-colors"
          >
            ← Back to app
          </Link>
        </p>
      </div>
    </GlassBackground>
  );
}
