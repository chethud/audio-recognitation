import { Link } from "react-router-dom";
import { useAuth } from "../context/AuthContext.jsx";

export default function SiteHeader() {
  const { user, logout } = useAuth();

  return (
    <header className="sticky top-0 z-20 border-b border-white/[0.06] bg-[#030712]/60 backdrop-blur-xl">
      <div className="mx-auto flex max-w-6xl items-center justify-between gap-4 px-4 py-4 sm:px-6">
        <Link to="/" className="group flex items-center gap-3 min-w-0">
          <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl border border-cyan-400/20 bg-cyan-500/10 text-cyan-300 shadow-[0_0_20px_rgba(34,211,238,0.15)] transition group-hover:border-cyan-400/40">
            <svg
              className="h-5 w-5"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={1.75}
              aria-hidden
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3"
              />
            </svg>
          </span>
          <div className="min-w-0">
            <p className="font-display text-base font-bold tracking-tight text-white">
              ALM-Lite
            </p>
            <p className="truncate text-xs text-slate-500">
              Audio language model
            </p>
          </div>
        </Link>

        <nav className="flex items-center gap-2 sm:gap-3">
          <a
            href="#about"
            className="hidden sm:inline-flex rounded-lg px-3 py-2 text-sm text-slate-400 transition hover:bg-white/5 hover:text-slate-200"
          >
            About
          </a>
          <a
            href="#analyze"
            className="hidden sm:inline-flex rounded-lg px-3 py-2 text-sm text-slate-400 transition hover:bg-white/5 hover:text-slate-200"
          >
            Analyze
          </a>
          <div className="glass-nav flex items-center gap-3 text-sm">
            {user ? (
              <>
                <span className="hidden max-w-[160px] truncate text-slate-400 sm:inline">
                  {user.email}
                </span>
                <button
                  type="button"
                  onClick={() => logout()}
                  className="text-cyan-300 transition hover:text-cyan-200"
                >
                  Log out
                </button>
              </>
            ) : (
              <>
                <Link
                  to="/login"
                  className="text-slate-300 transition hover:text-white"
                >
                  Log in
                </Link>
                <Link
                  to="/signup"
                  className="font-medium text-cyan-300 transition hover:text-cyan-100"
                >
                  Sign up
                </Link>
              </>
            )}
          </div>
        </nav>
      </div>
    </header>
  );
}
