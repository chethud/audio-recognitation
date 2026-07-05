import { Link } from "react-router-dom";

export default function AppHeader({ user, onLogout }) {
  return (
    <header className="flex flex-wrap items-center justify-between gap-4 mb-8 sm:mb-10">
      <Link to="/" className="flex items-center gap-3 group">
        <div className="flex h-10 w-10 items-center justify-center rounded-xl border border-cyan-400/20 bg-cyan-500/10 backdrop-blur-sm shadow-[0_0_20px_rgba(34,211,238,0.15)]">
          <svg
            className="h-5 w-5 text-cyan-300"
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
        </div>
        <div>
          <p className="font-display text-lg font-bold text-white tracking-tight group-hover:text-cyan-100 transition-colors">
            ALM-Lite
          </p>
          <p className="text-xs text-slate-500 hidden sm:block">
            Audio language model
          </p>
        </div>
      </Link>

      <div className="glass-nav flex flex-wrap items-center gap-x-4 gap-y-1 text-sm">
        {user ? (
          <>
            <span className="text-slate-400 truncate max-w-[180px] sm:max-w-xs">
              {user.email}
            </span>
            <button
              type="button"
              onClick={onLogout}
              className="text-cyan-300 hover:text-cyan-200 transition-colors font-medium"
            >
              Log out
            </button>
          </>
        ) : (
          <>
            <Link
              to="/login"
              className="text-slate-300 hover:text-white transition-colors"
            >
              Log in
            </Link>
            <Link
              to="/signup"
              className="text-cyan-300 hover:text-cyan-100 font-medium transition-colors"
            >
              Sign up
            </Link>
          </>
        )}
      </div>
    </header>
  );
}
