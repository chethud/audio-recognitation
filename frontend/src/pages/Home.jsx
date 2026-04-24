import { useState } from "react";
import { Link } from "react-router-dom";
import { analyzeAudio } from "../api/api.js";
import UploadAudio from "../components/UploadAudio.jsx";
import ResultDisplay from "../components/ResultDisplay.jsx";
import { useAuth } from "../context/AuthContext.jsx";

export default function Home() {
  const { user, logout } = useAuth();
  const [file, setFile] = useState(null);
  const [question, setQuestion] = useState(
    "What can be inferred from the audio?"
  );
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  async function handleSubmit() {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await analyzeAudio(file, question);
      setResult(data);
    } catch (e) {
      const d = e?.response?.data?.detail;
      let msg =
        d ||
        e?.message ||
        "Request failed. Is the API running on port 8000?";
      if (Array.isArray(d)) {
        msg = d.map((x) => x?.msg || JSON.stringify(x)).join(" ");
      } else if (typeof d === "object" && d !== null) {
        msg = JSON.stringify(d);
      }
      setError(typeof msg === "string" ? msg : String(msg));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-950 via-slate-900 to-slate-950">
      <div className="mx-auto max-w-3xl px-4 py-14">
        <nav className="flex flex-wrap items-center justify-end gap-x-4 gap-y-2 text-sm mb-8">
          {user ? (
            <>
              <span className="text-slate-400 truncate max-w-[200px] sm:max-w-xs">
                {user.email}
              </span>
              <button
                type="button"
                onClick={() => logout()}
                className="text-cyan-400 hover:text-cyan-300 hover:underline"
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
                className="text-cyan-400 hover:text-cyan-300 hover:underline font-medium"
              >
                Sign up
              </Link>
            </>
          )}
        </nav>
        <header className="mb-12 text-center">
          <p className="font-display text-sm uppercase tracking-[0.2em] text-cyan-400/80 mb-2">
            ALM-Lite
          </p>
          <h1 className="font-display text-3xl sm:text-4xl font-bold text-white tracking-tight">
            Audio language &amp; scene understanding
          </h1>
          <p className="mt-3 text-slate-400 max-w-xl mx-auto">
            Speech (Whisper), environmental sounds (SED), emotion, and LLM
            reasoning — via <code className="text-cyan-300/90">POST /analyze</code>
            .
          </p>
        </header>

        <div className="grid gap-8 md:grid-cols-1">
          <div className="rounded-2xl border border-slate-800 bg-slate-900/50 p-6 shadow-xl shadow-black/40 backdrop-blur">
            <UploadAudio
              file={file}
              onFileChange={setFile}
              question={question}
              onQuestionChange={setQuestion}
              onSubmit={handleSubmit}
              loading={loading}
              disabled={loading}
            />
          </div>

          <div className="rounded-2xl border border-slate-800 bg-slate-900/30 p-6 min-h-[200px]">
            <h2 className="font-display text-lg font-semibold text-white mb-4">
              Results
            </h2>
            <ResultDisplay result={result} error={error} />
          </div>
        </div>
      </div>
    </div>
  );
}
