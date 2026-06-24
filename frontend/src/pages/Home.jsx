import { useState } from "react";
import { Link } from "react-router-dom";
import { analyzeAudio } from "../api/api.js";
import UploadAudio from "../components/UploadAudio.jsx";
import ResultDisplay from "../components/ResultDisplay.jsx";
import GlassBackground from "../components/GlassBackground.jsx";
import { useAuth } from "../context/AuthContext.jsx";

export default function Home() {
  const { user, logout } = useAuth();
  const [file, setFile] = useState(null);
  const [question, setQuestion] = useState(
    "What can be inferred from the audio?"
  );
  const [loading, setLoading] = useState(false);
  const [loadingStage, setLoadingStage] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  async function handleSubmit() {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);
    const stages = [
      "Transcribing speech…",
      "Detecting sounds & emotion…",
      "Generating AI answer…",
    ];
    let i = 0;
    setLoadingStage(stages[0]);
    const timer = setInterval(() => {
      i = (i + 1) % stages.length;
      setLoadingStage(stages[i]);
    }, 8000);
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
      clearInterval(timer);
      setLoadingStage("");
      setLoading(false);
    }
  }

  return (
    <GlassBackground>
      <div className="mx-auto max-w-3xl px-4 py-10 sm:py-14">
        <nav className="flex flex-wrap items-center justify-end gap-3 text-sm mb-10">
          <div className="glass-nav flex flex-wrap items-center gap-x-4 gap-y-1">
            {user ? (
              <>
                <span className="text-slate-400 truncate max-w-[200px] sm:max-w-xs">
                  {user.email}
                </span>
                <button
                  type="button"
                  onClick={() => logout()}
                  className="text-cyan-300 hover:text-cyan-200 transition-colors"
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
        </nav>

        <header className="mb-10 sm:mb-12 text-center">
          <p className="font-display text-sm uppercase tracking-[0.25em] text-cyan-300/70 mb-3">
            ALM-Lite
          </p>
          <h1 className="font-display text-3xl sm:text-4xl font-bold text-gradient tracking-tight">
            Audio language &amp; scene understanding
          </h1>
          <p className="mt-4 text-slate-400/90 max-w-xl mx-auto leading-relaxed">
            Speech recognition, environmental sounds, emotion, and AI reasoning —
            powered by{" "}
            <code className="rounded-md border border-white/10 bg-white/5 px-1.5 py-0.5 text-cyan-200/90 text-sm backdrop-blur-sm">
              POST /analyze
            </code>
          </p>
        </header>

        <div className="grid gap-6 sm:gap-8">
          <div className="glass-panel p-6 sm:p-8">
            <h2 className="font-display text-lg font-semibold text-white/95 mb-6 flex items-center gap-2">
              <span className="h-8 w-8 rounded-lg border border-white/10 bg-cyan-500/10 backdrop-blur-sm flex items-center justify-center text-cyan-300 text-sm">
                1
              </span>
              Upload &amp; analyze
            </h2>
            <UploadAudio
              file={file}
              onFileChange={setFile}
              question={question}
              onQuestionChange={setQuestion}
              onSubmit={handleSubmit}
              loading={loading}
              loadingStage={loadingStage}
              disabled={loading}
            />
          </div>

          <div className="glass-panel p-6 sm:p-8 min-h-[220px]">
            <h2 className="font-display text-lg font-semibold text-white/95 mb-6 flex items-center gap-2">
              <span className="h-8 w-8 rounded-lg border border-white/10 bg-teal-500/10 backdrop-blur-sm flex items-center justify-center text-teal-300 text-sm">
                2
              </span>
              Results
            </h2>
            <ResultDisplay result={result} error={error} />
          </div>
        </div>
      </div>
    </GlassBackground>
  );
}
