import { useEffect, useState } from "react";
import { analyzeAudio, health } from "../api/api.js";
import UploadAudio from "../components/UploadAudio.jsx";
import ResultDisplay from "../components/ResultDisplay.jsx";
import GlassBackground from "../components/GlassBackground.jsx";
import AppHeader from "../components/AppHeader.jsx";
import ProjectInfo from "../components/ProjectInfo.jsx";
import AppFooter from "../components/AppFooter.jsx";
import { useAuth } from "../context/AuthContext.jsx";

export default function Home() {
  const { user, logout } = useAuth();
  const [file, setFile] = useState(null);
  const [language, setLanguage] = useState("en");
  const [question, setQuestion] = useState(
    "What can be inferred from the audio?"
  );
  const [loading, setLoading] = useState(false);
  const [loadingStage, setLoadingStage] = useState("");
  const [loadingProgress, setLoadingProgress] = useState(0);
  const [timeRemaining, setTimeRemaining] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [apiStatus, setApiStatus] = useState("checking");

  // Stage definitions: label + the % range this stage occupies
  const STAGES = [
    { label: "Uploading audio…",                              min: 0,  max: 18 },
    { label: "Transcribing speech (long files take a few minutes)…", min: 18, max: 62 },
    { label: "Detecting sounds and emotion…",                 min: 62, max: 82 },
    { label: "Building answer…",                              min: 82, max: 101 },
  ];

  useEffect(() => {
    if (loading) return undefined;
    let cancelled = false;
    async function poll() {
      try {
        const data = await health();
        if (cancelled) return;
        setApiStatus(data?.model_ready ? "ready" : "loading");
        if (!data?.model_ready) {
          setTimeout(poll, 5000);
        }
      } catch {
        if (!cancelled) {
          setApiStatus("waking");
          setTimeout(poll, 5000);
        }
      }
    }
    poll();
    return () => {
      cancelled = true;
    };
  }, [loading]);

  async function handleSubmit() {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);
    setLoadingProgress(0);
    setTimeRemaining(null);

    // Heuristic: estimate baseline of 16s + 8s per MB of audio file
    const fileSizeMB = file.size / (1024 * 1024);
    const totalEst = Math.max(16, Math.min(180, Math.round(16 + fileSizeMB * 8)));

    let elapsedMs = 0;
    const intervalMs = 250;

    setLoadingStage(STAGES[0].label);

    // Single unified interval for smooth progress and remaining time
    const timer = setInterval(() => {
      elapsedMs += intervalMs;
      const elapsedSec = elapsedMs / 1000;

      // Use a smooth asymptotic curve so progress never gets stuck or goes backwards,
      // creeping slowly towards 96% if the server is taking longer than expected.
      setLoadingProgress((prev) => {
        if (prev >= 96) return 96;
        const ratio = elapsedSec / totalEst;
        const target = 96 * (1 - Math.exp(-1.8 * ratio));
        const nextVal = Math.max(prev + 0.05, target);

        // Update stage label based on current progress percentage
        const currentStage = STAGES.find(s => nextVal >= s.min && nextVal <= s.max) || STAGES[STAGES.length - 1];
        setLoadingStage(currentStage.label);

        return nextVal;
      });

      // Update remaining time estimate
      setTimeRemaining(() => {
        return Math.max(1, Math.round(totalEst - elapsedSec));
      });
    }, intervalMs);

    try {
      const data = await analyzeAudio(file, question, {
        language,
        skipWarmup: apiStatus === "ready",
        onStatus: (s) => {
          if (s === "ready") setApiStatus("ready");
          else if (s === "loading") setLoadingStage("Waiting for AI models…");
          else if (s === "waking") setLoadingStage("Connecting to API…");
          else if (s === "analyzing") setLoadingStage("Analyzing audio — please wait…");
        },
      });
      setApiStatus("ready");
      setLoadingProgress(100);
      setTimeRemaining(0);
      setResult(data);
    } catch (e) {
      const d = e?.response?.data?.detail;
      let msg =
        d ||
        e?.message ||
        "Request failed. Is the API running? Try: python run.py --port 8002";
      if (Array.isArray(d)) {
        msg = d.map((x) => x?.msg || JSON.stringify(x)).join(" ");
      } else if (typeof d === "object" && d !== null) {
        msg = JSON.stringify(d);
      }
      setError(typeof msg === "string" ? msg : String(msg));
    } finally {
      clearInterval(timer);
      setLoadingStage("");
      setLoadingProgress(0);
      setTimeRemaining(null);
      setLoading(false);
    }
  }

  return (
    <GlassBackground>
      <div className="mx-auto max-w-6xl px-4 py-8 sm:py-10">
        <AppHeader user={user} onLogout={logout} />

        <section className="mb-8 sm:mb-10 text-center lg:text-left">
          <p className="inline-flex items-center gap-2 rounded-full border border-violet-400/25 bg-violet-500/15 px-3 py-1 text-xs font-medium text-violet-200/90 mb-4">
            <span className="h-1.5 w-1.5 rounded-full bg-fuchsia-400 animate-pulse" />
            Full audio understanding pipeline
          </p>
          <h1 className="font-display text-3xl sm:text-4xl lg:text-[2.75rem] font-bold text-gradient tracking-tight leading-tight max-w-2xl mx-auto lg:mx-0">
            Understand Audio Beyond Words
          </h1>
          <p className="mt-3 text-slate-400 max-w-xl mx-auto lg:mx-0 leading-relaxed text-sm sm:text-base">
            Upload a clip and get speech transcription, sound events, speaker
            emotion, and an AI-generated answer — with automatic multi-language
            and multi-sound detection.
          </p>
          {apiStatus === "waking" ? (
            <p className="mt-4 text-sm text-amber-200/90 rounded-lg border border-amber-400/25 bg-amber-500/10 px-3 py-2 max-w-xl mx-auto lg:mx-0">
              Cannot reach the API. Start the backend:{" "}
              <code className="text-amber-100">py -3.14 run.py --host 127.0.0.1 --port 8002</code>
              , then refresh this page.
            </p>
          ) : apiStatus === "checking" || apiStatus === "loading" ? (
            <p className="mt-4 inline-flex items-center gap-2 text-sm text-slate-400 rounded-lg border border-white/10 bg-white/5 px-3 py-2 max-w-xl mx-auto lg:mx-0">
              <span className="h-3.5 w-3.5 rounded-full border-2 border-violet-400/40 border-t-violet-300 animate-spin" />
              {apiStatus === "checking"
                ? "Connecting…"
                : "Getting AI models ready automatically…"}
            </p>
          ) : null}
        </section>

        <div className="grid gap-6 lg:grid-cols-12 lg:gap-8 items-start">
          <div className="lg:col-span-4 order-2 lg:order-1">
            <ProjectInfo />
          </div>

          <div className="lg:col-span-8 order-1 lg:order-2 space-y-6">
            <div className="glass-panel p-6 sm:p-7">
              <div className="flex items-center justify-between gap-3 mb-6">
                <h2 className="font-display text-lg font-semibold text-white flex items-center gap-2.5">
                  <span className="step-badge step-badge-lg">1</span>
                  Upload &amp; analyze
                </h2>
                <span className="text-xs text-slate-500 hidden sm:inline">
                  WAV · MP3 · FLAC · M4A
                </span>
              </div>
              <UploadAudio
                file={file}
                onFileChange={setFile}
                language={language}
                onLanguageChange={setLanguage}
                question={question}
                onQuestionChange={setQuestion}
                onSubmit={handleSubmit}
                loading={loading}
                loadingStage={loadingStage}
                loadingProgress={loadingProgress}
                timeRemaining={timeRemaining}
                disabled={loading}
              />
            </div>

            <div className="glass-panel p-6 sm:p-7 min-h-[240px]">
              <h2 className="font-display text-lg font-semibold text-white mb-6 flex items-center gap-2.5">
                <span className="step-badge step-badge-lg step-badge-accent">2</span>
                Analysis results
              </h2>
              <ResultDisplay
                result={result}
                error={error}
                loading={loading}
                loadingStage={loadingStage}
                loadingProgress={loadingProgress}
                timeRemaining={timeRemaining}
              />
            </div>
          </div>
        </div>

        <AppFooter />
      </div>
    </GlassBackground>
  );
}
