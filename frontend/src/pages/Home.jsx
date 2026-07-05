import { useState } from "react";
import { analyzeAudio } from "../api/api.js";
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
      "Detecting languages…",
      "Scanning sound effects…",
      "Building answer…",
    ];
    let i = 0;
    setLoadingStage(stages[0]);
    const timer = setInterval(() => {
      i = (i + 1) % stages.length;
      setLoadingStage(stages[i]);
    }, 4000);
    try {
      const data = await analyzeAudio(file, question);
      setResult(data);
    } catch (e) {
      const d = e?.response?.data?.detail;
      let msg =
        d ||
        e?.message ||
        "Request failed. Is the API running on port 8001?";
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
      <div className="mx-auto max-w-6xl px-4 py-8 sm:py-10">
        <AppHeader user={user} onLogout={logout} />

        <section className="mb-8 sm:mb-10 text-center lg:text-left">
          <p className="inline-flex items-center gap-2 rounded-full border border-violet-400/25 bg-violet-500/15 px-3 py-1 text-xs font-medium text-violet-200/90 mb-4">
            <span className="h-1.5 w-1.5 rounded-full bg-fuchsia-400 animate-pulse" />
            Full audio understanding pipeline
          </p>
          <h1 className="font-display text-3xl sm:text-4xl lg:text-[2.75rem] font-bold text-gradient tracking-tight leading-tight max-w-2xl mx-auto lg:mx-0">
            Understand audio beyond words
          </h1>
          <p className="mt-3 text-slate-400 max-w-xl mx-auto lg:mx-0 leading-relaxed text-sm sm:text-base">
            Upload a clip and get speech transcription, sound events, speaker
            emotion, and an AI-generated answer — with automatic multi-language
            and multi-sound detection.
          </p>
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
                question={question}
                onQuestionChange={setQuestion}
                onSubmit={handleSubmit}
                loading={loading}
                loadingStage={loadingStage}
                disabled={loading}
              />
            </div>

            <div className="glass-panel p-6 sm:p-7 min-h-[240px]">
              <h2 className="font-display text-lg font-semibold text-white mb-6 flex items-center gap-2.5">
                <span className="step-badge step-badge-lg step-badge-accent">2</span>
                Analysis results
              </h2>
              <ResultDisplay result={result} error={error} loading={loading} />
            </div>
          </div>
        </div>

        <AppFooter />
      </div>
    </GlassBackground>
  );
}
