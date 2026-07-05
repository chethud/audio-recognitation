const FEATURES = [
  {
    icon: "🎙️",
    title: "Speech recognition",
    model: "Whisper",
    desc: "Converts spoken words into a text transcript from your audio.",
  },
  {
    icon: "🔊",
    title: "Sound detection",
    model: "AST",
    desc: "Identifies environmental and background sounds in the scene.",
  },
  {
    icon: "😊",
    title: "Emotion analysis",
    model: "Wav2Vec2",
    desc: "Estimates the speaker's emotional tone from voice patterns.",
  },
  {
    icon: "✨",
    title: "AI reasoning",
    model: "Qwen2",
    desc: "Generates a natural-language answer to your question.",
  },
];

const STEPS = [
  "Upload an audio clip",
  "Models analyze in parallel",
  "Get transcript, sounds, emotion & answer",
];

const TECH = ["FastAPI", "React", "PyTorch", "SQLite", "Hugging Face"];

export default function ProjectInfo() {
  return (
    <aside className="space-y-5">
      <div className="glass-panel p-5 sm:p-6">
        <h2 className="font-display text-base font-semibold text-white mb-2">
          About this project
        </h2>
        <p className="text-sm text-slate-400 leading-relaxed">
          <span className="text-slate-300">ALM-Lite</span> is an audio language
          model that understands audio holistically — not just what was said, but
          what sounds are present, how the speaker feels, and what it all means
          together.
        </p>
        <p className="mt-3 text-sm text-slate-400 leading-relaxed">
          Upload any short clip, ask a question, and the pipeline returns a full
          analysis powered by pretrained AI models running locally on your
          machine.
        </p>
      </div>

      <div className="glass-panel p-5 sm:p-6">
        <h3 className="project-section-label mb-4">What it detects</h3>
        <ul className="space-y-3">
          {FEATURES.map((f) => (
            <li key={f.title} className="feature-card">
              <span className="feature-icon" aria-hidden>
                {f.icon}
              </span>
              <div className="min-w-0">
                <p className="text-sm font-medium text-slate-200">{f.title}</p>
                <p className="text-xs text-cyan-400/80 mb-0.5">{f.model}</p>
                <p className="text-xs text-slate-500 leading-relaxed">
                  {f.desc}
                </p>
              </div>
            </li>
          ))}
        </ul>
      </div>

      <div className="glass-panel p-5 sm:p-6">
        <h3 className="project-section-label mb-4">How it works</h3>
        <ol className="space-y-3">
          {STEPS.map((step, i) => (
            <li key={step} className="flex items-start gap-3 text-sm">
              <span className="step-badge">{i + 1}</span>
              <span className="text-slate-400 pt-0.5">{step}</span>
            </li>
          ))}
        </ol>
      </div>

      <div className="glass-panel-subtle px-5 py-4">
        <p className="project-section-label mb-3">Built with</p>
        <div className="flex flex-wrap gap-2">
          {TECH.map((t) => (
            <span key={t} className="tech-badge">
              {t}
            </span>
          ))}
        </div>
      </div>
    </aside>
  );
}
