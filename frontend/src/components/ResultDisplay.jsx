const LANGUAGE_LABELS = {
  en: "English",
  hi: "Hindi",
  es: "Spanish",
  fr: "French",
  de: "German",
  it: "Italian",
  pt: "Portuguese",
  ru: "Russian",
  ja: "Japanese",
  ko: "Korean",
  zh: "Chinese",
  ar: "Arabic",
  ta: "Tamil",
  te: "Telugu",
  bn: "Bengali",
  mr: "Marathi",
  ur: "Urdu",
};

function languageLabel(code) {
  if (!code || code === "en") return "English";
  return LANGUAGE_LABELS[code] || code.toUpperCase();
}

export default function ResultDisplay({ result, error, loading }) {
  if (error) {
    return <div className="glass-error">{error}</div>;
  }

  if (loading) {
    return (
      <div className="glass-panel-subtle px-4 py-10 text-center">
        <span className="inline-block h-8 w-8 rounded-full border-2 border-violet-400/30 border-t-violet-300 animate-spin mb-3" />
        <p className="text-slate-400 text-sm">Running full analysis…</p>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="glass-panel-subtle px-4 py-10 text-center">
        <div className="mx-auto mb-3 flex h-12 w-12 items-center justify-center rounded-xl border border-white/10 bg-white/[0.03]">
          <svg
            className="h-6 w-6 text-slate-600"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={1.5}
            aria-hidden
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z"
            />
          </svg>
        </div>
        <p className="text-slate-500 text-sm">
          Upload audio and click <span className="text-slate-400">Analyze</span>{" "}
          to see transcript, sounds, emotion, and AI answer here.
        </p>
      </div>
    );
  }

  const { transcript, sounds, emotion, answer, language } = result;
  const answerLang = language && language !== "en" ? languageLabel(language) : null;

  const sections = [
    {
      key: "transcript",
      label: "Transcript (English)",
      accent: "violet",
      content: (
        <p className="text-slate-100 whitespace-pre-wrap leading-relaxed text-sm sm:text-base">
          {transcript || "—"}
        </p>
      ),
    },
    {
      key: "sounds",
      label: "Detected sounds",
      accent: "violet",
      content: (
        <div className="flex flex-wrap gap-2">
          {(sounds || []).length === 0 ? (
            <span className="text-slate-500 text-sm">No sounds detected</span>
          ) : (
            sounds.map((s) => (
              <span key={s} className="glass-tag">
                {s}
              </span>
            ))
          )}
        </div>
      ),
    },
    {
      key: "emotion",
      label: "Speaker emotion",
      accent: "violet",
      content: (
        <p className="text-slate-100 capitalize text-sm sm:text-base">
          {emotion || "—"}
        </p>
      ),
    },
    {
      key: "answer",
      label: answerLang ? `AI answer (${answerLang})` : "AI answer",
      accent: "fuchsia",
      highlight: true,
      content: (
        <p className="text-slate-100 whitespace-pre-wrap leading-relaxed text-sm sm:text-base border-l-2 border-fuchsia-400/45 pl-4">
          {answer || "—"}
        </p>
      ),
    },
  ];

  return (
    <div className="grid gap-3 sm:grid-cols-2 text-left">
      {sections.map((s) => (
        <section
          key={s.key}
          className={`glass-panel-subtle p-4 ${
            s.highlight ? "sm:col-span-2 border-fuchsia-400/20" : ""
          }`}
        >
          <h3
            className={`section-label mb-2 ${
              s.accent === "fuchsia" ? "text-fuchsia-300/75" : ""
            }`}
          >
            {s.label}
          </h3>
          {s.content}
        </section>
      ))}
    </div>
  );
}
