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
  kn: "Kannada",
  ml: "Malayalam",
  gu: "Gujarati",
  pa: "Punjabi",
};

function languageLabel(code) {
  if (!code || code === "en") return "English";
  return LANGUAGE_LABELS[code] || code.toUpperCase();
}

function stripLanguageTag(text) {
  return (text || "").replace(/^\[[^\]]+\]\s*/, "").trim();
}

function formatTimestamp(sec) {
  if (sec == null || Number.isNaN(Number(sec))) return "--:--.---";
  const s = Math.max(0, Number(sec));
  const minutes = Math.floor(s / 60);
  const rem = s - minutes * 60;
  return `${String(minutes).padStart(2, "0")}:${rem.toFixed(3).padStart(6, "0")}`;
}

export default function ResultDisplay({
  result,
  error,
  loading,
  loadingStage = "",
  loadingProgress = 0,
  timeRemaining = null,
}) {
  if (error) {
    return <div className="glass-error">{error}</div>;
  }

  if (loading) {
    const pct = Math.round(Math.min(100, Math.max(0, loadingProgress)));

    const steps = [
      { label: "Upload",      min: 0,  max: 18 },
      { label: "Transcribe",  min: 18, max: 62 },
      { label: "Detect",      min: 62, max: 82 },
      { label: "Answer",      min: 82, max: 100 },
    ];
    const activeStep = steps.findIndex((s, i) => {
      const next = steps[i + 1];
      return pct >= s.min && (!next || pct < next.min);
    });

    return (
      <div className="glass-panel-subtle px-5 py-8 flex flex-col items-center gap-5">
        {/* Spinning ring + percentage circle */}
        <div className="relative flex items-center justify-center">
          <svg className="w-20 h-20 -rotate-90" viewBox="0 0 64 64">
            <circle cx="32" cy="32" r="26" fill="none" stroke="rgba(255,255,255,0.07)" strokeWidth="5" />
            <circle
              cx="32" cy="32" r="26"
              fill="none"
              stroke="url(#prog-grad)"
              strokeWidth="5"
              strokeLinecap="round"
              strokeDasharray={`${2 * Math.PI * 26}`}
              strokeDashoffset={`${2 * Math.PI * 26 * (1 - pct / 100)}`}
              style={{ transition: "stroke-dashoffset 0.35s ease-out" }}
            />
            <defs>
              <linearGradient id="prog-grad" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#a78bfa" />
                <stop offset="100%" stopColor="#f0abfc" />
              </linearGradient>
            </defs>
          </svg>
          <span className="absolute text-lg font-bold tabular-nums text-white">{pct}%</span>
        </div>

        {/* Stage label */}
        <div className="text-center">
          <p className="text-sm font-semibold text-slate-200">
            {loadingStage || "Running full analysis…"}
          </p>
          <p className="text-xs text-violet-300 font-medium mt-1">
            {timeRemaining !== null
              ? timeRemaining <= 2
                ? "Finishing up analysis details…"
                : `Estimated remaining time: ~${timeRemaining} seconds`
              : "Preparing analysis parameters…"}
          </p>
        </div>

        {/* Step pills */}
        <div className="flex items-center gap-1 flex-wrap justify-center">
          {steps.map((step, i) => {
            const done = pct >= step.max;
            const active = i === activeStep;
            return (
              <span
                key={step.label}
                className={`px-2.5 py-0.5 rounded-full text-xs font-medium border transition-all ${
                  done
                    ? "bg-violet-500/30 border-violet-400/50 text-violet-200"
                    : active
                    ? "bg-fuchsia-500/20 border-fuchsia-400/50 text-fuchsia-200 animate-pulse"
                    : "bg-white/5 border-white/10 text-slate-500"
                }`}
              >
                {done ? "✓ " : active ? "⟳ " : ""}{step.label}
              </span>
            );
          })}
        </div>

        {/* Linear progress bar */}
        <div className="w-full rounded-full bg-white/8 h-1.5 overflow-hidden">
          <div
            className="h-full rounded-full bg-gradient-to-r from-violet-400 to-fuchsia-400"
            style={{
              width: `${pct}%`,
              transition: "width 0.35s ease-out",
              boxShadow: "0 0 10px rgba(167,139,250,0.5)",
            }}
          />
        </div>
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
          to see transcript, sounds, emotion, and summary here.
        </p>
      </div>
    );
  }

  const {
    transcript,
    transcript_original,
    sounds,
    sound_details,
    emotion,
    speaker_emotions,
    answer,
    summary,
    language,
    language_name,
    languages,
    language_names,
    speaker_turns,
    num_speakers,
    detected_speakers,
    formatted_transcript,
  } = result;

  const langTags =
    (language_names && language_names.length > 0
      ? language_names
      : languages?.map(languageLabel)) || [];
  const multiLang = langTags.length > 1;
  const englishOnly =
    !multiLang && (!language || language === "en") && langTags.length <= 1;
  const detectedLabel = multiLang
    ? langTags.join(", ")
    : language_name || languageLabel(language);

  const soundItems =
    sound_details && sound_details.length > 0
      ? sound_details
      : (sounds || []).map((label) => ({ label, score: null }));

  const turns = speaker_turns || [];
  const speakersFromTurns = [];
  for (const t of turns) {
    const sp = (t.speaker || "").trim();
    if (sp && !speakersFromTurns.includes(sp)) speakersFromTurns.push(sp);
  }
  const speakers =
    detected_speakers && detected_speakers.length > 0
      ? detected_speakers
      : speakersFromTurns;

  const hasTurns = turns.length > 0;
  const emotionMap =
    speaker_emotions && typeof speaker_emotions === "object"
      ? speaker_emotions
      : {};
  const emotionLabel = (emotion || "neutral").trim() || "neutral";
  const summaryText = (summary || answer || "").trim() || "—";

  const originalText = stripLanguageTag(transcript_original || "");
  const englishText = (transcript || "").trim();
  const nonEnglish =
    !englishOnly && language && language !== "en" && language !== "multi";
  const showOriginalFirst = nonEnglish && originalText.length > 0 && !hasTurns;
  const mainTranscript = showOriginalFirst
    ? originalText
    : englishText || originalText || formatted_transcript || "—";

  const timedBlocks = (
    <div className="space-y-4">
      {turns.map((t, i) => (
        <div
          key={`${t.speaker}-${t.start_sec}-${i}`}
          className="rounded-lg border border-violet-400/15 bg-violet-950/20 px-3 py-2.5"
        >
          <p className="text-xs font-mono text-violet-300/80 mb-1">
            [{formatTimestamp(t.start_sec)} - {formatTimestamp(t.end_sec)}]
          </p>
          <p className="text-xs font-semibold text-violet-200/90 mb-1.5">
            {t.speaker || "Speaker 1"}
          </p>
          <p className="text-slate-100 text-sm leading-relaxed whitespace-pre-wrap break-words">
            {t.text || "—"}
          </p>
        </div>
      ))}
    </div>
  );

  const sections = [
    {
      key: "speakers",
      label: "Detected Speakers",
      accent: "violet",
      content: (
        <ul className="list-disc list-inside space-y-1 text-slate-100 text-sm">
          {(speakers.length > 0 ? speakers : ["Speaker 1"]).map((sp) => (
            <li key={sp}>{sp}</li>
          ))}
        </ul>
      ),
    },
    {
      key: "transcript",
      label: hasTurns
        ? `Transcript (${num_speakers || speakers.length || turns.length} speaker${
            (num_speakers || speakers.length || 1) === 1 ? "" : "s"
          })`
        : englishOnly
          ? "Transcript"
          : `Transcript (${detectedLabel})`,
      accent: "violet",
      highlight: hasTurns,
      full: true,
      content: (
        <>
          {!englishOnly && !hasTurns && langTags.length > 0 ? (
            <div className="flex flex-wrap gap-1.5 mb-2">
              {langTags.map((name) => (
                <span
                  key={name}
                  className="rounded-full border border-violet-400/25 bg-violet-500/10 px-2 py-0.5 text-xs text-violet-200"
                >
                  {name}
                </span>
              ))}
            </div>
          ) : null}
          {hasTurns ? (
            timedBlocks
          ) : (
            <p className="text-slate-100 whitespace-pre-wrap leading-relaxed text-sm sm:text-base break-words">
              {mainTranscript}
            </p>
          )}
              {hasTurns && language && language !== "en" && language !== "multi" ? (
            <p className="mt-2 text-xs text-violet-300/70">
              Language: {detectedLabel}
            </p>
          ) : null}
        </>
      ),
    },
    {
      key: "sounds",
      label: `Detected Sounds${soundItems.length ? ` (${soundItems.length})` : ""}`,
      accent: "violet",
      content: (
        <div className="flex flex-wrap gap-2">
          {soundItems.length === 0 ? (
            <span className="text-slate-500 text-sm">No sounds detected</span>
          ) : (
            soundItems.map((s) => (
              <span key={s.label} className="glass-tag">
                {s.label}
                {s.score != null ? (
                  <span className="ml-1 text-violet-200/60">
                    {Math.round(s.score * 100)}%
                  </span>
                ) : null}
              </span>
            ))
          )}
        </div>
      ),
    },
    {
      key: "emotion",
      label: "Speaker Emotion",
      accent: "violet",
      content: (
        <div className="space-y-3 text-sm sm:text-base">
          {(speakers.length > 0 ? speakers : ["Speaker 1"]).map((sp) => {
            const label =
              (emotionMap[sp] || emotionLabel || "neutral").trim() || "neutral";
            return (
              <div key={sp}>
                <p className="text-violet-200/90 font-semibold mb-0.5">{sp}:</p>
                <p className="text-slate-100 capitalize">{label}</p>
              </div>
            );
          })}
        </div>
      ),
    },
    {
      key: "summary",
      label: "Conversation Summary",
      accent: "fuchsia",
      highlight: true,
      full: true,
      content: (
        <div className="text-slate-100 whitespace-pre-wrap leading-relaxed text-sm sm:text-base border-l-2 border-fuchsia-400/45 pl-4 space-y-2">
          {summaryText
            .split("\n")
            .filter(Boolean)
            .map((line) => (
              <p key={line}>{line}</p>
            ))}
        </div>
      ),
    },
  ];

  return (
    <div className="grid gap-3 sm:grid-cols-2 text-left">
      {sections.map((s) => (
        <section
          key={s.key}
          className={`glass-panel-subtle p-4 ${
            s.full || s.highlight || s.key === "summary"
              ? "sm:col-span-2 border-fuchsia-400/20"
              : ""
          } ${s.key === "transcript" && s.highlight ? "border-violet-400/20" : ""}`}
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
