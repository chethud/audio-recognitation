import { useEffect, useState } from "react";

const LANGUAGE_OPTIONS = [
  { value: "en", label: "English" },
  { value: "kn", label: "Kannada" },
];

export default function UploadAudio({
  file,
  onFileChange,
  question,
  onQuestionChange,
  onSubmit,
  loading,
  loadingStage = "",
  loadingProgress = 0,
  timeRemaining = null,
  disabled,
  language = "en",
  onLanguageChange,
}) {
  const [previewUrl, setPreviewUrl] = useState(null);
  const [dragOver, setDragOver] = useState(false);

  useEffect(() => {
    if (!file) {
      setPreviewUrl(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  function handleDrop(e) {
    e.preventDefault();
    setDragOver(false);
    const dropped = e.dataTransfer.files?.[0];
    if (dropped) onFileChange(dropped);
  }

  const pct = Math.round(Math.min(100, Math.max(0, loadingProgress)));

  return (
    <div className="space-y-5">
      <div>
        <label className="block text-sm font-medium text-slate-300/90 mb-2">
          Audio file
        </label>
        <div
          className={`upload-dropzone ${dragOver ? "upload-dropzone-active" : ""}`}
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
        >
          <input
            type="file"
            accept="audio/*,.mp4,.webm,.mkv,.avi,.mov"
            onChange={(e) => onFileChange(e.target.files?.[0] || null)}
            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
            aria-label="Choose audio file"
          />
          <div className="pointer-events-none text-center py-6 px-4">
            <div className="mx-auto mb-2 flex h-10 w-10 items-center justify-center rounded-lg border border-white/10 bg-white/5">
              <svg
                className="h-5 w-5 text-violet-400/85"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth={1.75}
                aria-hidden
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5"
                />
              </svg>
            </div>
            <p className="text-sm text-slate-300">
              {file ? "Replace file" : "Drop audio here or click to browse"}
            </p>
            <p className="text-xs text-slate-500 mt-1">
              Full audio is transcribed (long videos may take a few minutes)
            </p>
          </div>
        </div>

        {file && previewUrl && (
          <div className="mt-3 glass-panel-subtle p-4">
            <p
              className="text-sm text-slate-400 mb-3 truncate flex items-center gap-2"
              title={file.name}
            >
              <span className="inline-block h-2 w-2 rounded-full bg-fuchsia-400 shadow-[0_0_8px_rgba(232,121,249,0.8)] shrink-0" />
              {file.name}
            </p>
            <audio
              controls
              src={previewUrl}
              className="w-full h-10 accent-violet-500 opacity-90"
              preload="metadata"
            >
              Your browser does not support audio playback.
            </audio>
          </div>
        )}
      </div>

      <div>
        <label className="block text-sm font-medium text-slate-300/90 mb-2">
          Audio language
        </label>
        <div className="grid grid-cols-2 gap-2">
          {LANGUAGE_OPTIONS.map((opt) => {
            const active = language === opt.value;
            return (
              <button
                key={opt.value}
                type="button"
                onClick={() => onLanguageChange?.(opt.value)}
                aria-pressed={active}
                className={`rounded-lg border px-4 py-2.5 text-sm font-medium transition ${
                  active
                    ? "border-violet-400/60 bg-violet-500/20 text-white shadow-[0_0_12px_rgba(139,92,246,0.35)]"
                    : "border-white/10 bg-white/5 text-slate-300 hover:border-white/25 hover:text-white"
                }`}
              >
                {opt.label}
              </button>
            );
          })}
        </div>
        <p className="text-xs text-slate-500 mt-1.5">
          Pick the spoken language for faster, more accurate transcription.
        </p>
      </div>

      <div>
        <label className="block text-sm font-medium text-slate-300/90 mb-2">
          Your question
        </label>
        <textarea
          value={question}
          onChange={(e) => onQuestionChange(e.target.value)}
          rows={2}
          className="glass-input resize-none text-sm"
          placeholder="e.g. What is happening in this scene?"
        />
      </div>

      <button
        type="button"
        onClick={onSubmit}
        disabled={disabled || loading || !file}
        className="glass-btn w-full overflow-hidden relative"
      >
        {loading ? (
          <span className="flex flex-col items-center gap-1.5 w-full py-0.5">
            {/* Top row: spinner + "Analyzing…" + percentage and remaining time badges */}
            <span className="flex items-center gap-2 w-full justify-center">
              <span className="h-4 w-4 rounded-full border-2 border-white/30 border-t-white animate-spin shrink-0" />
              <span className="font-semibold tracking-wide">Analyzing…</span>
              <span className="ml-1 rounded-full bg-white/15 px-2 py-0.5 text-xs font-bold tabular-nums text-violet-100 border border-white/20">
                {pct}%
              </span>
              {timeRemaining !== null ? (
                <span className="text-[10px] uppercase font-bold text-violet-200/90 tracking-wider">
                  {timeRemaining <= 2 ? "almost done" : `~${timeRemaining}s left`}
                </span>
              ) : null}
            </span>

            {/* Stage description */}
            {loadingStage ? (
              <span className="text-xs font-normal text-violet-100/75 text-center leading-tight px-2">
                {loadingStage}
              </span>
            ) : null}

            {/* Progress bar */}
            <span className="w-full mt-0.5 block px-0.5">
              <span className="flex items-center gap-2">
                <span className="flex-1 block rounded-full bg-white/10 h-2 overflow-hidden">
                  <span
                    className="block h-full rounded-full bg-gradient-to-r from-violet-400 via-fuchsia-400 to-pink-400"
                    style={{
                      width: `${pct}%`,
                      transition: "width 0.35s ease-out",
                      boxShadow: "0 0 8px rgba(167,139,250,0.6)",
                    }}
                  />
                </span>
              </span>
            </span>
          </span>
        ) : (
          "Run analysis"
        )}
      </button>
    </div>
  );
}
