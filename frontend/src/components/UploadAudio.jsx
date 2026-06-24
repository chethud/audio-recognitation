import { useEffect, useState } from "react";

export default function UploadAudio({
  file,
  onFileChange,
  question,
  onQuestionChange,
  onSubmit,
  loading,
  loadingStage = "",
  disabled,
}) {
  const [previewUrl, setPreviewUrl] = useState(null);

  useEffect(() => {
    if (!file) {
      setPreviewUrl(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  return (
    <div className="space-y-6">
      <div>
        <label className="block text-sm font-medium text-slate-300/90 mb-2">
          Audio file
        </label>
        <div className="glass-panel-subtle p-1">
          <input
            type="file"
            accept="audio/*,.mp4,.webm,.mkv,.avi,.mov"
            onChange={(e) => onFileChange(e.target.files?.[0] || null)}
            className="block w-full text-sm text-slate-300 file:mr-4 file:py-2.5 file:px-4 file:rounded-lg file:border file:border-white/10 file:bg-white/10 file:text-cyan-100 file:backdrop-blur-sm hover:file:bg-white/15 cursor-pointer"
          />
        </div>
        {file && previewUrl && (
          <div className="mt-4 glass-panel-subtle p-4">
            <p
              className="text-sm text-slate-400 mb-3 truncate flex items-center gap-2"
              title={file.name}
            >
              <span className="inline-block h-2 w-2 rounded-full bg-cyan-400 shadow-[0_0_8px_rgba(34,211,238,0.8)]" />
              {file.name}
            </p>
            <audio
              controls
              src={previewUrl}
              className="w-full h-10 accent-cyan-400 opacity-90"
              preload="metadata"
            >
              Your browser does not support audio playback.
            </audio>
          </div>
        )}
      </div>
      <div>
        <label className="block text-sm font-medium text-slate-300/90 mb-2">
          Question
        </label>
        <textarea
          value={question}
          onChange={(e) => onQuestionChange(e.target.value)}
          rows={3}
          className="glass-input resize-none"
          placeholder="e.g. Where is this scene likely taking place?"
        />
      </div>
      <button
        type="button"
        onClick={onSubmit}
        disabled={disabled || loading || !file}
        className="glass-btn w-full"
      >
        {loading ? (
          <span className="inline-flex flex-col items-center justify-center gap-1">
            <span className="inline-flex items-center gap-2">
              <span className="h-4 w-4 rounded-full border-2 border-white/30 border-t-white animate-spin" />
              Analyzing…
            </span>
            {loadingStage ? (
              <span className="text-xs font-normal text-cyan-100/80">
                {loadingStage}
              </span>
            ) : null}
          </span>
        ) : (
          "Analyze"
        )}
      </button>
    </div>
  );
}
