export default function UploadAudio({
  file,
  onFileChange,
  question,
  onQuestionChange,
  onSubmit,
  loading,
  disabled,
}) {
  return (
    <div className="space-y-6">
      <div>
        <label className="block text-sm font-medium text-slate-400 mb-2">
          Audio file
        </label>
        <input
          type="file"
          accept="audio/*,.mp4,.webm,.mkv,.avi,.mov"
          onChange={(e) => onFileChange(e.target.files?.[0] || null)}
          className="block w-full text-sm text-slate-300 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:bg-cyan-500/20 file:text-cyan-200 hover:file:bg-cyan-500/30 cursor-pointer"
        />
      </div>
      <div>
        <label className="block text-sm font-medium text-slate-400 mb-2">
          Question
        </label>
        <textarea
          value={question}
          onChange={(e) => onQuestionChange(e.target.value)}
          rows={3}
          className="w-full rounded-xl bg-slate-900/80 border border-slate-700 px-4 py-3 text-slate-100 placeholder:text-slate-600 focus:outline-none focus:ring-2 focus:ring-cyan-500/50"
          placeholder="e.g. Where is this scene likely taking place?"
        />
      </div>
      <button
        type="button"
        onClick={onSubmit}
        disabled={disabled || loading || !file}
        className="w-full rounded-xl bg-gradient-to-r from-cyan-600 to-teal-600 px-4 py-3 font-display font-semibold text-white shadow-lg shadow-cyan-900/30 disabled:opacity-40 disabled:cursor-not-allowed hover:from-cyan-500 hover:to-teal-500 transition"
      >
        {loading ? "Analyzing…" : "Analyze"}
      </button>
    </div>
  );
}
