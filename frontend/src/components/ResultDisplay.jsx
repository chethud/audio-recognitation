export default function ResultDisplay({ result, error }) {
  if (error) {
    return <div className="glass-error">{error}</div>;
  }
  if (!result) {
    return (
      <div className="glass-panel-subtle px-4 py-8 text-center">
        <p className="text-slate-500 text-sm">
          Results will appear here after you run an analysis.
        </p>
      </div>
    );
  }

  const { transcript, sounds, emotion, answer } = result;

  return (
    <div className="space-y-4 text-left">
      <section className="glass-panel-subtle p-4">
        <h3 className="font-display text-xs uppercase tracking-widest text-cyan-300/70 mb-2">
          Transcript
        </h3>
        <p className="text-slate-100 whitespace-pre-wrap leading-relaxed">
          {transcript || "—"}
        </p>
      </section>
      <section className="glass-panel-subtle p-4">
        <h3 className="font-display text-xs uppercase tracking-widest text-cyan-300/70 mb-2">
          Sounds
        </h3>
        <div className="flex flex-wrap gap-2">
          {(sounds || []).length === 0 ? (
            <span className="text-slate-500">—</span>
          ) : (
            sounds.map((s) => (
              <span key={s} className="glass-tag">
                {s}
              </span>
            ))
          )}
        </div>
      </section>
      <section className="glass-panel-subtle p-4">
        <h3 className="font-display text-xs uppercase tracking-widest text-cyan-300/70 mb-2">
          Emotion
        </h3>
        <p className="text-slate-100 capitalize">{emotion || "—"}</p>
      </section>
      <section className="glass-panel-subtle p-4 border-teal-400/20">
        <h3 className="font-display text-xs uppercase tracking-widest text-teal-300/70 mb-2">
          Answer
        </h3>
        <p className="text-slate-100 whitespace-pre-wrap leading-relaxed border-l-2 border-teal-400/40 pl-4">
          {answer || "—"}
        </p>
      </section>
    </div>
  );
}
