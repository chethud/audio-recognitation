export default function ResultDisplay({ result, error }) {
  if (error) {
    return (
      <div className="rounded-xl border border-red-500/40 bg-red-950/40 px-4 py-3 text-red-200 text-sm">
        {error}
      </div>
    );
  }
  if (!result) {
    return (
      <p className="text-slate-500 text-sm">
        Results will appear here after you run an analysis.
      </p>
    );
  }

  const { transcript, sounds, emotion, answer } = result;

  return (
    <div className="space-y-5 text-left">
      <section>
        <h3 className="font-display text-xs uppercase tracking-widest text-slate-500 mb-1">
          Transcript
        </h3>
        <p className="text-slate-100 whitespace-pre-wrap leading-relaxed">
          {transcript || "—"}
        </p>
      </section>
      <section>
        <h3 className="font-display text-xs uppercase tracking-widest text-slate-500 mb-1">
          Sounds
        </h3>
        <div className="flex flex-wrap gap-2">
          {(sounds || []).length === 0 ? (
            <span className="text-slate-500">—</span>
          ) : (
            sounds.map((s) => (
              <span
                key={s}
                className="rounded-full bg-slate-800 px-3 py-1 text-sm text-cyan-200/90"
              >
                {s}
              </span>
            ))
          )}
        </div>
      </section>
      <section>
        <h3 className="font-display text-xs uppercase tracking-widest text-slate-500 mb-1">
          Emotion
        </h3>
        <p className="text-slate-100 capitalize">{emotion || "—"}</p>
      </section>
      <section>
        <h3 className="font-display text-xs uppercase tracking-widest text-slate-500 mb-1">
          Answer
        </h3>
        <p className="text-slate-100 whitespace-pre-wrap leading-relaxed border-l-2 border-teal-500/50 pl-4">
          {answer || "—"}
        </p>
      </section>
    </div>
  );
}
