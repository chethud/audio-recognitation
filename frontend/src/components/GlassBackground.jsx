export default function GlassBackground({ children, className = "" }) {
  return (
    <div className={`relative min-h-screen overflow-hidden ${className}`}>
      <div
        className="pointer-events-none fixed inset-0 bg-[#030712]"
        aria-hidden
      />
      <div
        className="pointer-events-none fixed inset-0 bg-gradient-to-br from-slate-950 via-[#0a1628] to-slate-950"
        aria-hidden
      />
      <div
        className="pointer-events-none fixed -top-32 -left-32 h-[28rem] w-[28rem] rounded-full bg-cyan-500/20 blur-[100px] animate-float"
        aria-hidden
      />
      <div
        className="pointer-events-none fixed top-1/3 -right-24 h-80 w-80 rounded-full bg-teal-500/15 blur-[90px] animate-float-delayed"
        aria-hidden
      />
      <div
        className="pointer-events-none fixed -bottom-24 left-1/3 h-96 w-96 rounded-full bg-violet-600/10 blur-[110px] animate-float-slow"
        aria-hidden
      />
      <div
        className="pointer-events-none fixed inset-0 bg-[radial-gradient(ellipse_at_top,rgba(34,211,238,0.08),transparent_50%)]"
        aria-hidden
      />
      <div className="relative z-10">{children}</div>
    </div>
  );
}
