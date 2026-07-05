export default function GlassBackground({ children, className = "" }) {
  return (
    <div className={`relative min-h-screen overflow-hidden ${className}`}>
      <div
        className="pointer-events-none fixed inset-0 bg-[#07040f]"
        aria-hidden
      />
      <div
        className="pointer-events-none fixed inset-0 bg-gradient-to-br from-[#0c0618] via-[#1a0b2e] to-[#0a0612]"
        aria-hidden
      />
      <div
        className="pointer-events-none fixed -top-32 -left-32 h-[28rem] w-[28rem] rounded-full bg-violet-600/25 blur-[100px] animate-float"
        aria-hidden
      />
      <div
        className="pointer-events-none fixed top-1/3 -right-24 h-80 w-80 rounded-full bg-fuchsia-600/20 blur-[90px] animate-float-delayed"
        aria-hidden
      />
      <div
        className="pointer-events-none fixed -bottom-24 left-1/3 h-96 w-96 rounded-full bg-amber-500/10 blur-[110px] animate-float-slow"
        aria-hidden
      />
      <div
        className="pointer-events-none fixed inset-0 bg-[radial-gradient(ellipse_at_top,rgba(139,92,246,0.12),transparent_50%)]"
        aria-hidden
      />
      <div className="relative z-10">{children}</div>
    </div>
  );
}
