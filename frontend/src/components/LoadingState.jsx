export function LoadingState({ current, total, ticker }) {
  return (
    <div className="bg-surface border border-edge p-6">
      <div className="font-mono text-sm text-gray-300">
        <div className="mb-1">
          <span className="text-accent">$</span> Analyzing {ticker} ({current}/{total})...
        </div>
        <span className="text-accent">█</span>
      </div>
    </div>
  )
}
