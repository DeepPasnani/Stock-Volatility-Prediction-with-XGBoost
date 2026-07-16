export function LoadingState({ current, total, ticker }) {
  return (
    <div className="bg-[#111827] border border-[#1e293b] p-6">
      <div className="font-mono text-sm text-gray-300">
        <div className="mb-1">
          <span className="text-[#22d3ee]">$</span> Analyzing {ticker} ({current}/{total})...
        </div>
        <span className="text-[#22d3ee]">█</span>
      </div>
    </div>
  )
}
