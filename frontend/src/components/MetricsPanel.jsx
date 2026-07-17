export function MetricsPanel({ data }) {
  if (!data) return null

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
      <div className="bg-[#111827] border border-[#1e293b] p-4">
        <div className="text-xs uppercase tracking-tight text-gray-300 font-mono mb-1">RMSE</div>
        <div className="text-2xl font-mono text-[#22d3ee]">{data.rmse.toFixed(4)}</div>
      </div>
      <div className="bg-[#111827] border border-[#1e293b] p-4">
        <div className="text-xs uppercase tracking-tight text-gray-300 font-mono mb-1">R²</div>
        <div className="text-2xl font-mono text-[#22d3ee]">{data.r2.toFixed(4)}</div>
      </div>
      <div className="bg-[#111827] border border-[#1e293b] p-4">
        <div className="text-xs uppercase tracking-tight text-gray-300 font-mono mb-1">Train Rows</div>
        <div className="text-2xl font-mono text-white">{data.train_rows}</div>
      </div>
      <div className="bg-[#111827] border border-[#1e293b] p-4">
        <div className="text-xs uppercase tracking-tight text-gray-300 font-mono mb-1">Test Rows</div>
        <div className="text-2xl font-mono text-white">{data.test_rows}</div>
      </div>
    </div>
  )
}
