import { useState } from 'react'
import { MetricsPanel } from './MetricsPanel'
import { PredictionChart } from './PredictionChart'
import { FeatureChart } from './FeatureChart'

function downloadCSV(result) {
  const rows = [['date', 'actual', 'predicted']]
  for (const p of result.predictions) {
    rows.push([p.date, p.actual, p.predicted])
  }
  const csv = rows.map((r) => r.join(',')).join('\n')
  const blob = new Blob([csv], { type: 'text/csv' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `${result.ticker}_volatility.csv`
  a.click()
  URL.revokeObjectURL(url)
}

export function ResultsAccordion({ results }) {
  const [openIndex, setOpenIndex] = useState(0)

  return (
    <div className="space-y-2">
      {results.map((result, i) => (
        <div key={result.ticker} className="bg-surface border border-edge">
          <button
            onClick={() => setOpenIndex(openIndex === i ? -1 : i)}
            className="w-full flex items-center justify-between p-4 text-left hover:bg-edge/50 transition-colors"
          >
            <span className="font-mono text-white font-semibold">{result.ticker}</span>
            <span className="font-mono text-xs text-gray-300">
              RMSE {result.rmse.toFixed(4)} · R² {result.r2.toFixed(4)}
              <span className="ml-3 text-accent">{openIndex === i ? '▲' : '▼'}</span>
            </span>
          </button>
          {openIndex === i && (
            <div className="p-4 pt-0 space-y-6">
              <MetricsPanel data={result} />
              <PredictionChart data={result} />
              <FeatureChart data={result} />
              <button
                onClick={() => downloadCSV(result)}
                className="font-mono text-xs text-accent border border-accent px-3 py-1 hover:bg-accent/10 transition-colors uppercase"
              >
                ↓ Download CSV
              </button>
            </div>
          )}
        </div>
      ))}
    </div>
  )
}
