import { useState, useMemo } from 'react'

function bestValue(results, key, higherBetter) {
  if (results.length === 0) return null
  return higherBetter
    ? Math.max(...results.map((r) => r[key]))
    : Math.min(...results.map((r) => r[key]))
}

export function ComparisonTable({ results }) {
  const [sortKey, setSortKey] = useState('rmse')
  const [sortDir, setSortDir] = useState('asc')

  const sorted = useMemo(() => {
    const copy = [...results]
    copy.sort((a, b) => {
      const av = a[sortKey] ?? 0
      const bv = b[sortKey] ?? 0
      return sortDir === 'asc' ? av - bv : bv - av
    })
    return copy
  }, [results, sortKey, sortDir])

  const toggleSort = (key) => {
    if (sortKey === key) {
      setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'))
    } else {
      setSortKey(key)
      setSortDir(key === 'r2' ? 'desc' : 'asc')
    }
  }

  const bestRmse = bestValue(results, 'rmse', false)
  const bestR2 = bestValue(results, 'r2', true)

  const arrow = (key) => {
    if (sortKey !== key) return ''
    return sortDir === 'asc' ? ' ▲' : ' ▼'
  }

  return (
    <div className="bg-surface border border-edge overflow-x-auto">
      <table className="w-full font-mono text-sm">
        <thead>
          <tr className="border-b border-edge text-muted text-xs uppercase">
            <th onClick={() => toggleSort('ticker')} className="cursor-pointer text-left p-3 hover:text-white transition-colors">Ticker{arrow('ticker')}</th>
            <th onClick={() => toggleSort('rmse')} className="cursor-pointer text-right p-3 hover:text-white transition-colors">RMSE{arrow('rmse')}</th>
            <th onClick={() => toggleSort('r2')} className="cursor-pointer text-right p-3 hover:text-white transition-colors">R²{arrow('r2')}</th>
            <th onClick={() => toggleSort('train_rows')} className="cursor-pointer text-right p-3 hover:text-white transition-colors">Train{arrow('train_rows')}</th>
            <th onClick={() => toggleSort('test_rows')} className="cursor-pointer text-right p-3 hover:text-white transition-colors">Test{arrow('test_rows')}</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((r) => (
            <tr key={r.ticker} className="border-b border-edge/50 hover:bg-edge/30 transition-colors">
              <td className="p-3 text-white font-semibold">{r.ticker}</td>
              <td className={`p-3 text-right ${r.rmse === bestRmse ? 'text-green-bright' : 'text-gray-300'}`}>
                {r.rmse.toFixed(4)}
              </td>
              <td className={`p-3 text-right ${r.r2 === bestR2 ? 'text-green-bright' : 'text-gray-300'}`}>
                {r.r2.toFixed(4)}
              </td>
              <td className="p-3 text-right text-gray-300">{r.train_rows}</td>
              <td className="p-3 text-right text-gray-300">{r.test_rows}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
