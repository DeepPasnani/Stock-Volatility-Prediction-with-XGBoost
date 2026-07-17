import { useState, useEffect } from 'react'

const STORAGE_KEY = 'recentSearches'

function loadRecent() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]')
  } catch {
    return []
  }
}

function saveRecent(entry) {
  const list = loadRecent().filter((s) => s.tickers !== entry.tickers)
  list.unshift(entry)
  if (list.length > 5) list.length = 5
  localStorage.setItem(STORAGE_KEY, JSON.stringify(list))
}

export function SearchForm({ onSubmit, loading }) {
  const [tickerInput, setTickerInput] = useState('AAPL')
  const [startDate, setStartDate] = useState('2020-01-01')
  const [endDate, setEndDate] = useState('')
  const [validationError, setValidationError] = useState('')
  const [recentSearches, setRecentSearches] = useState([])

  useEffect(() => {
    const today = new Date().toISOString().split('T')[0]
    setEndDate(today)
    setRecentSearches(loadRecent())
  }, [])

  const validate = (tickers, start, end) => {
    if (tickers.length === 0) return 'Enter at least one ticker'
    const startD = new Date(start)
    const endD = new Date(end)
    const diff = (endD - startD) / (1000 * 60 * 60 * 24)
    if (diff < 60) return 'Date range must be at least 60 days'
    return ''
  }

  const handleSubmit = (e) => {
    e.preventDefault()
    const tickers = tickerInput
      .split(',')
      .map((t) => t.trim().toUpperCase())
      .filter(Boolean)

    const err = validate(tickers, startDate, endDate)
    setValidationError(err)
    if (err) return

    const entry = { tickers: tickers.join(','), startDate, endDate }
    saveRecent(entry)
    setRecentSearches(loadRecent())
    onSubmit(tickers, startDate, endDate)
  }

  const handleRecentClick = (entry) => {
    setTickerInput(entry.tickers)
    setStartDate(entry.startDate)
    setEndDate(entry.endDate)
    const tickers = entry.tickers.split(',').filter(Boolean)
    onSubmit(tickers, entry.startDate, entry.endDate)
  }

  return (
    <div>
      <form onSubmit={handleSubmit} className="grid grid-cols-1 md:grid-cols-4 gap-4 items-end">
        <div className="flex flex-col gap-1">
          <label className="text-xs uppercase tracking-tight text-gray-300 font-mono">Tickers</label>
          <input
            type="text"
            value={tickerInput}
            onChange={(e) => { setTickerInput(e.target.value); setValidationError('') }}
            disabled={loading}
            placeholder="AAPL,MSFT,GOOG"
            className="bg-[#0b0f1a] border border-[#334155] px-3 py-2 font-mono text-sm text-white focus:border-[#22d3ee] focus:outline-none disabled:opacity-50"
          />
        </div>
        <div className="flex flex-col gap-1">
          <label className="text-xs uppercase tracking-tight text-gray-300 font-mono">Start Date</label>
          <input
            type="date"
            value={startDate}
            onChange={(e) => setStartDate(e.target.value)}
            disabled={loading}
            className="bg-[#0b0f1a] border border-[#334155] px-3 py-2 font-mono text-sm text-white focus:border-[#22d3ee] focus:outline-none disabled:opacity-50"
          />
        </div>
        <div className="flex flex-col gap-1">
          <label className="text-xs uppercase tracking-tight text-gray-300 font-mono">End Date</label>
          <input
            type="date"
            value={endDate}
            onChange={(e) => setEndDate(e.target.value)}
            disabled={loading}
            className="bg-[#0b0f1a] border border-[#334155] px-3 py-2 font-mono text-sm text-white focus:border-[#22d3ee] focus:outline-none disabled:opacity-50"
          />
        </div>
        <button
          type="submit"
          disabled={loading}
          className="bg-[#0b0f1a] border border-[#22d3ee] px-4 py-2 font-mono text-sm text-[#22d3ee] uppercase tracking-wider hover:bg-[#22d3ee] hover:text-[#0b0f1a] transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {loading ? 'Processing...' : '▶ Analyze'}
        </button>
      </form>

      {validationError && (
        <p className="text-red-400 font-mono text-xs mt-2">{validationError}</p>
      )}

      {recentSearches.length > 0 && !loading && (
        <div className="flex gap-2 mt-3 flex-wrap">
          <span className="text-xs text-gray-400 font-mono self-center">Recent:</span>
          {recentSearches.map((entry, i) => (
            <button
              key={i}
              onClick={() => handleRecentClick(entry)}
              className="text-xs font-mono text-[#22d3ee] border border-[#22d3ee]/30 px-2 py-1 hover:bg-[#22d3ee]/10 transition-colors"
            >
              {entry.tickers}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}
