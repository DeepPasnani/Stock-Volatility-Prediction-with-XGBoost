import { useState, useEffect } from 'react'

export function SearchForm({ onSubmit, loading }) {
  const [ticker, setTicker] = useState('AAPL')
  const [startDate, setStartDate] = useState('2020-01-01')
  const [endDate, setEndDate] = useState('')

  useEffect(() => {
    const today = new Date().toISOString().split('T')[0]
    setEndDate(today)
  }, [])

  const handleSubmit = (e) => {
    e.preventDefault()
    if (ticker.trim()) {
      onSubmit(ticker.trim().toUpperCase(), startDate, endDate)
    }
  }

  return (
    <form onSubmit={handleSubmit} className="grid grid-cols-1 md:grid-cols-4 gap-4 items-end">
      <div className="flex flex-col gap-1">
        <label className="text-xs uppercase tracking-tight text-gray-400 font-mono">Ticker</label>
        <input
          type="text"
          value={ticker}
          onChange={(e) => setTicker(e.target.value)}
          disabled={loading}
          className="bg-[#0b0f1a] border border-[#334155] px-3 py-2 font-mono text-sm text-white focus:border-[#22d3ee] focus:outline-none disabled:opacity-50"
        />
      </div>
      <div className="flex flex-col gap-1">
        <label className="text-xs uppercase tracking-tight text-gray-400 font-mono">Start Date</label>
        <input
          type="date"
          value={startDate}
          onChange={(e) => setStartDate(e.target.value)}
          disabled={loading}
          className="bg-[#0b0f1a] border border-[#334155] px-3 py-2 font-mono text-sm text-white focus:border-[#22d3ee] focus:outline-none disabled:opacity-50"
        />
      </div>
      <div className="flex flex-col gap-1">
        <label className="text-xs uppercase tracking-tight text-gray-400 font-mono">End Date</label>
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
        disabled={loading || !ticker.trim()}
        className="bg-[#0b0f1a] border border-[#22d3ee] px-4 py-2 font-mono text-sm text-[#22d3ee] uppercase tracking-wider hover:bg-[#22d3ee] hover:text-[#0b0f1a] transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {loading ? 'Processing...' : '▶ Analyze'}
      </button>
    </form>
  )
}
