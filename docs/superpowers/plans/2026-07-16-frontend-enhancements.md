# Frontend Enhancements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) for syntax tracking.

**Goal:** Upgrade the React frontend with multi-ticker comparison, form validation, real loading progress, saved searches, CSV download — no new dependencies.

**Architecture:** New `usePredictions` hook replaces `usePredict` with sequential multi-ticker support. New `ComparisonTable` and `ResultsAccordion` components. `SearchForm` gets validation + recent searches. `LoadingState` uses real progress.

**Tech Stack:** React 19, Recharts 3, Tailwind CSS 4, Vite 8

## Global Constraints

- No new npm dependencies (use existing recharts, react, tailwind)
- No backend changes
- No routing, no state management library

---

### Task 1: Multi-ticker hook

**Files:**
- Create: `src/hooks/usePredictions.js`
- Delete: `src/hooks/usePredict.js`

- [ ] **Step 1: Create `src/hooks/usePredictions.js`**

```jsx
import { useState, useCallback } from 'react'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export function usePredictions() {
  const [results, setResults] = useState([])
  const [loading, setLoading] = useState(false)
  const [progress, setProgress] = useState({ current: 0, total: 0, ticker: '' })
  const [errors, setErrors] = useState({})

  const run = useCallback(async (tickers, startDate, endDate) => {
    setLoading(true)
    setResults([])
    setErrors({})
    setProgress({ current: 0, total: tickers.length, ticker: '' })

    const collected = []

    for (let i = 0; i < tickers.length; i++) {
      const ticker = tickers[i]
      setProgress({ current: i + 1, total: tickers.length, ticker })

      try {
        const response = await fetch(`${API_URL}/api/predict`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            ticker,
            start_date: startDate,
            end_date: endDate,
          }),
        })

        const result = await response.json()

        if (!response.ok) {
          throw new Error(result.detail || 'Prediction failed')
        }

        collected.push(result)
        setResults([...collected])
      } catch (err) {
        setErrors((prev) => ({ ...prev, [ticker]: err.message }))
      }
    }

    setLoading(false)
    setProgress({ current: 0, total: 0, ticker: '' })
  }, [])

  const reset = useCallback(() => {
    setResults([])
    setLoading(false)
    setProgress({ current: 0, total: 0, ticker: '' })
    setErrors({})
  }, [])

  return { results, loading, progress, errors, run, reset }
}
```

- [ ] **Step 2: Delete old hook**

```bash
rm src/hooks/usePredict.js
```

- [ ] **Step 3: Verify no import errors yet** (App.jsx still imports usePredict, will fix later)

---

### Task 2: SearchForm with multi-ticker, validation, saved searches

**Files:**
- Modify: `src/components/SearchForm.jsx`

- [ ] **Step 1: Rewrite `SearchForm.jsx`**

```jsx
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
          <label className="text-xs uppercase tracking-tight text-gray-400 font-mono">Tickers</label>
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
          <span className="text-xs text-gray-500 font-mono self-center">Recent:</span>
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
```

---

### Task 3: ComparisonTable component

**Files:**
- Create: `src/components/ComparisonTable.jsx`

- [ ] **Step 1: Create `ComparisonTable.jsx`**

```jsx
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
    <div className="bg-[#111827] border border-[#1e293b] overflow-x-auto">
      <table className="w-full font-mono text-sm">
        <thead>
          <tr className="border-b border-[#1e293b] text-gray-400 text-xs uppercase">
            <th onClick={() => toggleSort('ticker')} className="cursor-pointer text-left p-3 hover:text-white transition-colors">Ticker{arrow('ticker')}</th>
            <th onClick={() => toggleSort('rmse')} className="cursor-pointer text-right p-3 hover:text-white transition-colors">RMSE{arrow('rmse')}</th>
            <th onClick={() => toggleSort('r2')} className="cursor-pointer text-right p-3 hover:text-white transition-colors">R²{arrow('r2')}</th>
            <th onClick={() => toggleSort('train_rows')} className="cursor-pointer text-right p-3 hover:text-white transition-colors">Train{arrow('train_rows')}</th>
            <th onClick={() => toggleSort('test_rows')} className="cursor-pointer text-right p-3 hover:text-white transition-colors">Test{arrow('test_rows')}</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((r) => (
            <tr key={r.ticker} className="border-b border-[#1e293b]/50 hover:bg-[#1e293b]/30 transition-colors">
              <td className="p-3 text-white font-semibold">{r.ticker}</td>
              <td className={`p-3 text-right ${r.rmse === bestRmse ? 'text-green-400' : 'text-gray-300'}`}>
                {r.rmse.toFixed(4)}
              </td>
              <td className={`p-3 text-right ${r.r2 === bestR2 ? 'text-green-400' : 'text-gray-300'}`}>
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
```

---

### Task 4: ResultsAccordion component

**Files:**
- Create: `src/components/ResultsAccordion.jsx`

- [ ] **Step 1: Create `ResultsAccordion.jsx`**

```jsx
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
        <div key={result.ticker} className="bg-[#111827] border border-[#1e293b]">
          <button
            onClick={() => setOpenIndex(openIndex === i ? -1 : i)}
            className="w-full flex items-center justify-between p-4 text-left hover:bg-[#1e293b]/50 transition-colors"
          >
            <span className="font-mono text-white font-semibold">{result.ticker}</span>
            <span className="font-mono text-xs text-gray-400">
              RMSE {result.rmse.toFixed(4)} · R² {result.r2.toFixed(4)}
              <span className="ml-3 text-[#22d3ee]">{openIndex === i ? '▲' : '▼'}</span>
            </span>
          </button>
          {openIndex === i && (
            <div className="p-4 pt-0 space-y-6">
              <MetricsPanel data={result} />
              <PredictionChart data={result} />
              <FeatureChart data={result} />
              <button
                onClick={() => downloadCSV(result)}
                className="font-mono text-xs text-[#22d3ee] border border-[#22d3ee] px-3 py-1 hover:bg-[#22d3ee]/10 transition-colors uppercase"
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
```

---

### Task 5: LoadingState with real progress

**Files:**
- Modify: `src/components/LoadingState.jsx`

- [ ] **Step 1: Rewrite `LoadingState.jsx`**

```jsx
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
```

---

### Task 6: Wire up App.jsx

**Files:**
- Modify: `src/App.jsx`

- [ ] **Step 1: Rewrite `App.jsx`**

```jsx
import { useState } from 'react'
import { SearchForm } from './components/SearchForm'
import { ComparisonTable } from './components/ComparisonTable'
import { ResultsAccordion } from './components/ResultsAccordion'
import { LoadingState } from './components/LoadingState'
import { usePredictions } from './hooks/usePredictions'

function App() {
  const { results, loading, progress, errors, run, reset } = usePredictions()
  const [hasRun, setHasRun] = useState(false)

  const handleSubmit = (tickers, startDate, endDate) => {
    setHasRun(true)
    run(tickers, startDate, endDate)
  }

  return (
    <div className="min-h-screen bg-[#0b0f1a] text-white p-6 md:p-12">
      <div className="max-w-7xl mx-auto">
        <header className="mb-8">
          <h1 className="text-3xl md:text-4xl font-bold uppercase tracking-tight" style={{ fontFamily: 'Syne, sans-serif' }}>
            Stock Volatility Prediction
          </h1>
          <p className="text-gray-400 mt-2 font-mono text-sm">
            XGBoost-powered volatility forecasting
          </p>
        </header>

        <section className="mb-8">
          <SearchForm onSubmit={handleSubmit} loading={loading} />
        </section>

        {loading && (
          <section className="mb-8">
            <LoadingState current={progress.current} total={progress.total} ticker={progress.ticker} />
          </section>
        )}

        {Object.keys(errors).length > 0 && (
          <section className="mb-8">
            {Object.entries(errors).map(([ticker, msg]) => (
              <div key={ticker} className="bg-[#111827] border border-red-500/50 p-4 mb-2">
                <div className="font-mono text-sm text-red-400">
                  <span className="text-red-500">{ticker}:</span> {msg}
                </div>
              </div>
            ))}
          </section>
        )}

        {results.length > 0 && !loading && (
          <>
            {hasRun && results.length > 1 && (
              <section className="mb-8">
                <h3 className="text-xs uppercase tracking-tight text-gray-400 font-mono mb-3">Comparison</h3>
                <ComparisonTable results={results} />
              </section>
            )}

            <section className="mb-8">
              <h3 className="text-xs uppercase tracking-tight text-gray-400 font-mono mb-3">
                {results.length === 1 ? 'Results' : 'Details'}
              </h3>
              <ResultsAccordion results={results} />
            </section>
          </>
        )}

        {!hasRun && !loading && (
          <div className="text-center text-gray-500 font-mono text-sm mt-20">
            <p>Enter ticker symbols above to analyze volatility.</p>
            <p className="mt-1">Try <span className="text-[#22d3ee]">AAPL</span> or <span className="text-[#22d3ee]">AAPL,MSFT,GOOG</span>.</p>
          </div>
        )}
      </div>
    </div>
  )
}

export default App
```

---

### Build check

- [ ] **Step 1: Install deps and build**

```bash
cd frontend && npm install 2>&1 | tail -3 && npm run build 2>&1
```
Expected: Build succeeds with no errors.
