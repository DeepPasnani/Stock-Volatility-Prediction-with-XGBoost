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
