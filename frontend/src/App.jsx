import { useState } from 'react'
import { SearchForm } from './components/SearchForm'
import { ComparisonTable } from './components/ComparisonTable'
import { ResultsAccordion } from './components/ResultsAccordion'
import { LoadingState } from './components/LoadingState'
import { ResultsSkeleton } from './components/Skeleton'
import { usePredictions } from './hooks/usePredictions'

function App() {
  const { results, loading, progress, errors, run, reset } = usePredictions()
  const [hasRun, setHasRun] = useState(false)

  const handleSubmit = (tickers, startDate, endDate) => {
    setHasRun(true)
    run(tickers, startDate, endDate)
  }

  return (
    <div className="min-h-screen bg-ink text-white p-6 md:p-12">
      <div className="max-w-7xl mx-auto">
        <header className="mb-8">
          <h1 className="text-3xl md:text-4xl font-bold uppercase tracking-tight font-display" style={{ textWrap: 'balance' }}>
            Stock Volatility Prediction
          </h1>
          <p className="text-gray-300 mt-2 font-mono text-sm">
            XGBoost-powered volatility forecasting
          </p>
        </header>

        <section className="mb-8">
          <SearchForm onSubmit={handleSubmit} loading={loading} />
        </section>

        {loading && (
          <section className="mb-8">
            <LoadingState current={progress.current} total={progress.total} ticker={progress.ticker} />
            {progress.current > 0 && <ResultsSkeleton />}
          </section>
        )}

        {Object.keys(errors).length > 0 && (
          <section className="mb-8">
            {Object.entries(errors).map(([ticker, msg]) => (
              <div key={ticker} className="bg-surface border border-red/50 p-4 mb-2">
                <div className="font-mono text-sm text-red/80">
                  <span className="text-red">{ticker}:</span> {msg}
                </div>
              </div>
            ))}
          </section>
        )}

        {results.length > 0 && !loading && (
          <>
            {hasRun && results.length > 1 && (
              <section className="mb-8">
                <h3 className="text-xs uppercase tracking-tight text-gray-300 font-mono mb-3">Comparison</h3>
                <ComparisonTable results={results} />
              </section>
            )}

            <section className="mb-8">
              <h3 className="text-xs uppercase tracking-tight text-gray-300 font-mono mb-3">
                {results.length === 1 ? 'Results' : 'Details'}
              </h3>
              <ResultsAccordion results={results} />
            </section>
          </>
        )}

        {!hasRun && !loading && (
          <div className="text-center font-mono text-sm mt-20 max-w-md mx-auto">
            <div className="text-accent text-4xl mb-4 font-semibold" style={{ fontFamily: 'var(--font-display)' }}>{">_"}</div>
            <p className="text-gray-300">Enter ticker symbols above to analyze volatility.</p>
            <p className="text-muted mt-2">Try <span className="text-accent">AAPL</span> or <span className="text-accent">AAPL,MSFT,GOOG</span> for a side-by-side comparison.</p>
            <div className="mt-6 grid grid-cols-3 gap-3 text-xs text-muted">
              <div className="bg-surface border border-edge p-3">
                <div className="text-gray-300 font-semibold mb-1">RMSE</div>
                <div>Root Mean Squared Error — lower is better</div>
              </div>
              <div className="bg-surface border border-edge p-3">
                <div className="text-gray-300 font-semibold mb-1">R²</div>
                <div>Variance explained — higher is better</div>
              </div>
              <div className="bg-surface border border-edge p-3">
                <div className="text-gray-300 font-semibold mb-1">Multi-Ticker</div>
                <div>Compare multiple stocks side by side</div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default App
