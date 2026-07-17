import { useState } from 'react'
import { SearchForm } from './components/SearchForm'
import { MetricsPanel } from './components/MetricsPanel'
import { PredictionChart } from './components/PredictionChart'
import { FeatureChart } from './components/FeatureChart'
import { LoadingState } from './components/LoadingState'
import { usePredict } from './hooks/usePredict'

function App() {
  const { data, loading, error, predict, reset } = usePredict()
  const [currentTicker, setCurrentTicker] = useState('AAPL')

  const handleSubmit = (ticker, startDate, endDate) => {
    setCurrentTicker(ticker)
    predict(ticker, startDate, endDate)
  }

  const handleRetry = () => {
    reset()
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

        {loading && <LoadingState ticker={currentTicker} />}

        {error && (
          <div className="bg-[#111827] border border-red-500/50 p-4 mb-8">
            <div className="flex items-center justify-between">
              <div className="font-mono text-sm text-red-400">
                <span className="text-red-500">ERROR:</span> {error}
              </div>
              <button
                onClick={handleRetry}
                className="font-mono text-xs text-[#22d3ee] hover:underline uppercase"
              >
                Try Again
              </button>
            </div>
          </div>
        )}

        {data && !loading && (
          <>
            <section className="mb-8">
              <MetricsPanel data={data} />
            </section>

            <section className="mb-8">
              <PredictionChart data={data} />
            </section>

            <section className="mb-8">
              <FeatureChart data={data} />
            </section>
          </>
        )}
      </div>
    </div>
  )
}

export default App
