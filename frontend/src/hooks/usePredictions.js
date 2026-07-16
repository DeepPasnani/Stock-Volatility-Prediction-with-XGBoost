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
