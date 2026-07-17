import { useState, useEffect } from 'react'

const STEPS = [
  { key: 'fetch', text: 'Fetching {ticker} data...', delay: 0 },
  { key: 'features', text: 'Engineering features...', delay: 1500 },
  { key: 'train', text: 'Training XGBoost model...', delay: 3000 },
  { key: 'predict', text: 'Generating predictions...', delay: 4500 },
]

export function LoadingState({ ticker }) {
  const [visibleSteps, setVisibleSteps] = useState([])
  const [cursorVisible, setCursorVisible] = useState(true)

  useEffect(() => {
    setVisibleSteps([])
    
    const timers = STEPS.map((step) =>
      setTimeout(() => {
        setVisibleSteps((prev) => [...prev, step])
      }, step.delay)
    )

    return () => timers.forEach(clearTimeout)
  }, [ticker])

  useEffect(() => {
    const interval = setInterval(() => {
      setCursorVisible((v) => !v)
    }, 500)
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="bg-[#111827] border border-[#1e293b] p-6">
      <div className="font-mono text-sm text-gray-300">
        {visibleSteps.map((step) => {
          const text = step.text.replace('{ticker}', ticker)
          return (
            <div key={step.key} className="mb-1">
              <span className="text-[#22d3ee]">$</span> {text}
            </div>
          )
        })}
        {visibleSteps.length > 0 && (
          <span className="text-[#22d3ee]">
            {cursorVisible ? '█' : ' '}
          </span>
        )}
      </div>
    </div>
  )
}
