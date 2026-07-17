import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts'

export function PredictionChart({ data }) {
  if (!data || !data.predictions) return null

  const chartData = data.predictions.map((p) => ({
    date: p.date,
    actual: p.actual,
    predicted: p.predicted,
  }))

  const showEveryNth = Math.ceil(chartData.length / 10)

  return (
    <div className="bg-surface border border-edge p-4">
      <h3 className="text-xs uppercase tracking-tight text-muted font-mono mb-4">
        Actual vs Predicted Volatility
      </h3>
      <div className="h-[300px]">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ top: 5, right: 30, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
            <XAxis
              dataKey="date"
              tick={{ fill: 'var(--muted)', fontSize: 11, fontFamily: 'IBM Plex Mono' }}
              tickFormatter={(value, index) => (index % showEveryNth === 0 ? value : '')}
              stroke="var(--stroke)"
            />
            <YAxis
              tick={{ fill: 'var(--muted)', fontSize: 11, fontFamily: 'IBM Plex Mono' }}
              stroke="var(--stroke)"
            />
            <Tooltip
              contentStyle={{
                backgroundColor: 'var(--ink)',
                border: '1px solid var(--stroke)',
                fontFamily: 'IBM Plex Mono',
                fontSize: '12px',
              }}
              labelStyle={{ color: 'var(--muted)' }}
            />
            <Line
              type="monotone"
              dataKey="actual"
              stroke="var(--orange)"
              strokeWidth={2}
              dot={false}
              name="Actual"
            />
            <Line
              type="monotone"
              dataKey="predicted"
              stroke="var(--accent)"
              strokeWidth={2}
              strokeDasharray="5 5"
              dot={false}
              name="Predicted"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div className="flex gap-6 mt-2">
        <div className="flex items-center gap-2">
          <div className="w-3 h-0.5 bg-orange"></div>
          <span className="text-xs font-mono text-muted">Actual</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-0.5 bg-accent border-dashed border-t border-accent"></div>
          <span className="text-xs font-mono text-muted">Predicted</span>
        </div>
      </div>
    </div>
  )
}
