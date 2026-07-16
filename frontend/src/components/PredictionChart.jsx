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
    <div className="bg-[#111827] border border-[#1e293b] p-4">
      <h3 className="text-xs uppercase tracking-tight text-gray-400 font-mono mb-4">
        Actual vs Predicted Volatility
      </h3>
      <div className="h-[300px]">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ top: 5, right: 30, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis
              dataKey="date"
              tick={{ fill: '#9ca3af', fontSize: 11, fontFamily: 'IBM Plex Mono' }}
              tickFormatter={(value, index) => (index % showEveryNth === 0 ? value : '')}
              stroke="#334155"
            />
            <YAxis
              tick={{ fill: '#9ca3af', fontSize: 11, fontFamily: 'IBM Plex Mono' }}
              stroke="#334155"
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#0b0f1a',
                border: '1px solid #334155',
                fontFamily: 'IBM Plex Mono',
                fontSize: '12px',
              }}
              labelStyle={{ color: '#9ca3af' }}
            />
            <Line
              type="monotone"
              dataKey="actual"
              stroke="#f97316"
              strokeWidth={2}
              dot={false}
              name="Actual"
            />
            <Line
              type="monotone"
              dataKey="predicted"
              stroke="#22d3ee"
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
          <div className="w-3 h-0.5 bg-[#f97316]"></div>
          <span className="text-xs font-mono text-gray-400">Actual</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-0.5 bg-[#22d3ee] border-dashed border-t border-[#22d3ee]"></div>
          <span className="text-xs font-mono text-gray-400">Predicted</span>
        </div>
      </div>
    </div>
  )
}