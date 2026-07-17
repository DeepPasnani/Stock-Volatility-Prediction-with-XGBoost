import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts'

export function FeatureChart({ data }) {
  if (!data || !data.feature_importance) return null

  const chartData = [...data.feature_importance].reverse()

  return (
    <div className="bg-surface border border-edge p-4">
      <h3 className="text-xs uppercase tracking-tight text-muted font-mono mb-4">
        Feature Importance (Top 15)
      </h3>
      <div className="h-[300px]">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={chartData}
            layout="vertical"
            margin={{ top: 5, right: 30, left: 80, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" horizontal={true} vertical={false} />
            <XAxis
              type="number"
              tick={{ fill: 'var(--muted)', fontSize: 11, fontFamily: 'IBM Plex Mono' }}
              stroke="var(--stroke)"
            />
            <YAxis
              type="category"
              dataKey="feature"
              tick={{ fill: 'var(--muted)', fontSize: 11, fontFamily: 'IBM Plex Mono' }}
              stroke="var(--stroke)"
              width={75}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: 'var(--ink)',
                border: '1px solid var(--stroke)',
                fontFamily: 'IBM Plex Mono',
                fontSize: '12px',
              }}
              labelStyle={{ color: 'var(--muted)' }}
              formatter={(value) => value.toFixed(4)}
            />
            <Bar dataKey="importance" fill="var(--accent)" opacity={0.8} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
