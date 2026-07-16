import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts'

export function FeatureChart({ data }) {
  if (!data || !data.feature_importance) return null

  const chartData = [...data.feature_importance].reverse()

  return (
    <div className="bg-[#111827] border border-[#1e293b] p-4">
      <h3 className="text-xs uppercase tracking-tight text-gray-400 font-mono mb-4">
        Feature Importance (Top 15)
      </h3>
      <div className="h-[300px]">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={chartData}
            layout="vertical"
            margin={{ top: 5, right: 30, left: 80, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" horizontal={true} vertical={false} />
            <XAxis
              type="number"
              tick={{ fill: '#9ca3af', fontSize: 11, fontFamily: 'IBM Plex Mono' }}
              stroke="#334155"
            />
            <YAxis
              type="category"
              dataKey="feature"
              tick={{ fill: '#9ca3af', fontSize: 11, fontFamily: 'IBM Plex Mono' }}
              stroke="#334155"
              width={75}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#0b0f1a',
                border: '1px solid #334155',
                fontFamily: 'IBM Plex Mono',
                fontSize: '12px',
              }}
              labelStyle={{ color: '#9ca3af' }}
              formatter={(value) => value.toFixed(4)}
            />
            <Bar dataKey="importance" fill="#22d3ee" opacity={0.8} radius={[0, 4, 4, 0]}>
              {chartData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill="#22d3ee" opacity={0.8} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}