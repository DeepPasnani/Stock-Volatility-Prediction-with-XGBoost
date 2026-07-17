export function Skeleton({ className = '', style = {} }) {
  return (
    <div
      className={`bg-edge animate-pulse ${className}`}
      style={{ borderRadius: 0, ...style }}
    />
  )
}

export function ResultsSkeleton() {
  return (
    <div className="space-y-4">
      {/* Metrics row */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {[...Array(4)].map((_, i) => (
          <div key={i} className="bg-surface border border-edge p-4 space-y-2">
            <Skeleton className="h-3 w-16" />
            <Skeleton className="h-7 w-24" />
          </div>
        ))}
      </div>

      {/* Chart skeleton */}
      <div className="bg-surface border border-edge p-4">
        <Skeleton className="h-3 w-48 mb-4" />
        <Skeleton className="h-[300px] w-full" />
      </div>

      {/* Bar chart skeleton */}
      <div className="bg-surface border border-edge p-4">
        <Skeleton className="h-3 w-40 mb-4" />
        <div className="space-y-2">
          {[...Array(5)].map((_, i) => (
            <Skeleton key={i} className="h-4 w-full" style={{ width: `${60 + Math.random() * 40}%` }} />
          ))}
        </div>
      </div>
    </div>
  )
}
