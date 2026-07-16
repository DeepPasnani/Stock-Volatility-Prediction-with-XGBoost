# Frontend Enhancements — Design

## Scope

Improve the React frontend with multi-ticker comparison, form validation, real loading progress, saved searches, CSV download, and responsive polish — all within existing dependencies (no new npm packages).

## Changes by component

### `SearchForm.jsx`
- Accept comma-separated tickers (e.g. `AAPL,MSFT,GOOG`). Split on submit.
- Validate end date ≥ start + 60 days. Show inline error.
- On mount, restore last search from `localStorage` key `lastSearch`.
- On submit, save to `localStorage`; also maintain `recentSearches` in localStorage (last 5), shown as clickable chips below the form.

### `App.jsx`
- `usePredict` → `usePredictions` — handles multiple sequential API calls.
- State: `results[]` (one per ticker), `loading`, `currentIndex`, `error`.
- Progress text: `"Analyzing AAPL (1/3)..."`.
- On submit: clear results, call predict for each ticker sequentially, collect results.
- Display: **comparison table** (ticker, RMSE, R², rows), then **accordion** of individual charts.
- Download CSV button per ticker result.

### New component: `ComparisonTable.jsx`
- `sortable` columns: Ticker, RMSE, R², Train Rows, Test Rows.
- Click column header to sort ascending/descending.
- Highlight best score in green.

### New component: `ResultsAccordion.jsx`
- One expandable panel per ticker.
- Each panel contains PredictionChart + FeatureChart + MetricsPanel + Download CSV button.
- First result expanded by default.

### `LoadingState.jsx`
- Replace fake timeouts with real progress props: `current`, `total`, `ticker`.
- Show: `"Analyzing AAPL (1/3)..."`.

### CSV download
- Build CSV string from predictions array, trigger download via `Blob` + `URL.createObjectURL`.
- No library needed.

### `usePredictions.js` (new hook)
- Manages sequential fetch loop, collects results, exposes `results`, `loading`, `progress`, `error`, `run`.

### Files changed
| File | Action |
|------|--------|
| `src/hooks/usePredictions.js` | Create (replaces usePredict.js) |
| `src/components/SearchForm.jsx` | Modify — multi-ticker, validation, recent searches |
| `src/components/ComparisonTable.jsx` | Create |
| `src/components/ResultsAccordion.jsx` | Create |
| `src/components/LoadingState.jsx` | Modify — real progress |
| `src/App.jsx` | Modify — wire up new components |
| `src/hooks/usePredict.js` | Delete |

### Non-goals
- No backend changes (reuses existing `/api/predict`)
- No routing, no state management library
- No authentication
- No websockets or streaming

## Edge cases
- Empty ticker → show "Enter at least one ticker"
- Invalid ticker (API error) → show error per ticker, continue with others
- One ticker succeeds, another fails → show partial results + error for failed
- localStorage unavailable → gracefully degrade (no saved searches, no crash)
- CSV with no predictions → empty CSV with headers
