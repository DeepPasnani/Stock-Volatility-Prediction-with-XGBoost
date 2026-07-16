# ML Library Consolidation — Design

## Scope

Consolidate the three drifted ML pipeline implementations (`main.py`, `backend/services/model.py`, `main.ipynb`) into a single shared `ml/` library. Remove duplication, fix the notebook, add tests, and prepare the codebase for sub-project 2 (hyperparameter tuning).

## Module Layout

```
ml/
  __init__.py          # public exports: run_prediction, load_data, MlConfig
  config.py            # MlConfig dataclass
  data.py              # load_data, prepare_data, add_lagged_features
  features.py          # engineer_features
  train.py             # split_data, train_model
  pipeline.py          # run_prediction orchestrator
  test_ml.py           # tests (plain assert-based, no framework)
```

Each module lives in exactly one place. The backend, CLI, Streamlit app, and notebook all import from `ml/`.

## MlConfig

A single `@dataclass` with sensible defaults that match current behavior:

- `lag_windows`: `[1, 3, 5]` — shift lags for price/volume columns
- `lag_columns`: `["Open", "High", "Low", "Close", "Volume"]`
- `rolling_windows`: `[5]` — rolling mean/std windows
- `return_horizons`: `[1, 3, 5]` — pct_change horizons
- `rsi_window`: `14`
- `volatility_window`: `5`
- `test_size`: `0.2`
- `shuffle`: `False`
- `model_params`: `dict` — passed as `**kwargs` to `XGBRegressor()`
- `random_state`: `42`

Overridable per-call. Sub-project 2 (tuning) passes `model_params={"n_estimators": 300, ...}`.

## Data Flow

1. `load_data(ticker, start, end)` → `data.py` — fetches from yfinance, flattens MultiIndex columns
2. `prepare_data(data, config)` → `data.py` — drops NaNs, adds lag features
3. `engineer_features(data, config)` → `features.py` — rolling stats, returns, RSI, MACD, Bollinger Bands, volatility target
4. `split_data(data, config)` → `train.py` — separates X/y, train/test split
5. `train_model(X_train, y_train, config)` → `train.py` — fits XGBRegressor with `config.model_params`
6. `run_prediction(ticker, start, end, config)` → `pipeline.py` — orchestrates 1-5, evaluates, builds response dict

## Files Changed

| File | Action |
|------|--------|
| `ml/` (new directory) | Create 7 files |
| `main.py` | Strip to thin CLI/Streamlit entry points importing from `ml/`. Remove duplicated fn definitions. |
| `backend/services/model.py` | Replace body with `from ml import run_prediction; run_prediction = run_prediction` or just re-export. |
| `main.ipynb` | Update imports to use `ml` package. Remove duplicated fn definitions. |
| `requirements.txt` | No changes needed (all deps already listed). |

## Testing

`ml/test_ml.py` — plain assert-based self-check, no framework:

- `test_load_data()` — mock or skip (calls yfinance), test that flat columns work
- `test_prepare_data()` — small DataFrame with known shape, verify lag columns added
- `test_engineer_features()` — small DataFrame, verify expected columns exist
- `test_split_data()` — verify split sizes and non-shuffled ordering
- `test_train_model()` — tiny dataset, verify model fits and returns predictions
- `test_run_prediction()` — minimal integration test
- `test_config_defaults()` — verify MlConfig() has expected defaults

## Edge Cases

- Empty data from yfinance → raises `ValueError` with message
- < 30 rows after engineering → raises `ValueError` with message
- MultiIndex columns → flattened to single level (existing fix in backend, now shared)
- Invalid date format → FastAPI layer rejects before hitting ML lib
- CLI mode with no args → shows usage

## Non-Goals

- No model caching or persistence (YAGNI until needed)
- No async pipeline (model runs synchronously, fine for current use)
- No separate config files (`MlConfig` dataclass is sufficient)
- No CI/CD setup (deferred to sub-project 4)

## Next Steps

After spec approval and commit → invoke writing-plans to create implementation plan.
