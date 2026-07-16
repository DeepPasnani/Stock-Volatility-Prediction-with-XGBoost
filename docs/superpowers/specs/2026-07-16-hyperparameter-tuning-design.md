# Hyperparameter Tuning — Design

## Scope

Add an `ml/tune.py` module with a grid-search hyperparameter tuning function, and a CLI entry point in `main.py`. This is sub-project 2 of the ML improvements roadmap.

## Module: `ml/tune.py`

### Function signature

```python
def tune_hyperparameters(
    ticker: str,
    start_date: str,
    end_date: str,
    param_grid: dict | None = None,
    cv_folds: int = 3,
    config: MlConfig | None = None,
) -> dict
```

### Behaviour

1. Loads data via `ml.data.load_data`, prepares and engineers features using the existing pipeline.
2. Runs `GridSearchCV` from scikit-learn with an `XGBRegressor` over `param_grid`.
3. Uses `KFold(cv_folds, shuffle=False)` for time-series appropriate cross-validation (no shuffle).
4. Scoring metric: `neg_root_mean_squared_error` (negative RMSE — higher is better for sklearn).
5. Returns dict with:
   - `best_params`: the winning param combination
   - `best_score`: CV score (RMSE)
   - `cv_results`: full results dataframe serialized as a list of dicts
   - `ticker`, `param_grid`, `cv_folds` for context

### Default param grid

```python
{
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}
```

Overrideable via `param_grid` argument.

### Error handling

- Same as pipeline: empty data → `ValueError`, insufficient rows → `ValueError`.
- Invalid params → let sklearn raise (clear error messages).

## CLI

`main.py` gets a new mode:

```
python main.py tune AAPL 2023-01-01 2024-06-01
python main.py tune AAPL 2023-01-01 2024-06-01 --folds 5
```

Prints best params and score to stdout.

## Files changed

| File | Action |
|------|--------|
| `ml/tune.py` | Create — tuning function |
| `ml/__init__.py` | Add `tune_hyperparameters` export |
| `ml/test_ml.py` | Add `test_tune_hyperparameters` |
| `main.py` | Add `tune` CLI branch |

## Non-goals

- No parallelization of grid search (YAGNI for this project size)
- No model persistence after tuning (tuning just reports best params — user plugs them into `MlConfig.model_params`)
- No integration with the React frontend in this pass (CLI-only)
- No Bayesian/advanced search methods — grid search is sufficient

## Edge cases

- Parameter grid with a single combination → single fit, still returns valid result
- `cv_folds=1` → falls back to simple train/test split
- Large grids + large data → can be slow (expected for grid search)
