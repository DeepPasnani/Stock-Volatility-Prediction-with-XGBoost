# Hyperparameter Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) for syntax tracking.

**Goal:** Add a grid-search hyperparameter tuning function and CLI entry point.

**Architecture:** New `ml/tune.py` module uses `GridSearchCV` with `KFold` cross-validation. Reuses existing data loading/feature engineering from the `ml/` library. Tuning results are returned as a dict (best params, best score, CV results). CLI entry point in `main.py` prints findings.

**Tech Stack:** Python 3.12+, scikit-learn (`GridSearchCV`, `KFold`), xgboost (`XGBRegressor`)

## Global Constraints

- Use `GridSearchCV` with `KFold(cv_folds, shuffle=False)` (time-series appropriate)
- Scoring: `neg_root_mean_squared_error`
- Default param grid: `n_estimators=[100,200,300]`, `max_depth=[3,5,7]`, `learning_rate=[0.01,0.05,0.1]`, `subsample=[0.8,1.0]`, `colsample_bytree=[0.8,1.0]`
- Return dict with `best_params`, `best_score`, `cv_results`, `ticker`, `param_grid`, `cv_folds`
- CLI: `python main.py tune TICKER START END [--folds N]`

---

### Task 1: Tuning module

**Files:**
- Create: `ml/tune.py`
- Modify: `ml/__init__.py`
- Modify: `ml/test_ml.py`

**Interfaces:**
- Consumes: `MlConfig`, `load_data`, `prepare_data`, `engineer_features` from `ml`
- Produces: `ml.tune.tune_hyperparameters(ticker, start, end, param_grid, cv_folds, config) -> dict`

- [ ] **Step 1: Create `ml/tune.py`**

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, KFold
from xgboost import XGBRegressor
from .config import MlConfig
from .data import load_data, prepare_data
from .features import engineer_features


DEFAULT_PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}


def tune_hyperparameters(
    ticker: str,
    start_date: str,
    end_date: str,
    param_grid: dict | None = None,
    cv_folds: int = 3,
    config: MlConfig | None = None,
) -> dict:
    if config is None:
        config = MlConfig()
    if param_grid is None:
        param_grid = DEFAULT_PARAM_GRID

    data = load_data(ticker, start_date, end_date)
    if data.empty:
        raise ValueError("No data found for the given ticker and date range.")

    data = prepare_data(data, config)
    data = engineer_features(data, config)
    if data.empty or len(data) < 30:
        raise ValueError("Not enough data after cleaning and feature engineering. Try a wider date range.")

    X = data.drop(["volatility"], axis=1)
    y = data["volatility"]

    cv = KFold(n_splits=cv_folds, shuffle=False)
    model = XGBRegressor(random_state=config.random_state)
    gs = GridSearchCV(
        model,
        param_grid,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=0,
    )
    gs.fit(X, y)

    cv_results = []
    for i in range(len(gs.cv_results_["params"])):
        cv_results.append({
            "params": gs.cv_results_["params"][i],
            "mean_test_score": float(-gs.cv_results_["mean_test_score"][i]),
            "std_test_score": float(gs.cv_results_["std_test_score"][i]),
        })

    return {
        "ticker": ticker,
        "best_params": gs.best_params_,
        "best_score": float(-gs.best_score_),
        "cv_results": cv_results,
        "param_grid": {k: list(v) for k, v in param_grid.items()},
        "cv_folds": cv_folds,
        "total_rows": len(data),
    }
```

- [ ] **Step 2: Update `ml/__init__.py`**

```python
from .config import MlConfig
from .pipeline import run_prediction
from .data import load_data
from .tune import tune_hyperparameters

__all__ = ["MlConfig", "run_prediction", "load_data", "tune_hyperparameters"]
```

- [ ] **Step 3: Append test to `ml/test_ml.py`**

```python
def test_tune_hyperparameters():
    from ml.tune import tune_hyperparameters

    # Minimal grid for speed — 2 combinations
    small_grid = {
        "n_estimators": [50],
        "max_depth": [3],
        "learning_rate": [0.1],
    }
    result = tune_hyperparameters("AAPL", "2023-01-01", "2024-06-01", param_grid=small_grid, cv_folds=2)

    assert isinstance(result, dict)
    assert "best_params" in result
    assert "best_score" in result
    assert "cv_results" in result
    assert "ticker" in result
    assert result["ticker"] == "AAPL"
    assert result["best_score"] > 0
    assert len(result["cv_results"]) == 1
    assert "n_estimators" in result["best_params"]
```

- [ ] **Step 4: Update `__main__` runner in `ml/test_ml.py`**

```python
if __name__ == "__main__":
    test_config_defaults()
    test_prepare_data()
    test_engineer_features()
    test_split_data()
    test_train_model()
    test_pipeline_response_shape()
    test_tune_hyperparameters()
    print("All tests passed")
```

- [ ] **Step 5: Run tests**

Run: `cd /home/deepp/Deep-Files/Projects/Github/Project Repos/Stock-Volatility-Prediction-with-XGBoost-main && .venv/bin/python -m ml.test_ml`
Expected: "All tests passed" (tuning test calls yfinance + grid search, may take ~10s)

- [ ] **Step 6: Commit**

```bash
git add ml/tune.py ml/__init__.py ml/test_ml.py
git commit -m "feat: add hyperparameter tuning module with grid search"
```

---

### Task 2: CLI entry point

**Files:**
- Modify: `main.py`

- [ ] **Step 1: Add tune branch to `main.py`**

Replace the `if __name__ == "__main__"` block with this expanded version:

```python
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "streamlit":
        run_streamlit()
    elif len(sys.argv) > 1 and sys.argv[1] == "tune":
        from ml.tune import tune_hyperparameters

        if len(sys.argv) < 5:
            print("Usage: python main.py tune TICKER START_DATE END_DATE [--folds N]")
            sys.exit(1)

        ticker = sys.argv[2].upper()
        start = sys.argv[3]
        end = sys.argv[4]
        folds = 3
        if "--folds" in sys.argv:
            idx = sys.argv.index("--folds")
            if idx + 1 < len(sys.argv):
                folds = int(sys.argv[idx + 1])

        try:
            result = tune_hyperparameters(ticker, start, end, cv_folds=folds)
            print(f"\nTuning results for {ticker}:")
            print(f"  Best CV score (RMSE): {result['best_score']:.6f}")
            print(f"  Best params: {result['best_params']}")
            print(f"  CV folds: {result['cv_folds']}")
            print(f"  Total rows: {result['total_rows']}")
            print(f"  Combinations tested: {len(result['cv_results'])}")
            print("\nAll results:")
            for r in result["cv_results"]:
                print(f"    {r['params']}  RMSE={r['mean_test_score']:.6f} ±{r['std_test_score']:.6f}")
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        print("Usage:")
        print("  python main.py                    # CLI mode (interactive)")
        print("  python main.py streamlit          # Streamlit UI")
        print("  python main.py tune TICKER START END [--folds N]  # Hyperparameter tuning")
        sys.exit(1)
```

Also add `import sys` (already there) — verify it's present at top of file.

- [ ] **Step 2: Test CLI tuning**

Run: `.venv/bin/python main.py tune AAPL 2023-01-01 2024-06-01 --folds 2`
Expected: Prints best params and RMSE, exits 0

- [ ] **Step 3: Run full test suite**

Run: `.venv/bin/python -m ml.test_ml`
Expected: All 7 tests pass

- [ ] **Step 4: Commit**

```bash
git add main.py
git commit -m "feat: add tune CLI command for hyperparameter search"
```
