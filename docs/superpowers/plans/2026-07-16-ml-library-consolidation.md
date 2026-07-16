# ML Library Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) for syntax tracking.

**Goal:** Consolidate three drifted ML pipeline implementations into one shared `ml/` library with a config object, tests, and thin entry points.

**Architecture:** A flat `ml/` package with `config.py` for settings, `data.py` for data loading/cleaning, `features.py` for feature engineering, `train.py` for model training, and `pipeline.py` as the orchestrator. Existing `main.py`, `backend/services/model.py`, and `main.ipynb` become thin consumers.

**Tech Stack:** Python 3.12+, pandas, numpy, xgboost, scikit-learn, yfinance, ta

## Global Constraints

- All function signatures must match the existing backend API exactly (return dicts with `predictions`, `feature_importance`, `rmse`, `r2`, etc.)
- MlConfig defaults must preserve current behavior exactly
- No new dependencies
- Tests are plain assert-based (no pytest framework)
- Notebook must import from `ml/` and still run

---

### Task 1: MlConfig dataclass

**Files:**
- Create: `ml/__init__.py`
- Create: `ml/config.py`
- Test: `ml/test_ml.py` (test_config_defaults)

**Interfaces:**
- Consumes: nothing
- Produces: `ml.config.MlConfig` dataclass; `ml.__init__` exports `MlConfig`

- [ ] **Step 1: Create `ml/config.py`**

```python
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MlConfig:
    lag_windows: list[int] = field(default_factory=lambda: [1, 3, 5])
    lag_columns: list[str] = field(default_factory=lambda: ["Open", "High", "Low", "Close", "Volume"])
    rolling_windows: list[int] = field(default_factory=lambda: [5])
    return_horizons: list[int] = field(default_factory=lambda: [1, 3, 5])
    rsi_window: int = 14
    volatility_window: int = 5
    test_size: float = 0.2
    shuffle: bool = False
    model_params: dict = field(default_factory=dict)
    random_state: int = 42
```

- [ ] **Step 2: Create `ml/__init__.py`**

```python
from .config import MlConfig
from .pipeline import run_prediction, load_data

__all__ = ["MlConfig", "run_prediction", "load_data"]
```

- [ ] **Step 3: Write test in `ml/test_ml.py`**

```python
from ml.config import MlConfig


def test_config_defaults():
    cfg = MlConfig()
    assert cfg.lag_windows == [1, 3, 5]
    assert cfg.lag_columns == ["Open", "High", "Low", "Close", "Volume"]
    assert cfg.rolling_windows == [5]
    assert cfg.return_horizons == [1, 3, 5]
    assert cfg.rsi_window == 14
    assert cfg.volatility_window == 5
    assert cfg.test_size == 0.2
    assert cfg.shuffle is False
    assert cfg.model_params == {}
    assert cfg.random_state == 42


if __name__ == "__main__":
    test_config_defaults()
    print("Config tests passed")
```

- [ ] **Step 4: Run tests**

Run: `cd /home/deepp/Deep-Files/Projects/Github/Project Repos/Stock-Volatility-Prediction-with-XGBoost-main && python -m ml.test_ml`
Expected: "Config tests passed"

- [ ] **Step 5: Commit**

```bash
git add ml/config.py ml/__init__.py ml/test_ml.py
git commit -m "feat: add MlConfig dataclass and package init"
```

---

### Task 2: Data loading and preparation

**Files:**
- Create: `ml/data.py`
- Modify: `ml/test_ml.py` (append tests)

**Interfaces:**
- Consumes: `MlConfig` from `ml.config`
- Produces: `ml.data.load_data(ticker, start, end) -> pd.DataFrame`, `ml.data.prepare_data(data, config) -> pd.DataFrame`, `ml.data.add_lagged_features(data, config) -> pd.DataFrame`

- [ ] **Step 1: Create `ml/data.py`**

```python
import yfinance as yf
import pandas as pd
from .config import MlConfig


def load_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    start = str(start)
    end = str(end)
    data = yf.download(ticker, start=start, end=end)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data


def prepare_data(data: pd.DataFrame, config: MlConfig = MlConfig()) -> pd.DataFrame:
    data = data.dropna()
    data = add_lagged_features(data, config)
    return data


def add_lagged_features(data: pd.DataFrame, config: MlConfig = MlConfig()) -> pd.DataFrame:
    for col in config.lag_columns:
        for lag in config.lag_windows:
            data[f'{col}_lag{lag}'] = data[col].squeeze().shift(lag)
    return data
```

- [ ] **Step 2: Append data tests to `ml/test_ml.py`**

```python
def test_prepare_data():
    import pandas as pd
    from ml.data import prepare_data

    dates = pd.date_range("2020-01-01", periods=20, freq="D")
    data = pd.DataFrame({
        "Open": range(20),
        "High": range(1, 21),
        "Low": range(20),
        "Close": range(20),
        "Volume": [100] * 20,
    }, index=dates)

    result = prepare_data(data.copy())
    assert "Close_lag1" in result.columns
    assert "Close_lag3" in result.columns
    assert "Close_lag5" in result.columns
    assert "Open_lag1" in result.columns
    assert "Volume_lag5" in result.columns
    assert result["Close_lag1"].iloc[5] == 4.0  # row 5 value is 5, lag1 is row 4 = 4
```

- [ ] **Step 3: Run tests**

Run: `cd /home/deepp/Deep-Files/Projects/Github/Project Repos/Stock-Volatility-Prediction-with-XGBoost-main && python -m ml.test_ml`
Expected: Both "Config tests passed" line and no assertion failures

- [ ] **Step 4: Commit**

```bash
git add ml/data.py ml/test_ml.py
git commit -m "feat: add data loading and lag feature functions"
```

---

### Task 3: Feature engineering

**Files:**
- Create: `ml/features.py`
- Modify: `ml/test_ml.py` (append test_engineer_features)

**Interfaces:**
- Consumes: `MlConfig` from `ml.config`
- Produces: `ml.features.engineer_features(data, config) -> pd.DataFrame`

- [ ] **Step 1: Create `ml/features.py`**

```python
import pandas as pd
import ta
from .config import MlConfig


def engineer_features(data: pd.DataFrame, config: MlConfig = MlConfig()) -> pd.DataFrame:
    close = data['Close'].squeeze()
    high = data['High'].squeeze()
    low = data['Low'].squeeze()

    # Rolling statistics
    for w in config.rolling_windows:
        data[f'rolling_mean_{w}'] = close.rolling(w).mean()
        data[f'rolling_std_{w}'] = close.rolling(w).std()

    # Returns
    for h in config.return_horizons:
        data[f'return_{h}'] = close.pct_change(h)

    # RSI
    data['RSI'] = ta.momentum.RSIIndicator(close, window=config.rsi_window).rsi()

    # MACD
    macd = ta.trend.MACD(close)
    data['MACD'] = macd.macd()
    data['MACD_signal'] = macd.macd_signal()
    data['MACD_diff'] = macd.macd_diff()

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(close)
    data['BB_high'] = bb.bollinger_hband()
    data['BB_low'] = bb.bollinger_lband()
    data['BB_mid'] = bb.bollinger_mavg()
    data['BB_width'] = bb.bollinger_wband()

    # Target: volatility (5-day rolling std of returns, shifted back 1)
    data['volatility'] = data['return_1'].rolling(config.volatility_window).std().shift(-1)

    data = data.dropna()
    return data
```

- [ ] **Step 2: Append feature test to `ml/test_ml.py`**

```python
def test_engineer_features():
    import pandas as pd
    import numpy as np
    from ml.data import prepare_data
    from ml.features import engineer_features

    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=50, freq="D")
    data = pd.DataFrame({
        "Open": np.random.randn(50) + 100,
        "High": np.random.randn(50) + 101,
        "Low": np.random.randn(50) + 99,
        "Close": np.random.randn(50) + 100,
        "Volume": np.random.randint(100000, 500000, 50),
    }, index=dates)

    prepared = prepare_data(data)
    result = engineer_features(prepared)

    assert "RSI" in result.columns
    assert "MACD" in result.columns
    assert "MACD_signal" in result.columns
    assert "MACD_diff" in result.columns
    assert "BB_high" in result.columns
    assert "BB_low" in result.columns
    assert "BB_mid" in result.columns
    assert "BB_width" in result.columns
    assert "rolling_mean_5" in result.columns
    assert "rolling_std_5" in result.columns
    assert "return_1" in result.columns
    assert "return_3" in result.columns
    assert "return_5" in result.columns
    assert "volatility" in result.columns
    assert len(result) > 0
```

- [ ] **Step 3: Run tests**

Run: `cd /home/deepp/Deep-Files/Projects/Github/Project Repos/Stock-Volatility-Prediction-with-XGBoost-main && python -m ml.test_ml`
Expected: All tests pass silently

- [ ] **Step 4: Commit**

```bash
git add ml/features.py ml/test_ml.py
git commit -m "feat: add feature engineering module"
```

---

### Task 4: Model training

**Files:**
- Create: `ml/train.py`
- Modify: `ml/test_ml.py` (append train tests)

**Interfaces:**
- Consumes: `MlConfig` from `ml.config`; expects DataFrame with `volatility` column
- Produces: `ml.train.split_data(data, config) -> (X_train, X_test, y_train, y_test)`, `ml.train.train_model(X_train, y_train, config) -> XGBRegressor`

- [ ] **Step 1: Create `ml/train.py`**

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from typing import Tuple
from .config import MlConfig


def split_data(data: pd.DataFrame, config: MlConfig = MlConfig()) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = data.drop(['volatility'], axis=1)
    y = data['volatility']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, shuffle=config.shuffle
    )
    return X_train, X_test, y_train, y_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series, config: MlConfig = MlConfig()) -> XGBRegressor:
    model = XGBRegressor(random_state=config.random_state, **config.model_params)
    model.fit(X_train, y_train)
    return model
```

- [ ] **Step 2: Append train tests to `ml/test_ml.py`**

```python
def test_split_data():
    import pandas as pd
    import numpy as np
    from ml.train import split_data

    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    data = pd.DataFrame({
        "Close": np.random.randn(100) + 100,
        "volatility": np.random.rand(100) * 0.1,
    }, index=dates)

    X_train, X_test, y_train, y_test = split_data(data)
    assert len(X_train) == 80
    assert len(X_test) == 20
    assert "volatility" not in X_train.columns


def test_train_model():
    import pandas as pd
    import numpy as np
    from ml.train import train_model, split_data

    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    data = pd.DataFrame({
        "Close": np.random.randn(100) + 100,
        "return_1": np.random.randn(100) * 0.01,
        "volatility": np.random.rand(100) * 0.1,
    }, index=dates)

    X_train, X_test, y_train, y_test = split_data(data)
    model = train_model(X_train, y_train)
    y_pred = model.predict(X_test)
    assert len(y_pred) == 20
    assert all(np.isfinite(y_pred))
```

- [ ] **Step 3: Run tests**

Run: `cd /home/deepp/Deep-Files/Projects/Github/Project Repos/Stock-Volatility-Prediction-with-XGBoost-main && python -m ml.test_ml`
Expected: All tests pass

- [ ] **Step 4: Commit**

```bash
git add ml/train.py ml/test_ml.py
git commit -m "feat: add model training and data splitting module"
```

---

### Task 5: Pipeline orchestrator

**Files:**
- Create: `ml/pipeline.py`
- Modify: `ml/__init__.py` (update imports)
- Modify: `ml/test_ml.py` (append integration test)

**Interfaces:**
- Consumes: `MlConfig`, `load_data`, `prepare_data`, `engineer_features`, `split_data`, `train_model`, `mean_squared_error`, `r2_score`
- Produces: `ml.pipeline.run_prediction(ticker, start_date, end_date, config) -> dict`

- [ ] **Step 1: Create `ml/pipeline.py`**

```python
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from .config import MlConfig
from .data import load_data, prepare_data
from .features import engineer_features
from .train import split_data, train_model


def run_prediction(
    ticker: str,
    start_date: str,
    end_date: str,
    config: MlConfig = MlConfig(),
) -> dict:
    data = load_data(ticker, start_date, end_date)

    if data.empty:
        raise ValueError("No data found for the given ticker and date range.")

    data = prepare_data(data, config)
    data = engineer_features(data, config)

    if data.empty or len(data) < 30:
        raise ValueError("Not enough data after cleaning and feature engineering. Try a wider date range.")

    X_train, X_test, y_train, y_test = split_data(data, config)

    train_rows = len(X_train)
    test_rows = len(X_test)
    total_rows = len(data)

    model = train_model(X_train, y_train, config)
    y_pred = model.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))

    predictions = []
    for i, (date, actual) in enumerate(zip(y_test.index, y_test)):
        predictions.append({
            "date": str(date.date()) if hasattr(date, 'date') else str(date),
            "actual": float(actual),
            "predicted": float(y_pred[i]),
        })

    feature_importance = []
    feat_imp_series = pd.Series(model.feature_importances_, index=X_train.columns)
    feat_imp_series = feat_imp_series.sort_values(ascending=False).head(15)
    for feature, importance in feat_imp_series.items():
        feature_importance.append({
            "feature": feature,
            "importance": float(importance),
        })

    return {
        "ticker": ticker,
        "rmse": rmse,
        "r2": r2,
        "predictions": predictions,
        "feature_importance": feature_importance,
        "total_rows": total_rows,
        "train_rows": train_rows,
        "test_rows": test_rows,
    }
```

- [ ] **Step 2: Update `ml/__init__.py`**

```python
from .config import MlConfig
from .pipeline import run_prediction
from .data import load_data

__all__ = ["MlConfig", "run_prediction", "load_data"]
```

- [ ] **Step 3: Append pipeline test to `ml/test_ml.py`**

```python
def test_pipeline_response_shape():
    """Test that run_prediction returns the expected dict shape with real data."""
    from ml.pipeline import run_prediction
    from ml.config import MlConfig

    # Use a small config with min data to speed up test
    result = run_prediction("AAPL", "2024-01-01", "2024-03-01")

    assert isinstance(result, dict)
    assert result["ticker"] == "AAPL"
    assert "rmse" in result
    assert "r2" in result
    assert "predictions" in result
    assert "feature_importance" in result
    assert "total_rows" in result
    assert "train_rows" in result
    assert "test_rows" in result
    assert result["r2"] is not None

    if result["predictions"]:
        first = result["predictions"][0]
        assert "date" in first
        assert "actual" in first
        assert "predicted" in first

    if result["feature_importance"]:
        first = result["feature_importance"][0]
        assert "feature" in first
        assert "importance" in first
```

- [ ] **Step 4: Add a `__main__` runner to `ml/test_ml.py`**

Append at bottom of `ml/test_ml.py`:

```python
if __name__ == "__main__":
    test_config_defaults()
    test_prepare_data()
    test_engineer_features()
    test_split_data()
    test_train_model()
    test_pipeline_response_shape()
    print("All tests passed!")
```

- [ ] **Step 5: Run tests**

Run: `cd /home/deepp/Deep-Files/Projects/Github/Project Repos/Stock-Volatility-Prediction-with-XGBoost-main && python -m ml.test_ml`
Expected: "All tests passed!" (pipeline test calls real yfinance, may take a few seconds)

- [ ] **Step 6: Commit**

```bash
git add ml/pipeline.py ml/__init__.py ml/test_ml.py
git commit -m "feat: add pipeline orchestrator with public API"
```

---

### Task 6: Clean up `main.py`

**Files:**
- Modify: `main.py`

**Interfaces:**
- Consumes: `ml.run_prediction`, `ml.MlConfig`, `ml.load_data`

- [ ] **Step 1: Replace `main.py` body with thin entry point wrappers**

```python
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import streamlit as st
from ml import run_prediction, MlConfig


def run_streamlit():
    st.title("Stock Volatility Prediction App")
    ticker = st.text_input("Ticker Symbol (e.g. AAPL)", "AAPL")
    start = st.date_input("Start Date", pd.to_datetime("2015-01-01"))
    end = st.date_input("End Date", pd.to_datetime("2025-08-13"))

    if st.button("Predict Volatility"):
        with st.spinner("Downloading and processing data..."):
            try:
                result = run_prediction(
                    ticker,
                    start.strftime("%Y-%m-%d"),
                    end.strftime("%Y-%m-%d"),
                )
            except ValueError as e:
                st.error(str(e))
                return

        preds = result["predictions"]
        chart_df = pd.DataFrame(
            {"Actual": [p["actual"] for p in preds], "Predicted": [p["predicted"] for p in preds]},
            index=[p["date"] for p in preds],
        )
        st.line_chart(chart_df)
        st.write(f"RMSE: {result['rmse']:.4f}")
        st.write(f"R²: {result['r2']:.4f}")

        st.subheader("Feature Importance")
        feat_df = pd.DataFrame(result["feature_importance"])
        if not feat_df.empty:
            feat_df = feat_df.set_index("feature")
            st.bar_chart(feat_df)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "streamlit":
        run_streamlit()
    else:
        print("Enter stock ticker, start date (YYYY-MM-DD), and end date (YYYY-MM-DD):")
        ticker = input("Ticker: ").strip().upper()
        start = input("Start Date (YYYY-MM-DD): ").strip()
        end = input("End Date (YYYY-MM-DD): ").strip()
        try:
            result = run_prediction(ticker, start, end)
            print(f"Ticker: {result['ticker']}")
            print(f"RMSE: {result['rmse']:.4f}")
            print(f"R²: {result['r2']:.4f}")
            print(f"Train rows: {result['train_rows']}")
            print(f"Test rows: {result['test_rows']}")
            print(f"Total rows: {result['total_rows']}")
        except ValueError as e:
            print(f"Error: {e}")
```

- [ ] **Step 2: Run CLI test**

Run: `echo -e "AAPL\n2024-01-01\n2024-03-01" | python main.py`
Expected: Prints RMSE, R², row counts, no errors

- [ ] **Step 3: Commit**

```bash
git add main.py
git commit -m "refactor: strip main.py to thin entry points using ml library"
```

---

### Task 7: Clean up backend

**Files:**
- Modify: `backend/services/model.py`

- [ ] **Step 1: Replace `backend/services/model.py`**

```python
from ml import run_prediction, MlConfig

__all__ = ["run_prediction", "MlConfig"]
```

- [ ] **Step 2: Verify backend API imports work**

Run: `cd /home/deepp/Deep-Files/Projects/Github/Project Repos/Stock-Volatility-Prediction-with-XGBoost-main/backend && python -c "from services.model import run_prediction; print('OK')"`
Expected: "OK"

- [ ] **Step 3: Commit**

```bash
git add backend/services/model.py
git commit -m "refactor: backend model service re-exports from ml library"
```

---

### Task 8: Clean up notebook

**Files:**
- Modify: `main.ipynb`

- [ ] **Step 1: Update notebook imports**

The notebook currently redefines all functions. Replace its code cells so the first cell does:

```python
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ml import run_prediction, MlConfig
from ml.data import load_data, prepare_data
from ml.features import engineer_features
from ml.train import split_data, train_model
```

Then remove the duplicate function definition cells (2-6), keeping only the exploration/visualization cells that use the imported functions. The last cell (`run_streamlit()` call) is deleted since that's now in `main.py`.

The notebook should become:
- Cell 1: imports (including `from ml import ...`)
- Cell 2: demonstration usage — call `run_prediction("AAPL", "2024-01-01", "2024-03-01")` and show results
- Cell 3: optional custom config demo

**Note:** Jupyter notebooks are JSON. Use `edit` on `main.ipynb` to update the source in cell 1. Remove cells 2-6 (the function definition cells) and the final `run_streamlit()` call cell by updating the `cells` array length and content.

Simpler approach: just replace the whole notebook content.

- [ ] **Step 2: Verify notebook loads**

Run: `cd /home/deepp/Deep-Files/Projects/Github/Project Repos/Stock-Volatility-Prediction-with-XGBoost-main && python -c "import json; nb = json.load(open('main.ipynb')); print(f'{len(nb[\"cells\"])} cells')"` 
Expected: 3-4 cells, no errors

- [ ] **Step 3: Commit**

```bash
git add main.ipynb
git commit -m "refactor: update notebook to import from ml library"
```
