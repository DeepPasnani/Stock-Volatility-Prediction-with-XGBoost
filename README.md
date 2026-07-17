# 📈 Stock Volatility Prediction with XGBoost

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-19-61DAFB.svg)](https://react.dev)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-EC4E20.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Forecast short-term stock volatility using historical market data, technical indicators, and XGBoost regression.

---

## Overview

This project predicts stock volatility — specifically the standard deviation of daily returns — using a full ML pipeline that transforms raw Yahoo Finance data into lag features, rolling statistics, and technical indicators (RSI, MACD, Bollinger Bands). An XGBoost regressor is trained on the engineered features and evaluated on a held-out test set.

The repo ships with three interfaces:
- **Web UI** — React + Vite frontend with a dark terminal-inspired design
- **REST API** — FastAPI backend for programmatic access
- **CLI / Streamlit** — Lightweight local scripts for quick experimentation

---

## Table of Contents

- [Architecture](#architecture)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Results](#results)
- [License](#license)

---

## Architecture

```
┌──────────┐     ┌──────────────┐     ┌──────────────┐
│  React    │────▶│  FastAPI     │────▶│  ML Pipeline  │
│  Frontend │     │  Backend     │     │  (XGBoost)    │
│  :5173    │◀────│  :8000       │◀────│               │
└──────────┘     └──────────────┘     └──────┬───────┘
                                             │
                                     ┌───────▼───────┐
                                     │  Yahoo Finance │
                                     │  (yfinance)    │
                                     └───────────────┘
```

The frontend communicates with the backend via REST (`POST /api/predict`). In production, the built frontend is served directly by the FastAPI server on a single port. In development, Vite proxies `/api` calls to the backend.

Three independent entry points exist for running the ML pipeline itself:

| Entry Point | When to Use |
|---|---|
| **Web UI** (`npm run dev`) | Interactive analysis, charts, comparison tables |
| **REST API** (`uvicorn`) | Integrate with other tools or automate predictions |
| **CLI / Streamlit** (`python main.py`) | Quick runs without frontend infrastructure |

---

## Features

- **Data Collection** — Fetches historical stock data from Yahoo Finance via `yfinance`
- **Feature Engineering** — Computes lag features, rolling windows, returns, volatility, RSI, MACD, Bollinger Bands, and more
- **XGBoost Regression** — Trains a gradient-boosted model with configurable hyperparameters
- **Evaluation Metrics** — Reports RMSE and R² with actual-vs-predicted visualizations
- **Feature Importance** — Displays top-15 contributing features with bar charts
- **Multi-Ticker Comparison** — Compare multiple stocks side-by-side in the web UI
- **Hyperparameter Tuning** — Grid search with cross-validation for optimal model params
- **Dark Terminal UI** — Bloomberg-inspired design with cyan-on-black color scheme

---

## Tech Stack

### Frontend
| Library | Purpose |
|---|---|
| [React 19](https://react.dev/) | UI framework |
| [Vite 8](https://vitejs.dev/) | Build tool & dev server |
| [Tailwind CSS 4](https://tailwindcss.com/) | Utility-first styling |
| [Recharts](https://recharts.org/) | Charting library |

### Backend
| Library | Purpose |
|---|---|
| [FastAPI](https://fastapi.tiangolo.com/) | REST API framework |
| [Uvicorn](https://www.uvicorn.org/) | ASGI server |
| [Pydantic](https://docs.pydantic.dev/) | Request/response validation |

### ML Pipeline
| Library | Purpose |
|---|---|
| [XGBoost](https://xgboost.readthedocs.io/) | Gradient-boosted regression |
| [scikit-learn](https://scikit-learn.org/) | Train-test split, metrics |
| [pandas](https://pandas.pydata.org/) / [numpy](https://numpy.org/) | Data manipulation |
| [ta](https://technical-analysis-library-in-python.readthedocs.io/) | Technical indicators (RSI, MACD, Bollinger Bands) |
| [yfinance](https://pypi.org/project/yfinance/) | Yahoo Finance data downloader |
| [Streamlit](https://streamlit.io/) | Lightweight local UI |

---

## Project Structure

```
.
├── backend/                    # FastAPI web backend
│   ├── main.py                 # App entry point, CORS, static file mount
│   ├── schemas.py              # Pydantic request/response models
│   ├── routers/
│   │   └── predict.py          # POST /api/predict endpoint
│   ├── services/
│   │   └── model.py            # Calls ml.pipeline.run_prediction
│   └── requirements.txt
├── frontend/                   # React + Vite web frontend
│   ├── src/
│   │   ├── App.jsx             # Main app component
│   │   ├── components/         # UI components
│   │   │   ├── SearchForm.jsx       # Ticker + date inputs
│   │   │   ├── PredictionChart.jsx  # Actual vs predicted line chart
│   │   │   ├── FeatureChart.jsx     # Feature importance bar chart
│   │   │   ├── MetricsPanel.jsx     # RMSE / R² display
│   │   │   ├── ComparisonTable.jsx  # Multi-ticker comparison
│   │   │   ├── ResultsAccordion.jsx # Expandable per-ticker results
│   │   │   ├── LoadingState.jsx     # Terminal-style loading prompt
│   │   │   └── Skeleton.jsx         # Skeleton loading placeholder
│   │   └── hooks/
│   │       └── usePredictions.js    # API call hook
│   ├── public/                 # Static assets
│   ├── vite.config.js          # Vite config with API proxy
│   └── package.json
├── ml/                         # Core ML pipeline (importable package)
│   ├── __init__.py
│   ├── config.py               # MlConfig dataclass
│   ├── data.py                 # Download & prepare data
│   ├── features.py             # Technical indicators & feature engineering
│   ├── train.py                # Train XGBoost model
│   ├── pipeline.py             # End-to-end run_prediction()
│   ├── tune.py                 # Hyperparameter grid search
│   └── test_ml.py              # Self-contained correctness tests
├── main.py                     # CLI / Streamlit entry point
├── requirements.txt            # ML pipeline dependencies
├── docker-compose.yml          # Dev Docker setup
├── docker-compose.prod.yml     # Production Docker setup
├── Dockerfile                  # Multi-stage build (frontend + backend)
└── DESIGN.md                   # Full design system spec
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js 20+
- Docker & Docker Compose (optional)

### Docker (recommended — runs everything)

```bash
docker compose up --build
```

| Service | URL |
|---|---|
| Frontend | http://localhost:5173 |
| Backend API | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |

### Local Development

#### 1. Backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

#### 2. Frontend

```bash
cd frontend
npm install
npm run dev       # starts on :5173, proxies /api to :8000
```

#### 3. CLI / Streamlit

```bash
pip install -r requirements.txt
python main.py                                # CLI usage info
python main.py streamlit                      # Streamlit UI
python main.py tune AAPL 2020-01-01 2025-01-01  # Hyperparameter tuning
```

---

## Usage

### Web UI

1. Open http://localhost:5173
2. Enter a ticker symbol (e.g. `AAPL`), start date, and end date
3. Click **Analyze** to run the prediction
4. View prediction chart, feature importance, and metrics
5. Add more tickers for side-by-side comparison

### REST API

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "start_date": "2020-01-01", "end_date": "2025-01-01"}'
```

### CLI

```bash
# Interactive mode
python main.py

# Streamlit dashboard
python main.py streamlit

# Hyperparameter tuning
python main.py tune AAPL 2020-01-01 2025-01-01 --folds 5
```

---

## API Reference

### `POST /api/predict`

Predict stock volatility for a given ticker and date range.

**Request Body:**

```json
{
  "ticker": "AAPL",
  "start_date": "2020-01-01",
  "end_date": "2025-01-01"
}
```

**Response:**

```json
{
  "ticker": "AAPL",
  "rmse": 0.0152,
  "r2": 0.7843,
  "total_rows": 1258,
  "train_rows": 1006,
  "test_rows": 252,
  "predictions": [
    {"date": "2024-12-15", "actual": 0.0182, "predicted": 0.0167}
  ],
  "feature_importance": [
    {"feature": "volatility_5", "importance": 0.142},
    {"feature": "return_1", "importance": 0.113}
  ]
}
```

| Field | Type | Description |
|---|---|---|
| `ticker` | string | Stock ticker symbol |
| `rmse` | float | Root Mean Squared Error |
| `r2` | float | R² coefficient of determination |
| `total_rows` | int | Total data points after engineering |
| `train_rows` / `test_rows` | int | Train/test split sizes |
| `predictions` | array | Actual vs predicted volatility per date |
| `feature_importance` | array | Top-15 features by importance score |

### `GET /api/health`

```json
{ "status": "ok" }
```

### Interactive Docs

http://localhost:8000/docs — auto-generated OpenAPI docs with try-it-out.

---

## Configuration

ML pipeline parameters are configured in `ml/config.py` via the `MlConfig` dataclass:

| Parameter | Default | Description |
|---|---|---|
| `lag_windows` | `[1, 3, 5]` | Lookback periods for lag features |
| `rolling_windows` | `[5]` | Rolling mean window sizes |
| `return_horizons` | `[1, 3, 5]` | Periods for return calculations |
| `rsi_window` | `14` | RSI calculation period |
| `volatility_window` | `5` | Rolling volatility window |
| `test_size` | `0.2` | Fraction of data held out for testing |
| `model_params` | `{}` | XGBoost parameters (overrides defaults) |
| `random_state` | `42` | Seed for reproducibility |

Pass a custom config when calling from code:

```python
from ml import MlConfig, run_prediction

config = MlConfig(test_size=0.3, model_params={"n_estimators": 200})
result = run_prediction("AAPL", "2020-01-01", "2025-01-01", config)
```

---

## Results

The model predicts volatility (rolling standard deviation of daily returns) using engineered features derived from price data. Typical performance on S&P 500 stocks:

| Metric | Typical Range |
|---|---|
| **RMSE** | 0.008 – 0.025 |
| **R²** | 0.65 – 0.85 |

Outputs include:
- **Prediction chart** — Actual vs predicted volatility over the test period
- **Feature importance chart** — Top-15 features ranked by XGBoost importance score
- **Metrics panel** — RMSE and R² with color-coded scoring

---

## License

[MIT](LICENSE)
