# 📈 Stock Volatility Prediction with XGBoost

Predict short-term stock volatility using historical market data, engineered technical indicators, and an XGBoost regression model — wrapped in a full-stack web app (React + FastAPI) for interactive exploration.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![React](https://img.shields.io/badge/React-Vite-61DAFB)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![Docker](https://img.shields.io/badge/Deploy-Docker-2496ED)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🔹 Overview

This project forecasts stock price volatility by combining:

- **Historical market data** pulled live from Yahoo Finance
- **Feature engineering**: lag features, rolling statistics, returns, and technical indicators (RSI, MACD, Bollinger Bands)
- **XGBoost regression** to model and predict volatility
- A **React + FastAPI** web application so anyone can enter a ticker, run a prediction, and visualize the results — no notebook required

The repo also includes a Jupyter notebook (`main.ipynb`) documenting the original research/experimentation workflow, and a standalone `main.py` for running the pipeline outside the web app.

---

## ✨ Features

| Stage | Description |
|---|---|
| 📊 Data Collection | Fetches historical OHLCV stock data via `yfinance` |
| 🧹 Data Preparation | Cleans raw data and builds lag features |
| 🛠 Feature Engineering | Adds RSI, MACD, Bollinger Bands, rolling stats, returns, and volatility measures via the `ta` library |
| 🤖 Modeling | Trains an `XGBoost` Regressor to predict volatility |
| 📉 Evaluation | Reports RMSE and R², plots predicted vs. actual values |
| 🌐 Web App | React frontend + FastAPI backend for interactive ticker input, prediction, and visualization |

---

## 🛠 Tech Stack

**Frontend**
- React (Vite)
- Tailwind CSS
- Recharts (charts/visualizations)

**Backend**
- Python, FastAPI, Uvicorn
- `yfinance` — market data retrieval
- `pandas`, `numpy` — data manipulation
- `ta` — technical indicators (RSI, MACD, Bollinger Bands)
- `scikit-learn` — train/test split, evaluation metrics
- `xgboost` — the prediction model

**Infra**
- Docker & Docker Compose

---

## 📁 Project Structure

```
Stock-Volatility-Prediction-with-XGBoost/
├── backend/              # FastAPI service — data pipeline, model, API endpoints
├── frontend/             # React (Vite) web app
├── ml/                   # Model training / feature engineering code
├── main.ipynb            # Exploratory notebook (research & experimentation)
├── main.py                # Standalone pipeline script
├── requirements.txt        # Python dependencies
├── Dockerfile
├── docker-compose.yml     # Local development
├── docker-compose.prod.yml   # Production deployment
└── DESIGN.md              # Architecture / design notes
```

---

## ▶️ Quick Start (Docker — recommended)

```bash
git clone https://github.com/DeepPasnani/Stock-Volatility-Prediction-with-XGBoost.git
cd Stock-Volatility-Prediction-with-XGBoost
docker-compose up --build
```

Once running:

| Service | URL |
|---|---|
| Frontend | http://localhost:5173 |
| Backend API | http://localhost:8000 |
| API Docs (Swagger) | http://localhost:8000/docs |

To stop the app: `docker-compose down`

---

## 💻 Local Development (without Docker)

**Backend**

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

**Frontend**

```bash
cd frontend
npm install
npm run dev
```

**Standalone pipeline / notebook** (optional, for research use)

```bash
pip install -r requirements.txt
python main.py
# or open main.ipynb in Jupyter
```

---

## 🚀 Deployment

The repo ships with a production Compose file for deploying both services together:

```bash
docker-compose -f docker-compose.prod.yml up --build -d
```

General deployment notes:

- **Containerized deploy (recommended)**: build the image with the provided `Dockerfile` and push it to any container host (Render, Railway, Fly.io, AWS ECS/App Runner, Azure Container Apps, GCP Cloud Run).
- **Backend**: deploy the FastAPI app behind Uvicorn/Gunicorn; expose port `8000` (or whatever port your host assigns) and set it as the frontend's API base URL.
- **Frontend**: build with `npm run build` and serve the static output (`frontend/dist`) via any static host (Vercel, Netlify, Cloudflare Pages) or through Nginx in the same container.
- **Environment variables**: if you split frontend/backend across different hosts, set the backend URL as an environment variable in the frontend build (e.g. `VITE_API_BASE_URL`) so the app knows where to send requests.

> ⚠️ Adjust the exact env var names/ports to match what's configured in `backend/` and `frontend/` — check those folders' config files before deploying.

---

## 📊 Results

Model performance is evaluated using:

- **RMSE** (Root Mean Squared Error)
- **R² Score**

The app/notebook outputs:

- Line charts comparing actual vs. predicted volatility
- Feature importance bar charts

---

## 🔮 Future Improvements

- Hyperparameter tuning for XGBoost
- Deep learning models (LSTM, Transformers) for comparison
- Portfolio-level analysis in the frontend
- Sentiment analysis (news & social media) as additional features

---

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request with improvements, bug fixes, or new features.

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details (add one if not already present).

## 🙋 Author

**Deep Pasnani** — [GitHub](https://github.com/DeepPasnani)
