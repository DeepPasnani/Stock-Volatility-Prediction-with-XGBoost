# 📈 Stock-Volatility-Prediction-with-XGBoost

🔹 Overview

This project predicts stock volatility using historical market data and machine learning techniques.
It combines financial technical indicators, feature engineering, and XGBoost regression models to forecast short-term volatility.

A full-stack web application with React frontend and FastAPI backend is included for interactive use.



🚀 Features

1. 📊 Data Collection: Fetches stock data from Yahoo Finance (via yfinance).
2. 🧹 Data Preparation: Cleans data and creates lag features.
3. 🛠 Feature Engineering: Adds technical indicators (RSI, MACD, Bollinger Bands), rolling statistics, returns, and volatility measures.
4. 🤖 Modeling: Trains an XGBoost Regressor to predict volatility.
5. 📉 Evaluation: Reports RMSE and R², and visualizes predictions vs actuals.
6. 🌐 Deployment: Full-stack React + FastAPI web application.



🛠 Tech Stack

1. Frontend: React + Vite, Tailwind CSS, Recharts
2. Backend: Python FastAPI, Uvicorn
3. yfinance – Stock market data
4. pandas, numpy – Data manipulation
5. ta (Technical Analysis library) – Indicators like RSI, MACD, Bollinger Bands
6. scikit-learn – Train-test split, metrics
7. xgboost – Machine learning model



▶️ Quick Start (Docker)

```bash
git clone <your-repo-url>
cd Stock-Volatility-Prediction-with-XGBoost
docker-compose up --build
```

Frontend → http://localhost:5173
Backend → http://localhost:8000
API Docs → http://localhost:8000/docs



▶️ Local Dev (no Docker)

Backend:
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

Frontend:
```bash
cd frontend
npm install
npm run dev
```



📊 Results

Predictions are evaluated using:

1. RMSE (Root Mean Squared Error)
2. R² Score

Outputs include:

1. Line charts of actual vs predicted volatility
2. Feature importance bar charts



🔮 Future Improvements

1. Add hyperparameter tuning for XGBoost
2. Integrate deep learning models (LSTM, Transformers)
3. Extend frontend with portfolio-level analysis
4. Add sentiment analysis (news & social media) as features
