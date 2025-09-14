# 📈 Stock-Volatility-Prediction-with-XGBoost

🔹 Overview

This project predicts stock volatility using historical market data and machine learning techniques.
It combines financial technical indicators, feature engineering, and XGBoost regression models to forecast short-term volatility.

Additionally, an interactive Streamlit web app is included, allowing users to input stock tickers and visualize predictions in real time.




🚀 Features

1. 📊 Data Collection: Fetches stock data from Yahoo Finance (via yfinance).
2. 🧹 Data Preparation: Cleans data and creates lag features.
3. 🛠 Feature Engineering: Adds technical indicators, rolling statistics, returns, and volatility measures.
4. 🤖 Modeling: Trains an XGBoost Regressor to predict volatility.
5. 📉 Evaluation: Reports RMSE and R², and visualizes predictions vs actuals.
6. 🌐 Deployment: Provides a Streamlit dashboard for interactive use.




🛠 Tech Stack

1. Python
2. yfinance – Stock market data
3. pandas, numpy – Data manipulation
4. ta (Technical Analysis library) – Indicators like RSI, MACD, Bollinger Bands
5. scikit-learn – Train-test split, metrics
6. xgboost – Machine learning model
7. matplotlib – Visualization
9. streamlit – Web app for deployment




▶️ Usage

1. Run from Command Line
python main.py


a. You will be prompted to enter:
b. Stock ticker (e.g., AAPL)
c. Start date (YYYY-MM-DD)
d. End date (YYYY-MM-DD)
e. The script will:
f. Fetch stock data
g. Train the XGBoost model
h. Print RMSE and R²

2. Run Streamlit App
streamlit run main.py
The web app will let you:

a. Enter ticker symbol
b. Select start and end dates
c. View predictions interactively
d. See evaluation metrics (RMSE, R²)
e. Explore feature importance




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
3. Extend Streamlit app with portfolio-level analysis
4. Add sentiment analysis (news & social media) as features