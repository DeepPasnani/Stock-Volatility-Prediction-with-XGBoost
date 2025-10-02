import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as pta
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import streamlit as st

# # 1. Data Collection
# def load_data(ticker, start, end):
#     print('^^^^', ticker, start, end)
#     data = yf.download(ticker, start=start, end=end)
#     return data

# 1. Data Collection
def load_data(ticker, start, end):
    print('^^^^', ticker, start, end)
    # Convert date objects to string format for yfinance
    start = str(start)
    end = str(end)
    data = yf.download(ticker, start=start, end=end)
    print(f"Downloaded data range: {data.index[0]} to {data.index[-1]}")
    print(f"Total rows: {len(data)}")
    return data

# 2. Data Cleaning & Preparation
def prepare_data(data):
    data = data.dropna()
    data = add_lagged_features(data)
    return data

def add_lagged_features(data):
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        print("Hello", data[col])
        for lag in [1, 3, 5]:
            data[f'{col}_lag{lag}'] = data[col].shift(lag)
    return data

# 3. Feature Engineering
def engineer_features(data):
# Ensure required columns exist and are not all NaN
    required_cols = ["Open", "High", "Low", "Close", "Volume"]
# for col in required_cols:
#     if col not in data.columns:
#         print(f"Column '{col}' is missing.")
#         return pd.DataFrame()
#     if data[col].isnull().all():
#         print(f"Column '{col}' contains only NaN values.")
#         return pd.DataFrame()

# Technical indicators
# # data.ta.strategy("all")
# Additional engineered features
# data['rolling_mean_5'] = data['Close'].rolling(5).mean()
    data['rolling_std_5'] = data['Close'].rolling(5).std()
    data['return_1'] = data['Close'].pct_change(1)
    data['return_3'] = data['Close'].pct_change(3)
    data['return_5'] = data['Close'].pct_change(5)
    data['volatility'] = data['return_1'].rolling(5).std().shift(-1)
    data = data.dropna()
    return data

# def engineer_features(data):
#     # Rolling stats and returns
#     data['rolling_mean_5'] = data['Close'].rolling(5).mean()
#     data['rolling_std_5'] = data['Close'].rolling(5).std()
#     data['return_1'] = data['Close'].pct_change(1)
#     data['return_3'] = data['Close'].pct_change(3)
#     data['return_5'] = data['Close'].pct_change(5)

#     # Technical indicators with pandas_ta
#     data['RSI'] = pta.rsi(data['Close'], length=14)
#     # macd = pta.macd(data['Close'], fast=12, slow=26, signal=9)
#     # data = pd.concat([data, macd], axis=1)
    
#     # Volatility target
#     data['volatility'] = data['return_1'].rolling(5).std().shift(-1)

#     data = data.dropna()
#     print(data)
#     return data


# 4. Train-Test Split
def split_data(data):
    X = data.drop(['volatility'], axis=1)
    y = data['volatility']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    return X_train, X_test, y_train, y_test

# 5. Modeling
def train_model(X_train, y_train):
    model = XGBRegressor()
    model.fit(X_train, y_train)
    return model

# 6. Model Evaluation
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    plt.figure(figsize=(10, 5))
    plt.plot(y_test.index, y_test, label='Actual')
    plt.plot(y_test.index, y_pred, label='Predicted')
    plt.title('Predicted vs Actual Volatility')
    plt.legend()
    plt.show()
    
    importance = model.feature_importances_
    return rmse, r2, importance

# 7. Optional Deployment: Streamlit App
def run_streamlit():
    st.title("Stock Volatility Prediction App")
    ticker = st.text_input("Ticker Symbol (e.g. AAPL)", "AAPL")
    start = st.date_input("Start Date", pd.to_datetime("2015-01-01"))
    end = st.date_input("End Date", pd.to_datetime("2025-08-13"))
    
    if st.button("Predict Volatility"):
        with st.spinner("Downloading and processing data..."):
            data = load_data(ticker, start, end)
            if data.empty:
                st.error("No data found for the given ticker and date range.")
                return
            
            print("Before engieering")
            data = prepare_data(data)
            print("Prepare done")
            data = engineer_features(data)
            print("Engieering fone")
            print(data.columns)
            if data.empty or len(data) < 30:
                st.error("Not enough data after cleaning and feature engineering. Try a wider date range.")
                return
            print("Before splitting")
            X_train, X_test, y_train, y_test = split_data(data)
            print("Split done")
            
            print(X_train.columns)
            model = train_model(X_train, y_train)
            y_pred = model.predict(X_test)
        
        # Visualizations in Streamlit
        st.line_chart(pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}, index=y_test.index))
        st.write(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
        st.write(f"R²: {r2_score(y_test, y_pred):.4f}")
        # st.bar_chart(pd.Series(model.feature_importances_, index=X_train.columns))

# Uncomment to run as script
# if __name__ == "__main__":
#     print("Enter stock ticker, start date (YYYY-MM-DD), and end date (YYYY-MM-DD):")
#     ticker = input("Ticker: ").strip().upper()
#     start = input("Start Date (YYYY-MM-DD): ").strip()
#     end = input("End Date (YYYY-MM-DD): ").strip()
#     data = load_data(ticker, start, end)
#     if data.empty:
#         print("No data found for the given ticker and date range.")
#     else:
#         data = prepare_data(data)
#         data = engineer_features(data)
#         if data.empty or len(data) < 30:
#             print("Not enough data after cleaning and feature engineering. Try a wider date range.")
#         else:
#             X_train, X_test, y_train, y_test = split_data(data)
#             model = train_model(X_train, y_train)
#             rmse, r2, importance = evaluate_model(model, X_test, y_test)
#             print(f"RMSE: {rmse:.4f}")
#             print(f"R²: {r2:.4f}")

run_streamlit()