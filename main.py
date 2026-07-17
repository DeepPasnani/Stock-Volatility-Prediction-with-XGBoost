import yfinance as yf
import pandas as pd
import numpy as np
import ta
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import streamlit as st
import sys

def load_data(ticker, start, end):
    print('^^^^', ticker, start, end)
    start = str(start)
    end = str(end)
    data = yf.download(ticker, start=start, end=end)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    print(f"Downloaded data range: {data.index[0]} to {data.index[-1]}")
    print(f"Total rows: {len(data)}")
    return data

def prepare_data(data):
    data = data.dropna()
    data = add_lagged_features(data)
    return data

def add_lagged_features(data):
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        for lag in [1, 3, 5]:
            data[f'{col}_lag{lag}'] = data[col].squeeze().shift(lag)
    return data

def engineer_features(data):
    close = data['Close'].squeeze()
    high = data['High'].squeeze()
    low = data['Low'].squeeze()

    data['rolling_mean_5'] = close.rolling(5).mean()
    data['rolling_std_5'] = close.rolling(5).std()
    data['return_1'] = close.pct_change(1)
    data['return_3'] = close.pct_change(3)
    data['return_5'] = close.pct_change(5)

    data['RSI'] = ta.momentum.RSIIndicator(close, window=14).rsi()

    macd = ta.trend.MACD(close)
    data['MACD'] = macd.macd()
    data['MACD_signal'] = macd.macd_signal()
    data['MACD_diff'] = macd.macd_diff()

    bb = ta.volatility.BollingerBands(close)
    data['BB_high'] = bb.bollinger_hband()
    data['BB_low'] = bb.bollinger_lband()
    data['BB_mid'] = bb.bollinger_mavg()
    data['BB_width'] = bb.bollinger_wband()

    data['volatility'] = data['return_1'].rolling(5).std().shift(-1)
    data = data.dropna()
    return data

def split_data(data):
    X = data.drop(['volatility'], axis=1)
    y = data['volatility']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    model = XGBRegressor()
    model.fit(X_train, y_train)
    return model

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
            
            print("Before engineering")
            data = prepare_data(data)
            print("Prepare done")
            data = engineer_features(data)
            print("Engineering done")
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
        
        st.line_chart(pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}, index=y_test.index))
        st.write(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
        st.write(f"R²: {r2_score(y_test, y_pred):.4f}")
        
        st.subheader("Feature Importance")
        feat_importance = pd.Series(model.feature_importances_, index=X_train.columns)
        feat_importance = feat_importance.sort_values(ascending=False).head(15)
        st.bar_chart(feat_importance)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "streamlit":
        run_streamlit()
    else:
        print("Enter stock ticker, start date (YYYY-MM-DD), and end date (YYYY-MM-DD):")
        ticker = input("Ticker: ").strip().upper()
        start = input("Start Date (YYYY-MM-DD): ").strip()
        end = input("End Date (YYYY-MM-DD): ").strip()
        data = load_data(ticker, start, end)
        if data.empty:
            print("No data found.")
        else:
            data = prepare_data(data)
            data = engineer_features(data)
            if data.empty or len(data) < 30:
                print("Not enough data. Try a wider date range.")
            else:
                X_train, X_test, y_train, y_test = split_data(data)
                model = train_model(X_train, y_train)
                rmse, r2, importance = evaluate_model(model, X_test, y_test)
                print(f"RMSE: {rmse:.4f}")
                print(f"R²: {r2:.4f}")

if not any("streamlit" in arg for arg in sys.argv):
    pass
else:
    run_streamlit()
