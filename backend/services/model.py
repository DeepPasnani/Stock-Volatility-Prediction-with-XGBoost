import yfinance as yf
import pandas as pd
import numpy as np
import ta
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from typing import Tuple, Dict, Any, List


def load_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    start = str(start)
    end = str(end)
    data = yf.download(ticker, start=start, end=end)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data


def prepare_data(data: pd.DataFrame) -> pd.DataFrame:
    data = data.dropna()
    data = add_lagged_features(data)
    return data


def add_lagged_features(data: pd.DataFrame) -> pd.DataFrame:
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        for lag in [1, 3, 5]:
            data[f'{col}_lag{lag}'] = data[col].squeeze().shift(lag)
    return data


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
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


def split_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = data.drop(['volatility'], axis=1)
    y = data['volatility']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    return X_train, X_test, y_train, y_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> XGBRegressor:
    model = XGBRegressor()
    model.fit(X_train, y_train)
    return model


def run_prediction(ticker: str, start_date: str, end_date: str) -> Dict[str, Any]:
    data = load_data(ticker, start_date, end_date)
    
    if data.empty:
        raise ValueError("No data found for the given ticker and date range.")
    
    data = prepare_data(data)
    data = engineer_features(data)
    
    if data.empty or len(data) < 30:
        raise ValueError("Not enough data after cleaning and feature engineering. Try a wider date range.")
    
    X_train, X_test, y_train, y_test = split_data(data)
    
    train_rows = len(X_train)
    test_rows = len(X_test)
    total_rows = len(data)
    
    model = train_model(X_train, y_train)
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    predictions = []
    for i, (date, actual) in enumerate(zip(y_test.index, y_test)):
        predictions.append({
            "date": str(date.date()) if hasattr(date, 'date') else str(date),
            "actual": float(actual),
            "predicted": float(y_pred[i])
        })
    
    feature_importance = []
    feat_imp_series = pd.Series(model.feature_importances_, index=X_train.columns)
    feat_imp_series = feat_imp_series.sort_values(ascending=False).head(15)
    for feature, importance in feat_imp_series.items():
        feature_importance.append({
            "feature": feature,
            "importance": float(importance)
        })
    
    return {
        "ticker": ticker,
        "rmse": float(rmse),
        "r2": float(r2),
        "predictions": predictions,
        "feature_importance": feature_importance,
        "total_rows": total_rows,
        "train_rows": train_rows,
        "test_rows": test_rows
    }
