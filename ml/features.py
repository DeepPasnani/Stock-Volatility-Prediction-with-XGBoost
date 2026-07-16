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
