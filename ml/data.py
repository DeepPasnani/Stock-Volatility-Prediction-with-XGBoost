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
