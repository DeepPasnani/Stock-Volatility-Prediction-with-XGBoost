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
