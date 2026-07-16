import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, KFold
from xgboost import XGBRegressor
from .config import MlConfig
from .data import load_data, prepare_data
from .features import engineer_features


DEFAULT_PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}


def tune_hyperparameters(
    ticker: str,
    start_date: str,
    end_date: str,
    param_grid: dict | None = None,
    cv_folds: int = 3,
    config: MlConfig | None = None,
) -> dict:
    if config is None:
        config = MlConfig()
    if param_grid is None:
        param_grid = DEFAULT_PARAM_GRID

    data = load_data(ticker, start_date, end_date)
    if data.empty:
        raise ValueError("No data found for the given ticker and date range.")

    data = prepare_data(data, config)
    data = engineer_features(data, config)
    if data.empty or len(data) < 30:
        raise ValueError("Not enough data after cleaning and feature engineering. Try a wider date range.")

    X = data.drop(["volatility"], axis=1)
    y = data["volatility"]

    cv = KFold(n_splits=cv_folds, shuffle=False)
    model = XGBRegressor(random_state=config.random_state)
    gs = GridSearchCV(
        model,
        param_grid,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=0,
    )
    gs.fit(X, y)

    cv_results = []
    for i in range(len(gs.cv_results_["params"])):
        cv_results.append({
            "params": gs.cv_results_["params"][i],
            "mean_test_score": float(-gs.cv_results_["mean_test_score"][i]),
            "std_test_score": float(gs.cv_results_["std_test_score"][i]),
        })

    return {
        "ticker": ticker,
        "best_params": gs.best_params_,
        "best_score": float(-gs.best_score_),
        "cv_results": cv_results,
        "param_grid": {k: list(v) for k, v in param_grid.items()},
        "cv_folds": cv_folds,
        "total_rows": len(data),
    }
