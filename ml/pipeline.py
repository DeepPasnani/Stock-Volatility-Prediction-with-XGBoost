import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from .config import MlConfig
from .data import load_data, prepare_data
from .features import engineer_features
from .train import split_data, train_model


def run_prediction(
    ticker: str,
    start_date: str,
    end_date: str,
    config: MlConfig = MlConfig(),
) -> dict:
    data = load_data(ticker, start_date, end_date)

    if data.empty:
        raise ValueError("No data found for the given ticker and date range.")

    data = prepare_data(data, config)
    data = engineer_features(data, config)

    if data.empty or len(data) < 30:
        raise ValueError("Not enough data after cleaning and feature engineering. Try a wider date range.")

    X_train, X_test, y_train, y_test = split_data(data, config)

    train_rows = len(X_train)
    test_rows = len(X_test)
    total_rows = len(data)

    model = train_model(X_train, y_train, config)
    y_pred = model.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))

    predictions = []
    for i, (date, actual) in enumerate(zip(y_test.index, y_test)):
        predictions.append({
            "date": str(date.date()) if hasattr(date, 'date') else str(date),
            "actual": float(actual),
            "predicted": float(y_pred[i]),
        })

    feature_importance = []
    feat_imp_series = pd.Series(model.feature_importances_, index=X_train.columns)
    feat_imp_series = feat_imp_series.sort_values(ascending=False).head(15)
    for feature, importance in feat_imp_series.items():
        feature_importance.append({
            "feature": feature,
            "importance": float(importance),
        })

    return {
        "ticker": ticker,
        "rmse": rmse,
        "r2": r2,
        "predictions": predictions,
        "feature_importance": feature_importance,
        "total_rows": total_rows,
        "train_rows": train_rows,
        "test_rows": test_rows,
    }
