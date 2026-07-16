from ml.config import MlConfig


def test_config_defaults():
    cfg = MlConfig()
    assert cfg.lag_windows == [1, 3, 5]
    assert cfg.lag_columns == ["Open", "High", "Low", "Close", "Volume"]
    assert cfg.rolling_windows == [5]
    assert cfg.return_horizons == [1, 3, 5]
    assert cfg.rsi_window == 14
    assert cfg.volatility_window == 5
    assert cfg.test_size == 0.2
    assert cfg.shuffle is False
    assert cfg.model_params == {}
    assert cfg.random_state == 42


def test_prepare_data():
    import pandas as pd
    from ml.data import prepare_data

    dates = pd.date_range("2020-01-01", periods=20, freq="D")
    data = pd.DataFrame({
        "Open": range(20),
        "High": range(1, 21),
        "Low": range(20),
        "Close": range(20),
        "Volume": [100] * 20,
    }, index=dates)

    result = prepare_data(data.copy())
    assert "Close_lag1" in result.columns
    assert "Close_lag3" in result.columns
    assert "Close_lag5" in result.columns
    assert "Open_lag1" in result.columns
    assert "Volume_lag5" in result.columns
    assert result["Close_lag1"].iloc[5] == 4.0


def test_engineer_features():
    import pandas as pd
    import numpy as np
    from ml.data import prepare_data
    from ml.features import engineer_features

    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=50, freq="D")
    data = pd.DataFrame({
        "Open": np.random.randn(50) + 100,
        "High": np.random.randn(50) + 101,
        "Low": np.random.randn(50) + 99,
        "Close": np.random.randn(50) + 100,
        "Volume": np.random.randint(100000, 500000, 50),
    }, index=dates)

    prepared = prepare_data(data)
    result = engineer_features(prepared)

    assert "RSI" in result.columns
    assert "MACD" in result.columns
    assert "MACD_signal" in result.columns
    assert "MACD_diff" in result.columns
    assert "BB_high" in result.columns
    assert "BB_low" in result.columns
    assert "BB_mid" in result.columns
    assert "BB_width" in result.columns
    assert "rolling_mean_5" in result.columns
    assert "rolling_std_5" in result.columns
    assert "return_1" in result.columns
    assert "return_3" in result.columns
    assert "return_5" in result.columns
    assert "volatility" in result.columns
    assert len(result) > 0


def test_split_data():
    import pandas as pd
    import numpy as np
    from ml.train import split_data

    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    data = pd.DataFrame({
        "Close": np.random.randn(100) + 100,
        "volatility": np.random.rand(100) * 0.1,
    }, index=dates)

    X_train, X_test, y_train, y_test = split_data(data)
    assert len(X_train) == 80
    assert len(X_test) == 20
    assert "volatility" not in X_train.columns


def test_train_model():
    import pandas as pd
    import numpy as np
    from ml.train import train_model, split_data

    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    data = pd.DataFrame({
        "Close": np.random.randn(100) + 100,
        "return_1": np.random.randn(100) * 0.01,
        "volatility": np.random.rand(100) * 0.1,
    }, index=dates)

    X_train, X_test, y_train, y_test = split_data(data)
    model = train_model(X_train, y_train)
    y_pred = model.predict(X_test)
    assert len(y_pred) == 20
    assert all(np.isfinite(y_pred))


def test_pipeline_response_shape():
    """Test that run_prediction returns the expected dict shape with real data."""
    from ml.pipeline import run_prediction

    result = run_prediction("AAPL", "2023-01-01", "2024-06-01")

    assert isinstance(result, dict)
    assert result["ticker"] == "AAPL"
    assert "rmse" in result
    assert "r2" in result
    assert "predictions" in result
    assert "feature_importance" in result
    assert "total_rows" in result
    assert "train_rows" in result
    assert "test_rows" in result
    assert result["r2"] is not None

    if result["predictions"]:
        first = result["predictions"][0]
        assert "date" in first
        assert "actual" in first
        assert "predicted" in first

    if result["feature_importance"]:
        first = result["feature_importance"][0]
        assert "feature" in first
        assert "importance" in first


if __name__ == "__main__":
    test_config_defaults()
    test_prepare_data()
    test_engineer_features()
    test_split_data()
    test_train_model()
    test_pipeline_response_shape()
    print("All tests passed")
