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


if __name__ == "__main__":
    test_config_defaults()
    test_prepare_data()
    print("All tests passed")
