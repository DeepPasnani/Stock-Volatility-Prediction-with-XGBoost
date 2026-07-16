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


if __name__ == "__main__":
    test_config_defaults()
    print("Config tests passed")
