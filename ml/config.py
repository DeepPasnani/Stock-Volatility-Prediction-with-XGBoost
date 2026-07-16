from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MlConfig:
    lag_windows: list[int] = field(default_factory=lambda: [1, 3, 5])
    lag_columns: list[str] = field(default_factory=lambda: ["Open", "High", "Low", "Close", "Volume"])
    rolling_windows: list[int] = field(default_factory=lambda: [5])
    return_horizons: list[int] = field(default_factory=lambda: [1, 3, 5])
    rsi_window: int = 14
    volatility_window: int = 5
    test_size: float = 0.2
    shuffle: bool = False
    model_params: dict = field(default_factory=dict)
    random_state: int = 42
