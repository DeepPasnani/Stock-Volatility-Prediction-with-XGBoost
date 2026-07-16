import sys
from pathlib import Path

# Add project root so ml package is importable from backend/
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml import run_prediction, MlConfig  # noqa: E402

__all__ = ["run_prediction", "MlConfig"]
