from pydantic import BaseModel
from typing import List

class PredictRequest(BaseModel):
    ticker: str
    start_date: str
    end_date: str

class DataPoint(BaseModel):
    date: str
    actual: float
    predicted: float

class FeatureImportance(BaseModel):
    feature: str
    importance: float

class PredictResponse(BaseModel):
    ticker: str
    rmse: float
    r2: float
    predictions: List[DataPoint]
    feature_importance: List[FeatureImportance]
    total_rows: int
    train_rows: int
    test_rows: int
