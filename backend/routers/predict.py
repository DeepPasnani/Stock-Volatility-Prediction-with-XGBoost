from fastapi import APIRouter, HTTPException
from datetime import datetime, timedelta
from schemas import PredictRequest, PredictResponse
from services.model import run_prediction

router = APIRouter()

@router.post("/api/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if not request.ticker or not request.ticker.strip():
        raise HTTPException(status_code=400, detail="Ticker cannot be empty.")
    
    try:
        start_date = datetime.strptime(request.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(request.end_date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")
    
    if (end_date - start_date).days < 60:
        raise HTTPException(status_code=400, detail="Date range must be at least 60 days.")
    
    try:
        result = run_prediction(request.ticker, request.start_date, request.end_date)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ML pipeline failed: {str(e)}")