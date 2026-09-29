from fastapi import APIRouter, Query, status
from typing import List, Dict, Any
from api.schemas import (
    PredictionInput,
    PredictionOutput,
    BatchPredictionInput,
    BatchPredictionOutput,
    MessageResponse
)
from api.services import model_service
from api.config import settings

router = APIRouter(prefix="/predict", tags=["Inference & Prediction"])

@router.post("", response_model=PredictionOutput, summary="Run prediction with default model (POST)")
def predict_default(payload: PredictionInput):
    """
    Executes inference using the system's default active model.
    """
    return model_service.predict(
        model_id=settings.DEFAULT_MODEL_ID,
        features=payload.features
    )

@router.post("/{model_id}", response_model=PredictionOutput, summary="Run prediction with specific model (POST)")
def predict_by_model(model_id: str, payload: PredictionInput):
    """
    Executes inference using the specified model ID (e.g. linear_regression, random_forest).
    """
    return model_service.predict(
        model_id=model_id,
        features=payload.features
    )

@router.post("/{model_id}/batch", response_model=BatchPredictionOutput, summary="Run batch predictions (POST)")
def predict_batch(model_id: str, payload: BatchPredictionInput):
    """
    Executes batch inference over a list of feature sets.
    """
    return model_service.predict_batch(
        model_id=model_id,
        items=payload.items
    )

@router.get("/history", response_model=List[Dict[str, Any]], summary="Get prediction history (GET)")
def get_prediction_history(limit: int = Query(50, ge=1, le=500)):
    """
    Retrieve logged history of recent predictions.
    """
    return model_service.get_history(limit=limit)

@router.delete("/history", response_model=MessageResponse, summary="Clear prediction history (DELETE)")
def clear_prediction_history():
    """
    Deletes all recorded prediction history logs.
    """
    res = model_service.clear_history()
    return MessageResponse(
        success=True,
        message=res["message"],
        details={"cleared_count": res["cleared_count"]}
    )
