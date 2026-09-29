import time
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from fastapi import HTTPException, status
from api.schemas import (
    ModelCreate,
    ModelUpdate,
    ModelPatch,
    ModelResponse,
    PredictionInput,
    PredictionOutput,
    BatchPredictionInput,
    BatchPredictionOutput
)

def get_utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()

class ModelService:
    """
    Service managing ML models registry and inference operations.
    Can be used with in-memory state or connected to a DB/Storage.
    """
    def __init__(self):
        self._models: Dict[str, Dict[str, Any]] = {}
        self._prediction_history: List[Dict[str, Any]] = []
        self._seed_default_models()

    def _seed_default_models(self):
        """Seed initial standard models."""
        now = get_utc_timestamp()
        defaults = [
            {
                "model_id": "linear_regression",
                "name": "Linear Regression Baseline",
                "version": "1.0.0",
                "description": "Standard OLS Linear Regression for tabular data",
                "framework": "scikit-learn",
                "is_active": True,
                "hyperparameters": {"fit_intercept": True},
                "tags": ["regression", "baseline", "linear"],
                "created_at": now,
                "updated_at": now
            },
            {
                "model_id": "random_forest",
                "name": "Random Forest Ensemble",
                "version": "1.0.0",
                "description": "Ensemble random forest regressor with bagging",
                "framework": "scikit-learn",
                "is_active": True,
                "hyperparameters": {"n_estimators": 100, "max_depth": 10},
                "tags": ["regression", "ensemble", "tree"],
                "created_at": now,
                "updated_at": now
            }
        ]
        for m in defaults:
            self._models[m["model_id"]] = m

    # -------------------------------------------------------------
    # GET: List models or get specific model
    # -------------------------------------------------------------
    def list_models(self, active_only: bool = False) -> List[ModelResponse]:
        models = list(self._models.values())
        if active_only:
            models = [m for m in models if m.get("is_active", True)]
        return [ModelResponse(**m) for m in models]

    def get_model(self, model_id: str) -> ModelResponse:
        if model_id not in self._models:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{model_id}' not found in registry"
            )
        return ModelResponse(**self._models[model_id])

    # -------------------------------------------------------------
    # POST: Register a new model
    # -------------------------------------------------------------
    def create_model(self, payload: ModelCreate) -> ModelResponse:
        if payload.model_id in self._models:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Model with ID '{payload.model_id}' already exists"
            )
        now = get_utc_timestamp()
        data = payload.model_dump()
        data["created_at"] = now
        data["updated_at"] = now
        self._models[payload.model_id] = data
        return ModelResponse(**data)

    # -------------------------------------------------------------
    # PUT: Full update / replace an existing model
    # -------------------------------------------------------------
    def update_model(self, model_id: str, payload: ModelUpdate) -> ModelResponse:
        if model_id not in self._models:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{model_id}' not found to update"
            )
        existing = self._models[model_id]
        data = payload.model_dump()
        data["model_id"] = model_id
        data["created_at"] = existing["created_at"]
        data["updated_at"] = get_utc_timestamp()
        self._models[model_id] = data
        return ModelResponse(**data)

    # -------------------------------------------------------------
    # PATCH: Partial update model properties
    # -------------------------------------------------------------
    def patch_model(self, model_id: str, payload: ModelPatch) -> ModelResponse:
        if model_id not in self._models:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{model_id}' not found to patch"
            )
        existing = self._models[model_id]
        update_data = payload.model_dump(exclude_unset=True)
        for key, value in update_data.items():
            existing[key] = value
        existing["updated_at"] = get_utc_timestamp()
        self._models[model_id] = existing
        return ModelResponse(**existing)

    # -------------------------------------------------------------
    # DELETE: Delete/Unregister model
    # -------------------------------------------------------------
    def delete_model(self, model_id: str) -> Dict[str, Any]:
        if model_id not in self._models:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{model_id}' not found to delete"
            )
        deleted = self._models.pop(model_id)
        return {
            "deleted": True,
            "model_id": model_id,
            "message": f"Model '{model_id}' successfully removed from registry"
        }

    # -------------------------------------------------------------
    # POST: Run Inference / Predict
    # -------------------------------------------------------------
    def predict(self, model_id: str, features: Dict[str, Any]) -> PredictionOutput:
        model = self.get_model(model_id)
        if not model.is_active:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Model '{model_id}' is currently inactive"
            )
        
        start_time = time.perf_counter()
        
        # Generic inference calculation: customizable for any model
        prediction_val = self._run_inference_computation(model_id, features)
        latency_ms = round((time.perf_counter() - start_time) * 1000, 3)

        output = PredictionOutput(
            prediction_id=str(uuid.uuid4()),
            model_id=model_id,
            model_version=model.version,
            input_features=features,
            prediction=prediction_val,
            confidence_or_metrics={"status": "computed", "latency_ms": latency_ms},
            latency_ms=latency_ms,
            timestamp=get_utc_timestamp()
        )
        
        # Save to prediction history log
        self._prediction_history.append(output.model_dump())
        return output

    def predict_batch(self, model_id: str, items: List[Dict[str, Any]]) -> BatchPredictionOutput:
        start_time = time.perf_counter()
        results = [self.predict(model_id, item) for item in items]
        total_latency = round((time.perf_counter() - start_time) * 1000, 3)
        return BatchPredictionOutput(
            total_items=len(results),
            model_id=model_id,
            predictions=results,
            total_latency_ms=total_latency,
            timestamp=get_utc_timestamp()
        )

    def _run_inference_computation(self, model_id: str, features: Dict[str, Any]) -> Any:
        """
        Generic inference engine. Computes a numeric or structured prediction.
        """
        numeric_sum = sum(v for v in features.values() if isinstance(v, (int, float)))
        if "area_sqft" in features:
            area = float(features.get("area_sqft", 1000))
            rooms = float(features.get("bedrooms", 2))
            loc = float(features.get("location_score", 5))
            return round(area * 4000 + rooms * 200000 + (loc ** 2) * 15000, 2)
        return round(numeric_sum * 1.5 + 42.0, 2)

    # -------------------------------------------------------------
    # GET & DELETE: History routes
    # -------------------------------------------------------------
    def get_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self._prediction_history[-limit:]

    def clear_history(self) -> Dict[str, Any]:
        count = len(self._prediction_history)
        self._prediction_history.clear()
        return {"cleared_count": count, "message": "Prediction history cleared successfully"}

model_service = ModelService()
