from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

def get_utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()

# ==========================================
# Common & Health Schemas
# ==========================================

class HealthResponse(BaseModel):
    status: str = Field(default="healthy", json_schema_extra={"example": "healthy"})
    app_name: str = Field(..., json_schema_extra={"example": "ML Model Service API"})
    version: str = Field(..., json_schema_extra={"example": "1.0.0"})
    timestamp: str = Field(default_factory=get_utc_timestamp)

class MessageResponse(BaseModel):
    success: bool = Field(default=True, json_schema_extra={"example": True})
    message: str = Field(..., json_schema_extra={"example": "Operation completed successfully"})
    details: Optional[Dict[str, Any]] = None

# ==========================================
# Model Management Schemas (CRUD)
# ==========================================

class ModelBase(BaseModel):
    name: str = Field(..., min_length=2, max_length=100, json_schema_extra={"example": "linear_regression"})
    version: str = Field(default="1.0.0", json_schema_extra={"example": "1.0.0"})
    description: str = Field(..., json_schema_extra={"example": "Standard Linear Regression model"})
    framework: str = Field(default="scikit-learn", json_schema_extra={"example": "scikit-learn"})
    is_active: bool = Field(default=True, json_schema_extra={"example": True})
    hyperparameters: Optional[Dict[str, Any]] = Field(default_factory=dict, json_schema_extra={"example": {"fit_intercept": True}})
    tags: Optional[List[str]] = Field(default_factory=list, json_schema_extra={"example": ["regression", "baseline"]})

class ModelCreate(ModelBase):
    model_id: str = Field(..., min_length=2, max_length=50, json_schema_extra={"example": "linear_regression"})

class ModelUpdate(ModelBase):
    """Schema for PUT requests (full update/replace)"""
    pass

class ModelPatch(BaseModel):
    """Schema for PATCH requests (partial update)"""
    name: Optional[str] = Field(None, json_schema_extra={"example": "Updated Model Name"})
    version: Optional[str] = Field(None, json_schema_extra={"example": "1.1.0"})
    description: Optional[str] = Field(None, json_schema_extra={"example": "Updated model description"})
    framework: Optional[str] = Field(None, json_schema_extra={"example": "scikit-learn"})
    is_active: Optional[bool] = Field(None, json_schema_extra={"example": False})
    hyperparameters: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None

class ModelResponse(ModelBase):
    model_id: str = Field(..., json_schema_extra={"example": "linear_regression"})
    created_at: str = Field(default_factory=get_utc_timestamp)
    updated_at: str = Field(default_factory=get_utc_timestamp)

class ModelListResponse(BaseModel):
    total: int = Field(..., json_schema_extra={"example": 1})
    models: List[ModelResponse]

# ==========================================
# Prediction / Inference Schemas
# ==========================================

class PredictionInput(BaseModel):
    features: Dict[str, Any] = Field(
        ...,
        description="Key-value pairs of input feature names and their values",
        json_schema_extra={
            "example": {
                "area_sqft": 1500,
                "bedrooms": 3,
                "bathrooms": 2,
                "age": 5,
                "parking": 1,
                "location_score": 8.5
            }
        }
    )

class BatchPredictionInput(BaseModel):
    items: List[Dict[str, Any]] = Field(
        ...,
        min_length=1,
        description="List of feature dictionaries to run predictions on",
        json_schema_extra={
            "example": [
                {"area_sqft": 1500, "bedrooms": 3, "bathrooms": 2, "age": 5, "parking": 1, "location_score": 8.5},
                {"area_sqft": 2200, "bedrooms": 4, "bathrooms": 3, "age": 2, "parking": 2, "location_score": 9.0}
            ]
        }
    )

class PredictionOutput(BaseModel):
    prediction_id: str
    model_id: str
    model_version: str
    input_features: Dict[str, Any]
    prediction: Any = Field(..., description="Predicted output value or class")
    confidence_or_metrics: Optional[Dict[str, Any]] = None
    latency_ms: float = Field(..., json_schema_extra={"example": 1.15})
    timestamp: str = Field(default_factory=get_utc_timestamp)

class BatchPredictionOutput(BaseModel):
    total_items: int
    model_id: str
    predictions: List[PredictionOutput]
    total_latency_ms: float
    timestamp: str = Field(default_factory=get_utc_timestamp)
