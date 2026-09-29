from fastapi import APIRouter, Query, status
from typing import List
from api.schemas import (
    ModelCreate,
    ModelUpdate,
    ModelPatch,
    ModelResponse,
    ModelListResponse,
    MessageResponse
)
from api.services import model_service

router = APIRouter(prefix="/models", tags=["Model Management"])

@router.get("", response_model=ModelListResponse, summary="List all registered models (GET)")
def list_models(active_only: bool = Query(False, description="Filter only active models")):
    """
    Retrieve all registered ML models with their configurations, versions, and statuses.
    """
    models = model_service.list_models(active_only=active_only)
    return ModelListResponse(total=len(models), models=models)

@router.get("/{model_id}", response_model=ModelResponse, summary="Get specific model details (GET)")
def get_model(model_id: str):
    """
    Fetch comprehensive metadata and hyperparameters for a specific model ID.
    """
    return model_service.get_model(model_id)

@router.post("", response_model=ModelResponse, status_code=status.HTTP_201_CREATED, summary="Register a new model (POST)")
def create_model(payload: ModelCreate):
    """
    Register and add a new model definition to the registry.
    """
    return model_service.create_model(payload)

@router.put("/{model_id}", response_model=ModelResponse, summary="Full update / replace a model (PUT)")
def update_model(model_id: str, payload: ModelUpdate):
    """
    Completely replace the configuration and metadata of an existing model.
    """
    return model_service.update_model(model_id, payload)

@router.patch("/{model_id}", response_model=ModelResponse, summary="Partial update model (PATCH)")
def patch_model(model_id: str, payload: ModelPatch):
    """
    Partially update model fields such as active state, tags, hyperparameters, or version.
    """
    return model_service.patch_model(model_id, payload)

@router.delete("/{model_id}", response_model=MessageResponse, summary="Delete / unregister a model (DELETE)")
def delete_model(model_id: str):
    """
    Remove a model completely from the active registry.
    """
    res = model_service.delete_model(model_id)
    return MessageResponse(
        success=True,
        message=res["message"],
        details={"model_id": res["model_id"]}
    )
