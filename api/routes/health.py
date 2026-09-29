from fastapi import APIRouter
from api.schemas import HealthResponse
from api.config import settings

router = APIRouter(tags=["Health"])

@router.get("/health", response_model=HealthResponse, summary="Service Health Check")
def health_check():
    """
    Returns API operational status, service name, and version.
    """
    return HealthResponse(
        status="healthy",
        app_name=settings.APP_TITLE,
        version=settings.APP_VERSION
    )
