from api.routes.health import router as health_router
from api.routes.models import router as models_router
from api.routes.predict import router as predict_router

__all__ = ["health_router", "models_router", "predict_router"]
