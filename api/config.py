from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    APP_TITLE: str = "ML Model Service API"
    APP_DESCRIPTION: str = "Standard reusable FastAPI template for Machine Learning model serving with complete CRUD (GET, POST, PUT, PATCH, DELETE) routes."
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CORS_ORIGINS: List[str] = ["*"]
    DEFAULT_MODEL_ID: str = "linear_regression"

settings = Settings()
