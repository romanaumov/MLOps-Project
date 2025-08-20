import os
from pathlib import Path

try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings


class Settings(BaseSettings):
    # Project paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    MODEL_DIR: Path = PROJECT_ROOT / "models"
    
    # MLflow configuration
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    MLFLOW_EXPERIMENT_NAME: str = "bike-sharing-prediction"
    MLFLOW_S3_ENDPOINT_URL: str = "http://localhost:9000"
    
    # Model configuration
    MODEL_NAME: str = "bike-sharing-model"
    MODEL_STAGE: str = "Production"
    
    # Data configuration
    TARGET_COLUMN: str = "cnt"
    FEATURE_COLUMNS: list = [
        "season", "yr", "mnth", "hr", "holiday", "weekday", "workingday",
        "weathersit", "temp", "atemp", "hum", "windspeed"
    ]
    
    # Training configuration
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42
    CV_FOLDS: int = 5
    
    # Monitoring configuration
    EVIDENTLY_SERVICE_URL: str = "http://localhost:8085"
    GRAFANA_URL: str = "http://localhost:3000"
    
    # API configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    # Database configuration
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str = "bike_sharing"
    DB_USER: str = "postgres"
    DB_PASSWORD: str = "postgres"
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"


settings = Settings()