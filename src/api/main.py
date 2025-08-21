import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import settings
from src.models.predict import BikeSharePredictor
from src.monitoring.data_drift import log_prediction_data

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Bike Sharing Demand Prediction API",
    description="API for predicting bike sharing demand using ML models",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor
predictor = BikeSharePredictor()


# Pydantic models for request/response
class PredictionInput(BaseModel):
    season: int = Field(
        ..., ge=1, le=4, description="Season (1:spring, 2:summer, 3:fall, 4:winter)"
    )
    yr: int = Field(..., ge=0, le=1, description="Year (0: 2011, 1:2012)")
    mnth: int = Field(..., ge=1, le=12, description="Month (1-12)")
    hr: int = Field(..., ge=0, le=23, description="Hour (0-23)")
    holiday: int = Field(..., ge=0, le=1, description="Holiday (0: no, 1: yes)")
    weekday: int = Field(..., ge=0, le=6, description="Day of week (0-6)")
    workingday: int = Field(..., ge=0, le=1, description="Working day (0: no, 1: yes)")
    weathersit: int = Field(..., ge=1, le=4, description="Weather situation (1-4)")
    temp: float = Field(..., ge=0, le=1, description="Normalized temperature")
    atemp: float = Field(..., ge=0, le=1, description="Normalized feeling temperature")
    hum: float = Field(..., ge=0, le=1, description="Normalized humidity")
    windspeed: float = Field(..., ge=0, le=1, description="Normalized wind speed")


class BatchPredictionInput(BaseModel):
    inputs: List[PredictionInput]


class PredictionOutput(BaseModel):
    prediction: float
    timestamp: datetime
    model_version: Optional[str] = None


class BatchPredictionOutput(BaseModel):
    predictions: List[float]
    timestamp: datetime
    model_version: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    model_loaded: bool


class ModelInfo(BaseModel):
    model_name: str
    model_stage: str
    is_loaded: bool
    feature_importance: Optional[Dict[str, float]] = None


@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup."""
    try:
        predictor.load_model_from_mlflow()
        logger.info("Model loaded successfully on startup")
    except Exception as e:
        logger.warning(f"Could not load model from MLflow on startup: {e}")
        try:
            predictor.load_latest_local_model()
            logger.info("Loaded latest local model on startup")
        except Exception as e2:
            logger.error(f"Could not load any model on startup: {e2}")


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "Bike Sharing Demand Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy", timestamp=datetime.now(), model_loaded=predictor.is_loaded
    )


@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint."""
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    
    # Generate Prometheus metrics
    metrics_data = generate_latest()
    
    return Response(
        content=metrics_data,
        media_type=CONTENT_TYPE_LATEST
    )


@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get information about the loaded model."""
    feature_importance = predictor.get_feature_importance()

    return ModelInfo(
        model_name=predictor.model_name,
        model_stage=predictor.model_stage,
        is_loaded=predictor.is_loaded,
        feature_importance=feature_importance,
    )


@app.post("/predict", response_model=PredictionOutput)
async def predict_single(
    input_data: PredictionInput, background_tasks: BackgroundTasks
):
    """Make a single prediction."""
    try:
        # Convert to dictionary
        data_dict = input_data.dict()

        # Make prediction
        prediction = predictor.predict_single(**data_dict)

        # Log prediction data for monitoring (async)
        background_tasks.add_task(
            log_prediction_data,
            input_data=data_dict,
            prediction=prediction,
            timestamp=datetime.now(),
        )

        return PredictionOutput(
            prediction=prediction,
            timestamp=datetime.now(),
            model_version=predictor.model_stage,
        )

    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionOutput)
async def predict_batch(
    input_data: BatchPredictionInput, background_tasks: BackgroundTasks
):
    """Make batch predictions."""
    try:
        # Convert to DataFrame
        data_list = [item.dict() for item in input_data.inputs]
        df = pd.DataFrame(data_list)

        # Make predictions
        predictions = predictor.predict(df)

        # Handle both numpy arrays and plain lists
        if hasattr(predictions, "tolist"):
            predictions_list = predictions.tolist()
        else:
            predictions_list = list(predictions)

        # Log batch prediction data for monitoring (async)
        background_tasks.add_task(
            log_prediction_data,
            input_data=data_list,
            prediction=predictions_list,
            timestamp=datetime.now(),
        )

        return BatchPredictionOutput(
            predictions=predictions_list,
            timestamp=datetime.now(),
            model_version=predictor.model_stage,
        )

    except Exception as e:
        logger.error(f"Error making batch prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/confidence")
async def predict_with_confidence(input_data: PredictionInput):
    """Make prediction with confidence intervals."""
    try:
        data_dict = input_data.dict()
        result = predictor.predict_with_confidence(data_dict)

        return {
            **result,
            "timestamp": datetime.now(),
            "model_version": predictor.model_stage,
        }

    except Exception as e:
        logger.error(f"Error making prediction with confidence: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/model/reload")
async def reload_model():
    """Reload the model from MLflow registry."""
    try:
        predictor.load_model_from_mlflow()
        return {
            "message": "Model reloaded successfully",
            "timestamp": datetime.now(),
            "model_name": predictor.model_name,
            "model_stage": predictor.model_stage,
        }
    except Exception as e:
        logger.error(f"Error reloading model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return HTTPException(status_code=400, detail=str(exc))


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unexpected error: {exc}")
    return HTTPException(status_code=500, detail="Internal server error")


if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app", host=settings.API_HOST, port=settings.API_PORT, reload=True
    )
