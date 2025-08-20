#!/usr/bin/env python3
"""
Download and validate production model from MLflow.

This script downloads the production model from MLflow model registry
and validates it for deployment.
"""

import os
import sys
import logging
import mlflow
import joblib
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

from src.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_production_model():
    """
    Download the production model from MLflow model registry.
    
    Returns:
        bool: True if download and validation successful, False otherwise
    """
    try:
        # Set MLflow tracking URI
        mlflow_uri = os.getenv('MLFLOW_TRACKING_URI')
        if not mlflow_uri:
            logger.error("MLFLOW_TRACKING_URI environment variable not set")
            return False
        
        mlflow.set_tracking_uri(mlflow_uri)
        logger.info(f"Using MLflow tracking URI: {mlflow_uri}")
        
        # Get model configuration from environment
        model_name = os.getenv('MODEL_NAME', settings.MLFLOW_EXPERIMENT_NAME)
        model_stage = os.getenv('MODEL_STAGE', 'Production')
        
        logger.info(f"Downloading model: {model_name} (stage: {model_stage})")
        
        # Initialize MLflow client
        client = mlflow.tracking.MlflowClient()
        
        # Get the latest production model
        try:
            model_versions = client.get_latest_versions(
                name=model_name,
                stages=[model_stage]
            )
            
            if not model_versions:
                logger.error(f"No model found in '{model_stage}' stage for '{model_name}'")
                return False
                
            latest_version = model_versions[0]
            logger.info(f"Found model version: {latest_version.version}")
            
        except Exception as e:
            logger.error(f"Error retrieving model from registry: {e}")
            return False
        
        # Download the model
        try:
            model_uri = f"models:/{model_name}/{model_stage}"
            logger.info(f"Downloading model from: {model_uri}")
            
            # Create models directory if it doesn't exist
            models_dir = project_root / "models"
            models_dir.mkdir(exist_ok=True)
            
            # Download model
            model_path = mlflow.artifacts.download_artifacts(
                artifact_uri=model_uri,
                dst_path=str(models_dir)
            )
            
            logger.info(f"Model downloaded to: {model_path}")
            
        except Exception as e:
            logger.error(f"Error downloading model: {e}")
            return False
        
        # Validate the downloaded model
        try:
            logger.info("Validating downloaded model...")
            
            # Try to load the model
            model = mlflow.pyfunc.load_model(model_uri)
            
            # Create test input data
            import pandas as pd
            test_data = pd.DataFrame({
                'season': [1], 'yr': [1], 'mnth': [1], 'hr': [12],
                'holiday': [0], 'weekday': [1], 'workingday': [1],
                'weathersit': [1], 'temp': [0.5], 'atemp': [0.5],
                'hum': [0.6], 'windspeed': [0.2]
            })
            
            # Test prediction
            prediction = model.predict(test_data)
            
            if prediction is None or len(prediction) == 0:
                raise ValueError("Model returned empty prediction")
            
            pred_value = prediction[0]
            if not isinstance(pred_value, (int, float)) or pred_value < 0:
                raise ValueError(f"Invalid prediction value: {pred_value}")
            
            logger.info(f"Model validation successful (test prediction: {pred_value:.2f})")
            
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return False
        
        # Get model metadata
        try:
            run_id = latest_version.run_id
            run = client.get_run(run_id)
            
            metrics = run.data.metrics
            params = run.data.params
            
            logger.info("Model metadata:")
            logger.info(f"  Run ID: {run_id}")
            logger.info(f"  Version: {latest_version.version}")
            logger.info(f"  Stage: {latest_version.current_stage}")
            
            if 'test_rmse' in metrics:
                logger.info(f"  Test RMSE: {metrics['test_rmse']:.4f}")
            if 'test_r2' in metrics:
                logger.info(f"  Test R²: {metrics['test_r2']:.4f}")
            
            # Create a metadata file
            metadata = {
                'model_name': model_name,
                'model_version': latest_version.version,
                'model_stage': latest_version.current_stage,
                'run_id': run_id,
                'mlflow_uri': model_uri,
                'download_path': model_path,
                'metrics': metrics,
                'params': params
            }
            
            import json
            metadata_file = models_dir / "production_model_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            logger.info(f"Model metadata saved to: {metadata_file}")
            
        except Exception as e:
            logger.warning(f"Could not retrieve model metadata: {e}")
        
        logger.info("✅ Production model download and validation successful")
        return True
        
    except Exception as e:
        logger.error(f"Error in download_production_model: {e}")
        return False


def main():
    """Main function."""
    logger.info("Starting production model download...")
    
    success = download_production_model()
    
    if success:
        logger.info("Production model download completed successfully")
        sys.exit(0)
    else:
        logger.error("Production model download failed")
        sys.exit(1)


if __name__ == "__main__":
    main()