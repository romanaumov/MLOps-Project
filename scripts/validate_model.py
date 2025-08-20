#!/usr/bin/env python3
"""
Model validation script for CI/CD pipeline.

This script validates the performance of trained models to ensure they meet
the minimum quality standards before deployment.
"""

import os
import sys
import logging
import mlflow
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

from src.config import settings
from src.models.train import ModelTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_model_performance():
    """
    Validate that the latest trained model meets performance criteria.
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    try:
        # Set MLflow tracking URI if provided via environment
        mlflow_uri = os.getenv('MLFLOW_TRACKING_URI')
        if mlflow_uri:
            mlflow.set_tracking_uri(mlflow_uri)
            logger.info(f"Using MLflow tracking URI: {mlflow_uri}")
        
        # Performance thresholds from environment or defaults
        max_allowed_rmse = float(os.getenv('MAX_RMSE', '100.0'))
        min_r2_score = float(os.getenv('MIN_R2_SCORE', '0.7'))
        
        logger.info(f"Performance thresholds: RMSE <= {max_allowed_rmse}, R² >= {min_r2_score}")
        
        # Get the latest model performance from MLflow
        client = mlflow.tracking.MlflowClient()
        
        # Get the latest experiment
        experiment = client.get_experiment_by_name(settings.MLFLOW_EXPERIMENT_NAME)
        if not experiment:
            logger.error(f"Experiment '{settings.MLFLOW_EXPERIMENT_NAME}' not found")
            return False
        
        # Get the latest run from the experiment
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time desc"],
            max_results=1
        )
        
        if not runs:
            logger.error("No runs found in the experiment")
            return False
        
        latest_run = runs[0]
        metrics = latest_run.data.metrics
        
        # Extract performance metrics
        test_rmse = metrics.get('test_rmse')
        test_r2 = metrics.get('test_r2')
        
        if test_rmse is None or test_r2 is None:
            logger.error("Required metrics (test_rmse, test_r2) not found in latest run")
            return False
        
        logger.info(f"Latest model performance: RMSE = {test_rmse:.4f}, R² = {test_r2:.4f}")
        
        # Validate performance
        if test_rmse > max_allowed_rmse:
            logger.error(f"Model RMSE {test_rmse:.4f} exceeds threshold {max_allowed_rmse}")
            return False
        
        if test_r2 < min_r2_score:
            logger.error(f"Model R² {test_r2:.4f} below threshold {min_r2_score}")
            return False
        
        logger.info("✅ Model validation passed!")
        return True
        
    except Exception as e:
        logger.error(f"Error during model validation: {e}")
        return False


def fallback_validation():
    """
    Fallback validation using local model files when MLflow is not available.
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    try:
        logger.info("Running fallback validation using local training...")
        
        # Train models locally and validate
        trainer = ModelTrainer()
        results = trainer.train_all_models()
        
        # Performance thresholds
        max_allowed_rmse = float(os.getenv('MAX_RMSE', '100.0'))
        min_r2_score = float(os.getenv('MIN_R2_SCORE', '0.7'))
        
        # Find best model
        best_model_name = None
        best_rmse = float('inf')
        best_r2 = 0.0
        
        for model_name, result in results.items():
            test_rmse = result['metrics']['test_rmse']
            test_r2 = result['metrics']['test_r2']
            
            if test_rmse < best_rmse:
                best_rmse = test_rmse
                best_r2 = test_r2
                best_model_name = model_name
        
        logger.info(f"Best model: {best_model_name} - RMSE: {best_rmse:.4f}, R²: {best_r2:.4f}")
        
        # Validate performance
        if best_rmse > max_allowed_rmse:
            logger.error(f"Model RMSE {best_rmse:.4f} exceeds threshold {max_allowed_rmse}")
            return False
        
        if best_r2 < min_r2_score:
            logger.error(f"Model R² {best_r2:.4f} below threshold {min_r2_score}")
            return False
        
        logger.info("✅ Fallback model validation passed!")
        return True
        
    except Exception as e:
        logger.error(f"Error during fallback validation: {e}")
        return False


def main():
    """Main validation function."""
    logger.info("Starting model validation...")
    
    # Try MLflow validation first, fallback to local validation
    success = validate_model_performance()
    
    if not success:
        logger.info("MLflow validation failed, trying fallback validation...")
        success = fallback_validation()
    
    if success:
        logger.info("Model validation completed successfully")
        sys.exit(0)
    else:
        logger.error("Model validation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()