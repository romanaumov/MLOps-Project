import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from typing import Union, Dict, Any, Optional
import logging
import joblib
from pathlib import Path

from src.config import settings
from src.data.preprocessing import DataPreprocessor

logger = logging.getLogger(__name__)


class BikeSharePredictor:
    def __init__(self, model_name: Optional[str] = None, model_stage: str = "Production"):
        self.model_name = model_name or settings.MODEL_NAME
        self.model_stage = model_stage
        self.model = None
        self.preprocessor = DataPreprocessor()
        self.is_loaded = False
    
    def load_model_from_mlflow(self) -> None:
        """Load model from MLflow registry."""
        try:
            mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
            
            model_uri = f"models:/{self.model_name}/{self.model_stage}"
            self.model = mlflow.sklearn.load_model(model_uri)
            self.is_loaded = True
            
            logger.info(f"Loaded model from MLflow: {model_uri}")
            
        except Exception as e:
            logger.error(f"Error loading model from MLflow: {e}")
            raise
    
    def load_model_from_file(self, model_path: Union[str, Path]) -> None:
        """Load model from local file."""
        try:
            self.model = joblib.load(model_path)
            self.is_loaded = True
            logger.info(f"Loaded model from file: {model_path}")
            
        except Exception as e:
            logger.error(f"Error loading model from file: {e}")
            raise
    
    def load_latest_local_model(self) -> None:
        """Load the latest model from local models directory."""
        model_dir = settings.MODEL_DIR
        
        if not model_dir.exists():
            raise FileNotFoundError(f"Models directory not found: {model_dir}")
        
        # Find all model files
        model_files = list(model_dir.glob("*_model.pkl"))
        
        if not model_files:
            raise FileNotFoundError(f"No model files found in {model_dir}")
        
        # Get the latest model file
        latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
        self.load_model_from_file(latest_model)
    
    def prepare_input_data(self, data: Union[Dict, pd.DataFrame]) -> pd.DataFrame:
        """Prepare input data for prediction."""
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        
        # For simple pipeline models, just ensure we have the basic features
        required_features = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 
                           'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed']
        
        missing_features = [col for col in required_features if col not in data.columns]
        
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Return data with required features in correct order
        return data[required_features]
    
    def predict(self, data: Union[Dict, pd.DataFrame]) -> np.ndarray:
        """Make predictions on input data."""
        if not self.is_loaded:
            # Try loading from MLflow first, then local file
            try:
                self.load_model_from_mlflow()
            except Exception:
                logger.warning("Could not load from MLflow, trying local model...")
                self.load_latest_local_model()
        
        # Simple input preparation - no complex preprocessing
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        
        # Required features for the pipeline model
        required_features = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 
                           'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed']
        
        missing_features = [col for col in required_features if col not in data.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        X = data[required_features]
        
        # Make predictions directly with the fitted pipeline 
        # (pipeline handles its own scaling internally)
        predictions = self.model.predict(X)
        
        # Ensure predictions are non-negative
        predictions = np.maximum(predictions, 0)
        
        return predictions
    
    def predict_single(self, **kwargs) -> float:
        """Make a single prediction from keyword arguments."""
        prediction = self.predict(kwargs)
        return float(prediction[0])
    
    def predict_with_confidence(
        self, 
        data: Union[Dict, pd.DataFrame],
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """Make predictions with confidence intervals (if model supports it)."""
        predictions = self.predict(data)
        
        result = {
            "predictions": predictions.tolist(),
            "mean_prediction": float(np.mean(predictions))
        }
        
        # For ensemble models, we could calculate confidence intervals
        # This is a simplified version
        if hasattr(self.model, "estimators_"):
            # For RandomForest, we can get predictions from all trees
            if hasattr(self.model, "estimators_"):
                X = self.prepare_input_data(data)
                tree_predictions = np.array([
                    tree.predict(X) for tree in self.model.estimators_
                ])
                
                prediction_std = np.std(tree_predictions, axis=0)
                
                # Calculate confidence intervals (assuming normal distribution)
                z_score = 1.96 if confidence_level == 0.95 else 2.576  # 99%
                margin_of_error = z_score * prediction_std
                
                result.update({
                    "confidence_intervals": {
                        "lower": (predictions - margin_of_error).tolist(),
                        "upper": (predictions + margin_of_error).tolist()
                    },
                    "prediction_std": prediction_std.tolist()
                })
        
        return result
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance if available."""
        if not self.is_loaded:
            return None
        
        if hasattr(self.model, "feature_importances_"):
            # Get feature names from the last preprocessing
            feature_names = self.preprocessor.feature_columns
            importance = self.model.feature_importances_
            
            return dict(zip(feature_names, importance))
        
        return None


def create_sample_prediction():
    """Create sample predictions for testing."""
    predictor = BikeSharePredictor()
    
    # Sample input data
    sample_data = {
        "season": 1,      # Spring
        "yr": 1,          # 2012
        "mnth": 6,        # June
        "hr": 8,          # 8 AM
        "holiday": 0,     # Not a holiday
        "weekday": 1,     # Monday
        "workingday": 1,  # Working day
        "weathersit": 1,  # Clear weather
        "temp": 0.5,      # Normalized temperature
        "atemp": 0.48,    # Normalized feeling temperature
        "hum": 0.6,       # Normalized humidity
        "windspeed": 0.2  # Normalized wind speed
    }
    
    try:
        prediction = predictor.predict_single(**sample_data)
        print(f"Predicted bike rentals: {prediction:.0f}")
        
        # Get prediction with confidence
        result = predictor.predict_with_confidence(sample_data)
        print(f"Prediction with confidence: {result}")
        
        # Get feature importance
        importance = predictor.get_feature_importance()
        if importance:
            print("Feature importance:")
            for feature, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
                print(f"  {feature}: {imp:.4f}")
        
    except Exception as e:
        logger.error(f"Error making prediction: {e}")


def main():
    """Main prediction function."""
    logging.basicConfig(level=logging.INFO)
    create_sample_prediction()


if __name__ == "__main__":
    main()