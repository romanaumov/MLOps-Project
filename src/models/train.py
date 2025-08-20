import logging
from typing import Any, Dict

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import cross_val_score

from src.config import settings
from src.data.preprocessing import DataPreprocessor

logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.models = {
            "linear_regression": LinearRegression(),
            "ridge_regression": Ridge(alpha=1.0),
            "random_forest": RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=settings.RANDOM_STATE,
                n_jobs=-1,
            ),
            "gradient_boosting": GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=settings.RANDOM_STATE,
            ),
        }

    def setup_mlflow(self) -> None:
        """Setup MLflow tracking."""
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)

        # Create experiment if it doesn't exist
        try:
            experiment = mlflow.get_experiment_by_name(settings.MLFLOW_EXPERIMENT_NAME)
            if experiment is None:
                mlflow.create_experiment(settings.MLFLOW_EXPERIMENT_NAME)
        except Exception as e:
            logger.warning(f"Could not create MLflow experiment: {e}")

        mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)
        logger.info(f"MLflow tracking URI: {settings.MLFLOW_TRACKING_URI}")

    def evaluate_model(
        self,
        model: Any,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> Dict[str, float]:
        """Evaluate model performance."""

        # Train predictions
        y_train_pred = model.predict(X_train)
        train_rmse = root_mean_squared_error(y_train, y_train_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)

        # Test predictions
        y_test_pred = model.predict(X_test)
        test_rmse = root_mean_squared_error(y_test, y_test_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        # Cross-validation score
        cv_scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=settings.CV_FOLDS,
            scoring="neg_root_mean_squared_error",
        )
        cv_rmse = -cv_scores.mean()
        cv_rmse_std = cv_scores.std()

        metrics = {
            "train_rmse": train_rmse,
            "train_mae": train_mae,
            "train_r2": train_r2,
            "test_rmse": test_rmse,
            "test_mae": test_mae,
            "test_r2": test_r2,
            "cv_rmse": cv_rmse,
            "cv_rmse_std": cv_rmse_std,
        }

        return metrics

    def train_single_model(
        self,
        model_name: str,
        model: Any,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> Dict[str, Any]:
        """Train a single model with MLflow tracking."""

        with mlflow.start_run(run_name=f"{model_name}_training"):
            # Log parameters
            if hasattr(model, "get_params"):
                params = model.get_params()
                mlflow.log_params(params)

            # Train model
            logger.info(f"Training {model_name}...")
            model.fit(X_train, y_train)

            # Evaluate model
            metrics = self.evaluate_model(model, X_train, X_test, y_train, y_test)

            # Log metrics
            mlflow.log_metrics(metrics)

            # Log model
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                registered_model_name=f"{settings.MODEL_NAME}_{model_name}",
            )

            # Save model locally
            model_dir = settings.MODEL_DIR
            model_dir.mkdir(exist_ok=True)
            model_path = model_dir / f"{model_name}_model.pkl"
            joblib.dump(model, model_path)

            # Log artifacts
            mlflow.log_artifact(str(model_path))

            logger.info(f"Model {model_name} - Test RMSE: {metrics['test_rmse']:.4f}")

            return {
                "model": model,
                "metrics": metrics,
                "run_id": mlflow.active_run().info.run_id,
            }

    def train_all_models(self) -> Dict[str, Dict[str, Any]]:
        """Train all models and compare performance."""
        logger.info("Starting model training pipeline...")

        # Setup MLflow
        self.setup_mlflow()

        # Load and preprocess data
        X_train, X_test, y_train, y_test = self.preprocessor.preprocess_pipeline()

        results = {}
        best_model = None
        best_rmse = float("inf")

        # Train each model
        for model_name, model in self.models.items():
            try:
                result = self.train_single_model(
                    model_name, model, X_train, X_test, y_train, y_test
                )
                results[model_name] = result

                # Track best model
                test_rmse = result["metrics"]["test_rmse"]
                if test_rmse < best_rmse:
                    best_rmse = test_rmse
                    best_model = model_name

            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                continue

        # Log best model comparison
        with mlflow.start_run(run_name="model_comparison"):
            comparison_metrics = {}
            for model_name, result in results.items():
                for metric_name, value in result["metrics"].items():
                    comparison_metrics[f"{model_name}_{metric_name}"] = value

            mlflow.log_metrics(comparison_metrics)
            mlflow.log_param("best_model", best_model)
            mlflow.log_metric("best_test_rmse", best_rmse)

        logger.info(
            f"Training completed. Best model: {best_model} (RMSE: {best_rmse:.4f})"
        )

        # Register best model for production
        if best_model and best_model in results:
            self.register_best_model(results[best_model], best_model)

        return results

    def register_best_model(self, best_result: Dict[str, Any], model_name: str) -> None:
        """Register the best model for production."""
        try:
            # Get the model URI
            run_id = best_result["run_id"]
            model_uri = f"runs:/{run_id}/model"

            # Register model
            model_version = mlflow.register_model(
                model_uri=model_uri, name=settings.MODEL_NAME
            )

            # Transition to production
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name=settings.MODEL_NAME,
                version=model_version.version,
                stage="Production",
            )

            logger.info(
                f"Registered {model_name} as production model (version {model_version.version})"
            )

        except Exception as e:
            logger.error(f"Error registering model: {e}")


def main():
    """Main training function."""
    logging.basicConfig(level=logging.INFO)

    trainer = ModelTrainer()
    results = trainer.train_all_models()

    # Print summary
    print("\n=== Training Results ===")
    for model_name, result in results.items():
        metrics = result["metrics"]
        print(f"{model_name}:")
        print(f"  Test RMSE: {metrics['test_rmse']:.4f}")
        print(f"  Test MAE: {metrics['test_mae']:.4f}")
        print(f"  Test R²: {metrics['test_r2']:.4f}")
        print(f"  CV RMSE: {metrics['cv_rmse']:.4f} ± {metrics['cv_rmse_std']:.4f}")
        print()


if __name__ == "__main__":
    main()
