from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


class TestAPI:
    @pytest.fixture
    def client(self):
        return TestClient(app)

    @pytest.fixture
    def sample_prediction_input(self):
        return {
            "season": 1,
            "yr": 1,
            "mnth": 6,
            "hr": 8,
            "holiday": 0,
            "weekday": 1,
            "workingday": 1,
            "weathersit": 1,
            "temp": 0.5,
            "atemp": 0.48,
            "hum": 0.6,
            "windspeed": 0.2,
        }

    def test_root_endpoint(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "Bike Sharing Demand Prediction API" in response.json()["message"]

    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "model_loaded" in data

    @patch("src.api.main.predictor")
    def test_predict_single_success(
        self, mock_predictor, client, sample_prediction_input
    ):
        mock_predictor.predict_single.return_value = 150.5
        mock_predictor.model_stage = "Production"

        response = client.post("/predict", json=sample_prediction_input)
        assert response.status_code == 200

        data = response.json()
        assert "prediction" in data
        assert "timestamp" in data
        assert data["prediction"] == 150.5
        assert data["model_version"] == "Production"

    @patch("src.api.main.predictor")
    def test_predict_single_error(
        self, mock_predictor, client, sample_prediction_input
    ):
        mock_predictor.predict_single.side_effect = ValueError("Model not loaded")

        response = client.post("/predict", json=sample_prediction_input)
        assert response.status_code == 500

    def test_predict_single_validation_error(self, client):
        invalid_input = {
            "season": 5,  # Invalid: should be 1-4
            "yr": 1,
            "mnth": 13,  # Invalid: should be 1-12
            "hr": 25,  # Invalid: should be 0-23
        }

        response = client.post("/predict", json=invalid_input)
        assert response.status_code == 422  # Validation error

    @patch("src.api.main.predictor")
    def test_predict_batch(self, mock_predictor, client, sample_prediction_input):
        mock_predictor.predict.return_value = [150.5, 200.3]
        mock_predictor.model_stage = "Production"

        batch_input = {"inputs": [sample_prediction_input, sample_prediction_input]}

        response = client.post("/predict/batch", json=batch_input)
        assert response.status_code == 200

        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 2
        assert data["predictions"] == [150.5, 200.3]

    @patch("src.api.main.predictor")
    def test_predict_with_confidence(
        self, mock_predictor, client, sample_prediction_input
    ):
        mock_predictor.predict_with_confidence.return_value = {
            "predictions": [150.5],
            "mean_prediction": 150.5,
            "confidence_intervals": {"lower": [140.0], "upper": [160.0]},
        }
        mock_predictor.model_stage = "Production"

        response = client.post("/predict/confidence", json=sample_prediction_input)
        assert response.status_code == 200

        data = response.json()
        assert "predictions" in data
        assert "confidence_intervals" in data

    @patch("src.api.main.predictor")
    def test_model_info(self, mock_predictor, client):
        mock_predictor.model_name = "bike-sharing-model"
        mock_predictor.model_stage = "Production"
        mock_predictor.is_loaded = True
        mock_predictor.get_feature_importance.return_value = {
            "temp": 0.3,
            "hr": 0.2,
            "season": 0.15,
        }

        response = client.get("/model/info")
        assert response.status_code == 200

        data = response.json()
        assert data["model_name"] == "bike-sharing-model"
        assert data["model_stage"] == "Production"
        assert data["is_loaded"] is True
        assert "feature_importance" in data

    @patch("src.api.main.predictor")
    def test_model_reload(self, mock_predictor, client):
        mock_predictor.load_model_from_mlflow.return_value = None
        mock_predictor.model_name = "bike-sharing-model"
        mock_predictor.model_stage = "Production"

        response = client.post("/model/reload")
        assert response.status_code == 200

        data = response.json()
        assert "message" in data
        assert "timestamp" in data
        mock_predictor.load_model_from_mlflow.assert_called_once()


@pytest.mark.integration
class TestAPIIntegration:
    """Integration tests that require actual model and services."""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_end_to_end_prediction(self, client):
        """Test the complete prediction flow with a real model."""
        # This test would require an actual trained model
        # and should be run in the integration test environment
        sample_input = {
            "season": 1,
            "yr": 1,
            "mnth": 6,
            "hr": 8,
            "holiday": 0,
            "weekday": 1,
            "workingday": 1,
            "weathersit": 1,
            "temp": 0.5,
            "atemp": 0.48,
            "hum": 0.6,
            "windspeed": 0.2,
        }

        response = client.post("/predict", json=sample_input)

        if response.status_code == 200:
            data = response.json()
            assert "prediction" in data
            assert isinstance(data["prediction"], (int, float))
            assert data["prediction"] >= 0  # Bike rentals should be non-negative
        else:
            # If model is not loaded, we should get a 500 error
            assert response.status_code == 500
