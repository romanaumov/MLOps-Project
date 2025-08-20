from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import DataPreprocessor


class TestDataPreprocessor:
    @pytest.fixture
    def preprocessor(self):
        return DataPreprocessor()

    @pytest.fixture
    def sample_data(self):
        return pd.DataFrame(
            {
                "season": [1, 2, 3, 4],
                "yr": [0, 1, 0, 1],
                "mnth": [1, 6, 9, 12],
                "hr": [8, 12, 18, 22],
                "holiday": [0, 0, 0, 1],
                "weekday": [1, 3, 5, 0],
                "workingday": [1, 1, 1, 0],
                "weathersit": [1, 1, 2, 1],
                "temp": [0.3, 0.6, 0.4, 0.2],
                "atemp": [0.3, 0.6, 0.4, 0.2],
                "hum": [0.5, 0.7, 0.6, 0.8],
                "windspeed": [0.2, 0.1, 0.3, 0.4],
                "cnt": [100, 200, 150, 50],
            }
        )

    def test_validate_data_no_issues(self, preprocessor, sample_data):
        result = preprocessor.validate_data(sample_data)
        assert result.equals(sample_data)

    def test_validate_data_negative_counts(self, preprocessor):
        data = pd.DataFrame(
            {"cnt": [-10, 50, 100], "casual": [5, -20, 30], "registered": [10, 70, 80]}
        )

        result = preprocessor.validate_data(data)

        assert result["cnt"].min() >= 0
        assert result["casual"].min() >= 0
        assert result["registered"].min() >= 0

    def test_create_features(self, preprocessor, sample_data):
        result = preprocessor.create_features(sample_data)

        # Check that new features are created
        expected_features = [
            "is_morning_rush",
            "is_evening_rush",
            "is_night",
            "temp_hum_interaction",
            "feels_like_temp",
            "is_weekend",
        ]

        for feature in expected_features:
            assert feature in result.columns

        # Test specific feature logic
        assert result.loc[0, "is_morning_rush"] == 1  # hr=8
        assert result.loc[2, "is_evening_rush"] == 1  # hr=18
        assert result.loc[3, "is_night"] == 1  # hr=22
        assert result.loc[3, "is_weekend"] == 1  # weekday=0 (Sunday)

    def test_prepare_features_scaling(self, preprocessor, sample_data):
        preprocessor.create_features(sample_data)
        X = preprocessor.prepare_features(sample_data, fit_scaler=True)

        # Check that numerical features are scaled
        numerical_cols = ["temp", "atemp", "hum", "windspeed"]
        for col in numerical_cols:
            if col in X.columns:
                # After scaling, mean should be close to 0 and std close to 1
                assert abs(X[col].mean()) < 0.1
                assert abs(X[col].std() - 1.0) < 0.1

    @patch.object(DataPreprocessor, "load_data")
    def test_preprocess_pipeline(self, mock_load_data, preprocessor, sample_data):
        mock_load_data.return_value = sample_data

        X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline()

        # Check shapes
        assert len(X_train) + len(X_test) == len(sample_data)
        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)

        # Check that target values are correct
        assert set(y_train.values).union(set(y_test.values)) == set(
            sample_data["cnt"].values
        )

    def test_split_data(self, preprocessor, sample_data):
        X_train, X_test, y_train, y_test = preprocessor.split_data(
            sample_data, test_size=0.25
        )

        total_size = len(sample_data)
        expected_test_size = int(total_size * 0.25)
        expected_train_size = total_size - expected_test_size

        assert len(X_test) == expected_test_size
        assert len(X_train) == expected_train_size
        assert len(y_test) == expected_test_size
        assert len(y_train) == expected_train_size
