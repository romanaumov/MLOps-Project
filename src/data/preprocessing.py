import pandas as pd
import numpy as np
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging

from src.config import settings

logger = logging.getLogger(__name__)


class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = settings.FEATURE_COLUMNS
        self.target_column = settings.TARGET_COLUMN
    
    def load_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """Load raw data from CSV file."""
        if file_path is None:
            file_path = settings.RAW_DATA_DIR / "hour.csv"
        
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded data with shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate data quality and handle missing values."""
        logger.info("Validating data quality...")
        
        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.any():
            logger.warning(f"Missing values found: {missing_values[missing_values > 0]}")
        
        # Check for negative values in count columns
        count_columns = ["casual", "registered", "cnt"]
        for col in count_columns:
            if col in df.columns:
                negative_count = (df[col] < 0).sum()
                if negative_count > 0:
                    logger.warning(f"Found {negative_count} negative values in {col}")
                    df[col] = df[col].clip(lower=0)
        
        # Check for outliers in weather features
        weather_features = ["temp", "atemp", "hum", "windspeed"]
        for feature in weather_features:
            if feature in df.columns:
                # These features should be normalized between 0 and 1
                out_of_range = ((df[feature] < 0) | (df[feature] > 1)).sum()
                if out_of_range > 0:
                    logger.warning(f"Found {out_of_range} out-of-range values in {feature}")
                    df[feature] = df[feature].clip(0, 1)
        
        return df
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional features for model training."""
        logger.info("Creating additional features...")
        
        df = df.copy()
        
        # Convert date column to datetime
        if "dteday" in df.columns:
            df["dteday"] = pd.to_datetime(df["dteday"])
        
        # Create hour-based features
        if "hr" in df.columns:
            df["is_morning_rush"] = ((df["hr"] >= 7) & (df["hr"] <= 9)).astype(int)
            df["is_evening_rush"] = ((df["hr"] >= 17) & (df["hr"] <= 19)).astype(int)
            df["is_night"] = ((df["hr"] >= 22) | (df["hr"] <= 5)).astype(int)
        
        # Create weather interaction features
        if all(col in df.columns for col in ["temp", "hum"]):
            df["temp_hum_interaction"] = df["temp"] * df["hum"]
        
        if all(col in df.columns for col in ["temp", "windspeed"]):
            df["feels_like_temp"] = df["temp"] - (df["windspeed"] * 0.1)
        
        # Create weekend indicator
        if "weekday" in df.columns:
            df["is_weekend"] = (df["weekday"].isin([0, 6])).astype(int)
        
        # Create season-weather interactions
        if all(col in df.columns for col in ["season", "weathersit"]):
            df["season_weather"] = df["season"] * df["weathersit"]
        
        logger.info(f"Created features. New shape: {df.shape}")
        return df
    
    def prepare_features(self, df: pd.DataFrame, fit_scaler: bool = True) -> pd.DataFrame:
        """Prepare features for model training."""
        logger.info("Preparing features for training...")
        
        # Select feature columns (including new ones)
        available_features = [col for col in self.feature_columns if col in df.columns]
        
        # Add new engineered features
        engineered_features = [
            "is_morning_rush", "is_evening_rush", "is_night",
            "temp_hum_interaction", "feels_like_temp", "is_weekend", "season_weather"
        ]
        available_features.extend([col for col in engineered_features if col in df.columns])
        
        X = df[available_features].copy()
        
        # Handle categorical variables
        categorical_features = ["season", "weathersit", "mnth", "weekday"]
        for feature in categorical_features:
            if feature in X.columns:
                X[feature] = X[feature].astype("category")
        
        # Scale numerical features
        numerical_features = [col for col in X.columns if col not in categorical_features]
        
        if fit_scaler:
            X[numerical_features] = self.scaler.fit_transform(X[numerical_features])
        else:
            X[numerical_features] = self.scaler.transform(X[numerical_features])
        
        logger.info(f"Prepared {len(available_features)} features for training")
        return X
    
    def split_data(
        self, 
        df: pd.DataFrame, 
        test_size: float = None,
        random_state: int = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into train and test sets."""
        if test_size is None:
            test_size = settings.TEST_SIZE
        if random_state is None:
            random_state = settings.RANDOM_STATE
        
        # Prepare features and target
        X = self.prepare_features(df, fit_scaler=True)
        y = df[self.target_column]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=True
        )
        
        logger.info(f"Split data: Train {X_train.shape}, Test {X_test.shape}")
        return X_train, X_test, y_train, y_test
    
    def preprocess_pipeline(self, file_path: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Complete preprocessing pipeline."""
        logger.info("Starting complete preprocessing pipeline...")
        
        # Load data
        df = self.load_data(file_path)
        
        # Validate and clean data
        df = self.validate_data(df)
        
        # Create features
        df = self.create_features(df)
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(df)
        
        # Save processed data
        processed_dir = settings.PROCESSED_DATA_DIR
        processed_dir.mkdir(exist_ok=True)
        
        X_train.to_csv(processed_dir / "X_train.csv", index=False)
        X_test.to_csv(processed_dir / "X_test.csv", index=False)
        y_train.to_csv(processed_dir / "y_train.csv", index=False)
        y_test.to_csv(processed_dir / "y_test.csv", index=False)
        
        logger.info("Preprocessing pipeline completed successfully")
        return X_train, X_test, y_train, y_test