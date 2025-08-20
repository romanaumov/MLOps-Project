import logging
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from evidently import ColumnMapping

# from evidently.metrics import DataDriftPreset, DataQualityPreset
from evidently.metrics import (
    ColumnDriftMetric,
    DatasetDriftMetric,
    DatasetMissingValuesMetric,
)
from evidently.report import Report
from evidently.test_suite import TestSuite
from evidently.tests import TestNumberOfDriftedColumns, TestShareOfMissingValues

from src.config import settings

logger = logging.getLogger(__name__)


class DataDriftMonitor:
    def __init__(self, reference_data_path: Optional[str] = None):
        self.reference_data = None
        self.monitoring_db_path = (
            settings.PROJECT_ROOT / "monitoring" / "drift_data.db"
        )
        self.reports_dir = settings.PROJECT_ROOT / "monitoring" / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_monitoring_db()

        # Load reference data
        if reference_data_path:
            self.load_reference_data(reference_data_path)
        else:
            self._load_default_reference_data()

    def _init_monitoring_db(self) -> None:
        """Initialize SQLite database for monitoring data."""
        self.monitoring_db_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.monitoring_db_path) as conn:
            cursor = conn.cursor()

            # Create predictions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    season INTEGER,
                    yr INTEGER,
                    mnth INTEGER,
                    hr INTEGER,
                    holiday INTEGER,
                    weekday INTEGER,
                    workingday INTEGER,
                    weathersit INTEGER,
                    temp REAL,
                    atemp REAL,
                    hum REAL,
                    windspeed REAL,
                    prediction REAL,
                    actual REAL DEFAULT NULL
                )
            """)

            # Create drift reports table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS drift_reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    report_type TEXT,
                    drift_detected BOOLEAN,
                    drift_score REAL,
                    report_path TEXT
                )
            """)

            conn.commit()

    def _load_default_reference_data(self) -> None:
        """Load default reference data from training set."""
        try:
            reference_path = settings.PROCESSED_DATA_DIR / "X_train.csv"
            if reference_path.exists():
                self.reference_data = pd.read_csv(reference_path)
                logger.info(
                    f"Loaded reference data with shape: {self.reference_data.shape}"
                )
            else:
                logger.warning(
                    "No reference data found. Please train a model first."
                )
        except Exception as e:
            logger.error(f"Error loading reference data: {e}")

    def load_reference_data(self, file_path: str) -> None:
        """Load reference data from file."""
        try:
            self.reference_data = pd.read_csv(file_path)
            logger.info(f"Loaded reference data from {file_path}")
        except Exception as e:
            logger.error(f"Error loading reference data: {e}")
            raise

    def log_prediction(
        self,
        input_data: Dict[str, Any],
        prediction: float,
        timestamp: datetime,
        actual: Optional[float] = None,
    ) -> None:
        """Log a single prediction to the monitoring database."""
        try:
            with sqlite3.connect(self.monitoring_db_path) as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    INSERT INTO predictions (
                        timestamp, season, yr, mnth, hr, holiday, weekday, 
                        workingday, weathersit, temp, atemp, hum, windspeed, 
                        prediction, actual
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        timestamp,
                        input_data.get("season"),
                        input_data.get("yr"),
                        input_data.get("mnth"),
                        input_data.get("hr"),
                        input_data.get("holiday"),
                        input_data.get("weekday"),
                        input_data.get("workingday"),
                        input_data.get("weathersit"),
                        input_data.get("temp"),
                        input_data.get("atemp"),
                        input_data.get("hum"),
                        input_data.get("windspeed"),
                        prediction,
                        actual,
                    ),
                )

                conn.commit()

        except Exception as e:
            logger.error(f"Error logging prediction: {e}")

    def get_recent_predictions(self, hours: int = 24) -> pd.DataFrame:
        """Get recent predictions from the database."""
        try:
            with sqlite3.connect(self.monitoring_db_path) as conn:
                query = """
                    SELECT * FROM predictions 
                    WHERE timestamp >= datetime('now', '-{} hours')
                    ORDER BY timestamp DESC
                """.format(hours)

                df = pd.read_sql_query(query, conn)
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                return df

        except Exception as e:
            logger.error(f"Error getting recent predictions: {e}")
            return pd.DataFrame()

    def detect_data_drift(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect data drift using Evidently."""
        if self.reference_data is None:
            raise ValueError("Reference data not loaded")

        try:
            # Ensure columns match
            common_columns = set(self.reference_data.columns) & set(
                current_data.columns
            )
            reference_subset = self.reference_data[list(common_columns)]
            current_subset = current_data[list(common_columns)]

            # Create column mapping
            column_mapping = ColumnMapping()

            # Create data drift report
            # report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
            report = Report(
                metrics=[
                    ColumnDriftMetric(column_name="prediction"),
                    DatasetDriftMetric(),
                    DatasetMissingValuesMetric(),
                ]
            )

            report.run(
                reference_data=reference_subset,
                current_data=current_subset,
                column_mapping=column_mapping,
            )

            # Save report
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = (
                self.reports_dir / f"drift_report_{timestamp_str}.html"
            )
            report.save_html(str(report_path))

            # Extract drift results
            report_dict = report.as_dict()
            drift_results = self._extract_drift_metrics(report_dict)

            # Log to database
            self._log_drift_report(
                timestamp=datetime.now(),
                report_type="data_drift",
                drift_detected=drift_results["drift_detected"],
                drift_score=drift_results["drift_score"],
                report_path=str(report_path),
            )

            logger.info(
                f"Data drift analysis completed. Drift detected: {drift_results['drift_detected']}"
            )
            return drift_results

        except Exception as e:
            logger.error(f"Error detecting data drift: {e}")
            raise

    def _extract_drift_metrics(self, report_dict: Dict) -> Dict[str, Any]:
        """Extract drift metrics from Evidently report."""
        try:
            metrics = report_dict.get("metrics", [])

            drift_detected = False
            drift_score = 0.0
            drifted_columns = []

            for metric in metrics:
                # if metric.get("metric") == "DataDriftPreset":
                if metric.get("metric") == "ColumnDriftMetric":
                    result = metric.get("result", {})
                    drift_detected = result.get("dataset_drift", False)
                    drift_score = result.get("drift_share", 0.0)

                    # Get drifted columns
                    drift_by_columns = result.get("drift_by_columns", {})
                    drifted_columns = [
                        col
                        for col, drift_info in drift_by_columns.items()
                        if drift_info.get("drift_detected", False)
                    ]
                    break

            return {
                "drift_detected": drift_detected,
                "drift_score": drift_score,
                "drifted_columns": drifted_columns,
                "total_columns": len(report_dict.get("metrics", [])),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error extracting drift metrics: {e}")
            return {
                "drift_detected": False,
                "drift_score": 0.0,
                "drifted_columns": [],
                "total_columns": 0,
                "timestamp": datetime.now().isoformat(),
            }

    def _log_drift_report(
        self,
        timestamp: datetime,
        report_type: str,
        drift_detected: bool,
        drift_score: float,
        report_path: str,
    ) -> None:
        """Log drift report to database."""
        try:
            with sqlite3.connect(self.monitoring_db_path) as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    INSERT INTO drift_reports (
                        timestamp, report_type, drift_detected, drift_score, report_path
                    ) VALUES (?, ?, ?, ?, ?)
                """,
                    (
                        timestamp,
                        report_type,
                        drift_detected,
                        drift_score,
                        report_path,
                    ),
                )

                conn.commit()

        except Exception as e:
            logger.error(f"Error logging drift report: {e}")

    def run_drift_tests(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """Run data drift tests using Evidently test suite."""
        if self.reference_data is None:
            raise ValueError("Reference data not loaded")

        try:
            # Create test suite
            test_suite = TestSuite(
                tests=[
                    TestNumberOfDriftedColumns(
                        lt=0.3
                    ),  # Less than 30% of columns should drift
                    TestShareOfMissingValues(
                        lt=0.1
                    ),  # Less than 10% missing values
                ]
            )

            # Run tests
            test_suite.run(
                reference_data=self.reference_data, current_data=current_data
            )

            # Save results
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            test_report_path = (
                self.reports_dir / f"test_suite_{timestamp_str}.html"
            )
            test_suite.save_html(str(test_report_path))

            # Extract results
            test_results = test_suite.as_dict()

            return {
                "tests_passed": all(
                    test["status"] == "SUCCESS"
                    for test in test_results["tests"]
                ),
                "test_results": test_results,
                "report_path": str(test_report_path),
            }

        except Exception as e:
            logger.error(f"Error running drift tests: {e}")
            raise

    def check_and_alert(self, hours_window: int = 1) -> Dict[str, Any]:
        """Check for drift in recent data and send alerts if needed."""
        try:
            # Get recent predictions
            recent_data = self.get_recent_predictions(hours_window)

            if recent_data.empty:
                return {
                    "status": "no_data",
                    "message": "No recent data to analyze",
                }

            # Prepare data for drift detection
            feature_columns = [
                "season",
                "yr",
                "mnth",
                "hr",
                "holiday",
                "weekday",
                "workingday",
                "weathersit",
                "temp",
                "atemp",
                "hum",
                "windspeed",
            ]

            current_features = recent_data[feature_columns]

            # Run drift detection
            drift_results = self.detect_data_drift(current_features)

            # Check if alert is needed
            alert_needed = (
                drift_results["drift_detected"]
                or drift_results["drift_score"]
                > 0.3  # Alert if >30% of features drift
            )

            if alert_needed:
                alert_message = f"Data drift detected! Score: {drift_results['drift_score']:.3f}"
                logger.warning(alert_message)

                # Send email alert to admin@bikesharing.com
                try:
                    import subprocess
                    import json
                    
                    # Create alert data
                    alert_data = {
                        'type': 'drift_alert',
                        'drift_results': drift_results,
                        'message': alert_message
                    }
                    
                    # Send email notification
                    script_path = str(settings.PROJECT_ROOT / "scripts" / "send_drift_alert.py")
                    result = subprocess.run([
                        'python', script_path,
                        '--data', json.dumps(alert_data)
                    ], capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        logger.info("Drift alert email sent successfully")
                    else:
                        logger.error(f"Failed to send drift alert email: {result.stderr}")
                        
                except Exception as e:
                    logger.error(f"Error sending drift alert email: {e}")

                return {
                    "status": "alert",
                    "message": alert_message,
                    "drift_results": drift_results,
                }

            return {
                "status": "ok",
                "message": "No significant drift detected",
                "drift_results": drift_results,
            }

        except Exception as e:
            logger.error(f"Error in drift check and alert: {e}")
            return {"status": "error", "message": str(e)}


# Utility function for API integration
def log_prediction_data(
    input_data: Union[Dict, List[Dict]],
    prediction: Union[float, List[float]],
    timestamp: datetime,
) -> None:
    """Log prediction data for monitoring (async task)."""
    try:
        monitor = DataDriftMonitor()

        if isinstance(input_data, list):
            # Batch predictions
            for i, (data, pred) in enumerate(zip(input_data, prediction)):
                monitor.log_prediction(data, pred, timestamp)
        else:
            # Single prediction
            monitor.log_prediction(input_data, prediction, timestamp)

    except Exception as e:
        logger.error(f"Error logging prediction data: {e}")


def main():
    """Main function for testing."""
    logging.basicConfig(level=logging.INFO)

    monitor = DataDriftMonitor()

    # Example: Check for drift in recent data
    results = monitor.check_and_alert(hours_window=24)
    print(f"Drift check results: {results}")


if __name__ == "__main__":
    main()
