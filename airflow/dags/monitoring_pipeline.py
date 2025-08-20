from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
import pandas as pd
import logging
import sys

# Add src to path
sys.path.append('/opt/airflow/dags/src')

from src.monitoring.data_drift import DataDriftMonitor
from src.models.predict import BikeSharePredictor

logger = logging.getLogger(__name__)

# Default arguments
default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

# Create monitoring DAG
dag = DAG(
    'bike_sharing_monitoring_pipeline',
    default_args=default_args,
    description='Monitoring pipeline for bike sharing prediction model',
    schedule_interval='@hourly',  # Run every hour
    catchup=False,
    tags=['monitoring', 'drift-detection', 'bike-sharing'],
    max_active_runs=1,
)


def check_model_health(**context):
    """Check if the model is healthy and responding."""
    logger.info("Checking model health...")
    
    try:
        predictor = BikeSharePredictor()
        
        # Test prediction with sample data
        sample_data = {
            "season": 1, "yr": 1, "mnth": 6, "hr": 8,
            "holiday": 0, "weekday": 1, "workingday": 1,
            "weathersit": 1, "temp": 0.5, "atemp": 0.48,
            "hum": 0.6, "windspeed": 0.2
        }
        
        prediction = predictor.predict_single(**sample_data)
        
        # Basic health checks
        if prediction < 0 or prediction > 1000:
            raise ValueError(f"Prediction out of expected range: {prediction}")
        
        health_status = {
            'model_loaded': predictor.is_loaded,
            'test_prediction': prediction,
            'model_name': predictor.model_name,
            'model_stage': predictor.model_stage,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Model health check passed: {health_status}")
        return health_status
        
    except Exception as e:
        logger.error(f"Model health check failed: {e}")
        raise


def analyze_prediction_metrics(**context):
    """Analyze recent prediction metrics and performance."""
    logger.info("Analyzing prediction metrics...")
    
    monitor = DataDriftMonitor()
    
    # Get recent predictions
    recent_predictions = monitor.get_recent_predictions(hours=24)
    
    if recent_predictions.empty:
        logger.warning("No recent predictions found")
        return {'status': 'no_data', 'predictions_count': 0}
    
    # Calculate metrics
    predictions_count = len(recent_predictions)
    avg_prediction = recent_predictions['prediction'].mean()
    std_prediction = recent_predictions['prediction'].std()
    
    # Analyze prediction patterns
    hourly_stats = recent_predictions.groupby('hr')['prediction'].agg(['count', 'mean', 'std'])
    peak_hour = hourly_stats['mean'].idxmax()
    peak_demand = hourly_stats['mean'].max()
    
    metrics = {
        'predictions_count': predictions_count,
        'avg_prediction': float(avg_prediction),
        'std_prediction': float(std_prediction),
        'peak_hour': int(peak_hour),
        'peak_demand': float(peak_demand),
        'data_coverage_hours': len(hourly_stats)
    }
    
    # Check for anomalies
    anomaly_threshold = avg_prediction + 3 * std_prediction
    anomalies = recent_predictions[recent_predictions['prediction'] > anomaly_threshold]
    
    if len(anomalies) > 0:
        metrics['anomalies_detected'] = len(anomalies)
        logger.warning(f"Detected {len(anomalies)} prediction anomalies")
    
    logger.info(f"Prediction metrics analysis: {metrics}")
    return metrics


def detect_data_drift(**context):
    """Detect data drift in recent predictions."""
    logger.info("Detecting data drift...")
    
    monitor = DataDriftMonitor()
    
    # Get recent data for drift analysis
    recent_data = monitor.get_recent_predictions(hours=24)
    
    if len(recent_data) < 30:
        logger.warning(f"Insufficient data for drift analysis: {len(recent_data)} records")
        return {'status': 'insufficient_data', 'records_count': len(recent_data)}
    
    # Prepare feature data
    feature_columns = [
        "season", "yr", "mnth", "hr", "holiday", "weekday",
        "workingday", "weathersit", "temp", "atemp", "hum", "windspeed"
    ]
    
    current_features = recent_data[feature_columns]
    
    try:
        # Run drift detection
        drift_results = monitor.detect_data_drift(current_features)
        
        # Run additional tests
        test_results = monitor.run_drift_tests(current_features)
        
        combined_results = {
            **drift_results,
            'tests_passed': test_results['tests_passed'],
            'records_analyzed': len(recent_data)
        }
        
        logger.info(f"Drift detection completed: {combined_results}")
        return combined_results
        
    except Exception as e:
        logger.error(f"Error in drift detection: {e}")
        return {'status': 'error', 'error': str(e)}


def check_performance_degradation(**context):
    """Check for model performance degradation."""
    logger.info("Checking for performance degradation...")
    
    monitor = DataDriftMonitor()
    
    # Get recent predictions with actuals (if available)
    recent_data = monitor.get_recent_predictions(hours=168)  # Last week
    
    # Filter data where we have actual values
    data_with_actuals = recent_data.dropna(subset=['actual'])
    
    if len(data_with_actuals) < 10:
        logger.warning("Insufficient data with actual values for performance analysis")
        return {'status': 'insufficient_data', 'records_with_actuals': len(data_with_actuals)}
    
    # Calculate performance metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import numpy as np
    
    y_true = data_with_actuals['actual']
    y_pred = data_with_actuals['prediction']
    
    current_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    current_mae = mean_absolute_error(y_true, y_pred)
    current_r2 = r2_score(y_true, y_pred)
    
    # Historical thresholds (would be stored from training)
    baseline_rmse = 50.0  # Example baseline
    degradation_threshold = 1.2  # 20% degradation
    
    performance_metrics = {
        'current_rmse': float(current_rmse),
        'current_mae': float(current_mae),
        'current_r2': float(current_r2),
        'baseline_rmse': baseline_rmse,
        'degradation_ratio': float(current_rmse / baseline_rmse),
        'records_evaluated': len(data_with_actuals)
    }
    
    # Check for degradation
    if current_rmse > baseline_rmse * degradation_threshold:
        performance_metrics['degradation_detected'] = True
        logger.warning(f"Performance degradation detected: RMSE {current_rmse:.2f} vs baseline {baseline_rmse:.2f}")
    else:
        performance_metrics['degradation_detected'] = False
    
    logger.info(f"Performance check completed: {performance_metrics}")
    return performance_metrics


def generate_monitoring_alerts(**context):
    """Generate alerts based on monitoring results."""
    logger.info("Generating monitoring alerts...")
    
    # Get results from previous tasks
    health_check = context['task_instance'].xcom_pull(task_ids='check_model_health')
    prediction_metrics = context['task_instance'].xcom_pull(task_ids='analyze_prediction_metrics')
    drift_results = context['task_instance'].xcom_pull(task_ids='detect_data_drift')
    performance_check = context['task_instance'].xcom_pull(task_ids='check_performance_degradation')
    
    alerts = []
    
    # Model health alerts
    if not health_check.get('model_loaded', False):
        alerts.append({
            'severity': 'critical',
            'message': 'Model is not loaded or not responding',
            'component': 'model_health'
        })
    
    # Data drift alerts
    if drift_results.get('drift_detected', False):
        alerts.append({
            'severity': 'warning',
            'message': f"Data drift detected in {len(drift_results.get('drifted_columns', []))} features",
            'component': 'data_drift',
            'details': drift_results
        })
    
    # Performance degradation alerts
    if performance_check.get('degradation_detected', False):
        alerts.append({
            'severity': 'warning',
            'message': f"Model performance degradation detected: RMSE {performance_check['current_rmse']:.2f}",
            'component': 'performance',
            'details': performance_check
        })
    
    # Anomaly alerts
    if prediction_metrics.get('anomalies_detected', 0) > 5:
        alerts.append({
            'severity': 'warning',
            'message': f"High number of prediction anomalies: {prediction_metrics['anomalies_detected']}",
            'component': 'predictions'
        })
    
    alert_summary = {
        'alerts_count': len(alerts),
        'alerts': alerts,
        'timestamp': datetime.now().isoformat()
    }
    
    if alerts:
        logger.warning(f"Generated {len(alerts)} alerts: {[alert['message'] for alert in alerts]}")
        # In a real system, send alerts via Slack, email, PagerDuty, etc.
    else:
        logger.info("No alerts generated - all systems healthy")
    
    return alert_summary


def update_monitoring_dashboard(**context):
    """Update monitoring dashboard with latest metrics."""
    logger.info("Updating monitoring dashboard...")
    
    # Get all monitoring results
    tasks_results = {
        'health_check': context['task_instance'].xcom_pull(task_ids='check_model_health'),
        'prediction_metrics': context['task_instance'].xcom_pull(task_ids='analyze_prediction_metrics'),
        'drift_results': context['task_instance'].xcom_pull(task_ids='detect_data_drift'),
        'performance_check': context['task_instance'].xcom_pull(task_ids='check_performance_degradation'),
        'alerts': context['task_instance'].xcom_pull(task_ids='generate_monitoring_alerts')
    }
    
    # In a real system, this would:
    # 1. Update Grafana dashboards
    # 2. Send metrics to Prometheus
    # 3. Update monitoring database
    # 4. Generate reports
    
    dashboard_update = {
        'dashboard_updated': True,
        'timestamp': datetime.now().isoformat(),
        'metrics_summary': {
            'model_healthy': tasks_results['health_check'].get('model_loaded', False),
            'predictions_last_24h': tasks_results['prediction_metrics'].get('predictions_count', 0),
            'drift_detected': tasks_results['drift_results'].get('drift_detected', False),
            'alerts_count': tasks_results['alerts'].get('alerts_count', 0)
        }
    }
    
    logger.info(f"Dashboard update completed: {dashboard_update}")
    return dashboard_update


# Task definitions
model_health_task = PythonOperator(
    task_id='check_model_health',
    python_callable=check_model_health,
    dag=dag,
)

prediction_metrics_task = PythonOperator(
    task_id='analyze_prediction_metrics',
    python_callable=analyze_prediction_metrics,
    dag=dag,
)

drift_detection_task = PythonOperator(
    task_id='detect_data_drift',
    python_callable=detect_data_drift,
    dag=dag,
)

performance_check_task = PythonOperator(
    task_id='check_performance_degradation',
    python_callable=check_performance_degradation,
    dag=dag,
)

alerts_task = PythonOperator(
    task_id='generate_monitoring_alerts',
    python_callable=generate_monitoring_alerts,
    dag=dag,
)

dashboard_update_task = PythonOperator(
    task_id='update_monitoring_dashboard',
    python_callable=update_monitoring_dashboard,
    dag=dag,
)

# Define task dependencies
[model_health_task, prediction_metrics_task, drift_detection_task, performance_check_task] >> alerts_task
alerts_task >> dashboard_update_task