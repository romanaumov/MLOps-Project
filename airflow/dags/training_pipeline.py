from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.sensors.filesystem import FileSensor
import pandas as pd
import logging
import sys
import os

# Add src to path
sys.path.append('/opt/airflow/dags/src')

from src.data.preprocessing import DataPreprocessor
from src.models.train import ModelTrainer
from src.monitoring.data_drift import DataDriftMonitor

logger = logging.getLogger(__name__)

# Default arguments for the DAG
default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Create DAG
dag = DAG(
    'bike_sharing_training_pipeline',
    default_args=default_args,
    description='End-to-end training pipeline for bike sharing demand prediction',
    schedule_interval='@weekly',  # Run weekly
    catchup=False,
    tags=['ml', 'training', 'bike-sharing'],
    max_active_runs=1,
)


def check_data_quality(**context):
    """Check data quality before training."""
    logger.info("Checking data quality...")
    
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data()
    
    # Basic quality checks
    total_rows = len(df)
    missing_values = df.isnull().sum().sum()
    missing_percentage = (missing_values / (total_rows * len(df.columns))) * 100
    
    # Check for minimum data requirements
    if total_rows < 1000:
        raise ValueError(f"Insufficient data: {total_rows} rows (minimum 1000 required)")
    
    if missing_percentage > 10:
        raise ValueError(f"Too many missing values: {missing_percentage:.2f}% (maximum 10% allowed)")
    
    # Check target variable distribution
    target_stats = df['cnt'].describe()
    if target_stats['std'] == 0:
        raise ValueError("Target variable has no variance")
    
    logger.info(f"Data quality check passed: {total_rows} rows, {missing_percentage:.2f}% missing")
    
    # Push metrics to XCom
    return {
        'total_rows': total_rows,
        'missing_percentage': missing_percentage,
        'target_mean': float(target_stats['mean']),
        'target_std': float(target_stats['std'])
    }


def preprocess_data(**context):
    """Preprocess data for training."""
    logger.info("Starting data preprocessing...")
    
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline()
    
    # Log preprocessing results
    preprocessing_info = {
        'train_shape': X_train.shape,
        'test_shape': X_test.shape,
        'feature_count': X_train.shape[1],
        'feature_names': list(X_train.columns)
    }
    
    logger.info(f"Preprocessing completed: {preprocessing_info}")
    return preprocessing_info


def train_models(**context):
    """Train multiple models and select the best one."""
    logger.info("Starting model training...")
    
    trainer = ModelTrainer()
    results = trainer.train_all_models()
    
    # Find best model
    best_model_name = None
    best_rmse = float('inf')
    
    for model_name, result in results.items():
        test_rmse = result['metrics']['test_rmse']
        if test_rmse < best_rmse:
            best_rmse = test_rmse
            best_model_name = model_name
    
    training_summary = {
        'models_trained': list(results.keys()),
        'best_model': best_model_name,
        'best_rmse': best_rmse,
        'all_results': {name: result['metrics'] for name, result in results.items()}
    }
    
    logger.info(f"Training completed. Best model: {best_model_name} (RMSE: {best_rmse:.4f})")
    return training_summary


def validate_model_performance(**context):
    """Validate that the trained model meets performance criteria."""
    logger.info("Validating model performance...")
    
    # Get training results from previous task
    training_summary = context['task_instance'].xcom_pull(task_ids='train_models')
    best_rmse = training_summary['best_rmse']
    
    # Performance thresholds
    max_allowed_rmse = 100.0  # Maximum allowed RMSE
    min_r2_score = 0.7  # Minimum R² score
    
    if best_rmse > max_allowed_rmse:
        raise ValueError(f"Model performance below threshold: RMSE {best_rmse:.4f} > {max_allowed_rmse}")
    
    # Additional validation could include:
    # - Cross-validation scores
    # - Feature importance analysis
    # - Prediction distribution checks
    
    logger.info(f"Model validation passed: RMSE {best_rmse:.4f}")
    return {'validation_passed': True, 'rmse': best_rmse}


def run_data_drift_analysis(**context):
    """Run data drift analysis on recent data."""
    logger.info("Running data drift analysis...")
    
    try:
        monitor = DataDriftMonitor()
        
        # Get recent predictions for drift analysis
        recent_data = monitor.get_recent_predictions(hours=168)  # Last week
        
        if len(recent_data) < 100:
            logger.warning(f"Insufficient recent data for drift analysis: {len(recent_data)} records")
            return {'drift_analysis': 'skipped', 'reason': 'insufficient_data'}
        
        # Run drift check
        drift_results = monitor.check_and_alert(hours_window=168)
        
        logger.info(f"Drift analysis completed: {drift_results['status']}")
        return drift_results
        
    except Exception as e:
        logger.error(f"Error in drift analysis: {e}")
        return {'drift_analysis': 'failed', 'error': str(e)}


def deploy_model(**context):
    """Deploy the trained model (placeholder for actual deployment)."""
    logger.info("Deploying model...")
    
    # Get training results
    training_summary = context['task_instance'].xcom_pull(task_ids='train_models')
    validation_result = context['task_instance'].xcom_pull(task_ids='validate_model_performance')
    
    if not validation_result['validation_passed']:
        raise ValueError("Cannot deploy model: validation failed")
    
    # In a real deployment, this would:
    # 1. Transition model to Production stage in MLflow
    # 2. Update model serving endpoints
    # 3. Run smoke tests
    # 4. Update monitoring dashboards
    
    deployment_info = {
        'model_name': training_summary['best_model'],
        'deployment_time': datetime.now().isoformat(),
        'rmse': validation_result['rmse']
    }
    
    logger.info(f"Model deployment completed: {deployment_info}")
    return deployment_info


def send_training_notification(**context):
    """Send email notification about training completion."""
    logger.info("Sending training notification...")
    
    try:
        # Get results from previous tasks
        training_summary = context['task_instance'].xcom_pull(task_ids='train_models')
        validation_result = context['task_instance'].xcom_pull(task_ids='validate_model_performance')
        deployment_info = context['task_instance'].xcom_pull(task_ids='deploy_model')
        
        # Import email notification utility
        import subprocess
        import json
        
        # Create notification data
        notification_data = {
            'training_summary': training_summary,
            'validation_result': validation_result,
            'deployment_info': deployment_info
        }
        
        # Send email notification using script
        result = subprocess.run([
            'python', '/opt/airflow/dags/scripts/send_training_email.py',
            '--data', json.dumps(notification_data)
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("Training notification email sent successfully")
            return {'notification_sent': True}
        else:
            logger.error(f"Failed to send email notification: {result.stderr}")
            return {'notification_sent': False}
            
    except Exception as e:
        logger.error(f"Error sending training notification: {e}")
        return {'notification_sent': False}


# Task definitions
check_data_quality_task = PythonOperator(
    task_id='check_data_quality',
    python_callable=check_data_quality,
    dag=dag,
)

preprocess_data_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    dag=dag,
)

train_models_task = PythonOperator(
    task_id='train_models',
    python_callable=train_models,
    dag=dag,
)

validate_model_task = PythonOperator(
    task_id='validate_model_performance',
    python_callable=validate_model_performance,
    dag=dag,
)

drift_analysis_task = PythonOperator(
    task_id='run_data_drift_analysis',
    python_callable=run_data_drift_analysis,
    dag=dag,
)

deploy_model_task = PythonOperator(
    task_id='deploy_model',
    python_callable=deploy_model,
    dag=dag,
)

notification_task = PythonOperator(
    task_id='send_training_notification',
    python_callable=send_training_notification,
    dag=dag,
)

# Define task dependencies
check_data_quality_task >> preprocess_data_task >> train_models_task
train_models_task >> validate_model_task >> deploy_model_task
train_models_task >> drift_analysis_task
[deploy_model_task, drift_analysis_task] >> notification_task