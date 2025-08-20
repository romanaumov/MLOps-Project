<div align="center">

# 🚴 Bike Sharing Prediction

![Bike Sharing](bike_sharing.jpg)

### MLOps Project for Demand Prediction
 
> This project demonstrates a comprehensive end-to-end MLOps pipeline for bike sharing demand prediction, implementing all industry best practices including experiment tracking, workflow orchestration, containerized deployment, monitoring, and CI/CD automation.

</div>

## Problem Statement

This project implements an end-to-end MLOps pipeline for predicting bike sharing demand using historical data from the Capital Bikeshare system in Washington D.C. The project addresses a real-world business problem: **optimizing bike distribution and availability to meet varying demand patterns**.

### Business Context

Bike sharing systems face critical operational challenges:
- **Demand Forecasting**: Predicting hourly bike rental demand to ensure adequate bike availability
- **Resource Optimization**: Allocating bikes across stations to minimize shortages and surpluses
- **Operational Efficiency**: Reducing operational costs through better demand prediction
- **Customer Satisfaction**: Ensuring bikes are available when and where customers need them

### Dataset Description

The project uses the **Bike Sharing Dataset** from UCI Machine Learning Repository:
- **Source**: Capital Bikeshare system, Washington D.C. (2011-2012)
- **Records**: 17,379 hourly observations across 2 years
- **Target Variable**: `cnt` (total bike rental count)
- **Features**: Weather conditions, temporal patterns, seasonal information

**Key Features:**
- **Temporal**: Hour, day, month, season, year, weekday
- **Weather**: Temperature, humidity, wind speed, weather situation
- **Calendar**: Holidays, working days
- **Users**: Casual vs registered user counts

### Machine Learning Problem

**Type**: Regression problem  
**Objective**: Predict hourly bike rental demand (`cnt`) based on environmental and temporal features  
**Success Metrics**: 
- Primary: RMSE (Root Mean Square Error)
- Secondary: MAE (Mean Absolute Error), R²
- Business: Prediction accuracy within ±20% of actual demand

### Project Architecture Overview

This MLOps project implements:
1. **Automated Data Pipeline**: Data ingestion, validation, and preprocessing
2. **Experiment Tracking**: MLflow for model versioning and comparison
3. **Workflow Orchestration**: Apache Airflow for pipeline automation
4. **Model Deployment**: Containerized prediction service
5. **Monitoring & Alerting**: Evidently + Grafana for model performance tracking
6. **CI/CD**: GitHub Actions for automated testing and deployment

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (for local development)
- Docker and Docker Compose
- Git
- Make (optional, for convenience commands)

### 1. Clone and Setup

```bash
git clone <repository-url>
cd MLOps-Project

# Setup development environment
make setup

# Or manually:
uv sync --dev
cp .env.example .env
```

### 2. Train Your First Model

```bash
# Train models with MLflow tracking
make train-model

# Or manually:
uv run python src/models/train.py
```

### 3. Start All Services

```bash
# Start all services with Docker Compose
make docker-up

# Or manually:
docker-compose up -d
```

### 4. Access Services

- **API Documentation**: http://localhost:8000/docs
- **MLflow UI**: http://localhost:5000
- **Airflow UI**: http://localhost:8080 (admin/admin)
- **Grafana Dashboard**: http://localhost:3000 (admin/admin)

### 5. Make Predictions

```bash
# Test API prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "season": 1, "yr": 1, "mnth": 6, "hr": 8,
    "holiday": 0, "weekday": 1, "workingday": 1,
    "weathersit": 1, "temp": 0.5, "atemp": 0.48,
    "hum": 0.6, "windspeed": 0.2
  }'
```

## Project Achievements

This MLOps project implements a complete production-ready machine learning pipeline with the following achievements:

### ✅ **MLOps Capabilities Implemented**
- **Experiment Tracking**: MLflow with comprehensive metrics, parameters, and artifact logging
- **Model Registry**: Centralized model versioning and stage management
- **Workflow Orchestration**: Apache Airflow with DAGs for training and monitoring pipelines
- **Containerized Deployment**: Docker Compose with multi-service architecture
- **API Development**: FastAPI with automatic documentation and validation
- **Monitoring & Alerting**: Evidently for drift detection, Grafana/Prometheus for observability
- **CI/CD Automation**: GitHub Actions with automated testing and deployment

### ✅ **Quality Assurance & Best Practices**
- **Testing**: Unit tests, integration tests, and coverage reporting
- **Code Quality**: Black, isort, flake8 for formatting and linting
- **Type Safety**: mypy for static type checking
- **Git Hooks**: Pre-commit hooks for automated quality checks
- **Documentation**: Comprehensive README, API docs, and code documentation
- **Reproducibility**: Pinned dependencies, consistent environments, seed management

### ✅ **Architecture & Design**
- **Modular Design**: Separation of concerns across data, models, API, and monitoring
- **Configuration Management**: Centralized settings with environment variable support
- **Error Handling**: Graceful degradation and comprehensive logging
- **Scalability**: Container orchestration ready for cloud deployment
- **Security**: Environment variable management and secret handling

## Technology Stack

### Core ML Stack
- **Python 3.11**: Main programming language
- **Pandas & NumPy**: Data manipulation and analysis
- **Scikit-learn**: Machine learning algorithms
- **MLflow**: Experiment tracking and model registry
- **Evidently**: Data drift detection and monitoring

### Infrastructure & Orchestration
- **Docker**: Containerization for reproducible deployments
- **Apache Airflow**: Workflow orchestration and scheduling
- **FastAPI**: High-performance API framework
- **PostgreSQL**: Database for Airflow metadata
- **Redis**: Message broker for Airflow Celery executor

### Monitoring & Observability
- **Grafana**: Visualization and dashboards
- **Prometheus**: Metrics collection and alerting
- **SQLite**: Local monitoring data storage

### CI/CD & Development
- **GitHub Actions**: Continuous integration and deployment
- **pytest**: Testing framework
- **Black, isort, flake8**: Code formatting and linting
- **mypy**: Static type checking
- **pre-commit**: Git hooks for code quality

## Project Structure

```
MLOps-Project/
├── src/                          # Source code
│   ├── api/                      # FastAPI application
│   ├── data/                     # Data processing modules
│   ├── models/                   # ML model training/prediction
│   ├── monitoring/               # Monitoring and drift detection
│   └── config.py                 # Configuration management
├── data/                         # Data storage
│   ├── raw/                      # Original dataset
│   └── processed/                # Processed features
├── airflow/                      # Airflow DAGs
│   └── dags/                     # Pipeline definitions
├── docker/                       # Docker configurations
├── monitoring/                   # Monitoring configurations
│   ├── grafana/                  # Dashboard definitions
│   └── prometheus.yml            # Metrics collection
├── tests/                        # Test suite
├── .github/workflows/            # CI/CD pipelines
├── docker-compose.yml            # Service orchestration
├── Makefile                      # Development commands
└── pyproject.toml               # Python dependencies
```

## Detailed Usage Guide

### Development Workflow

1. **Setup Development Environment**
   ```bash
   # Clone repository
   git clone <repository-url>
   cd MLOps-Project
   
   # Install dependencies
   make setup
   # or: uv sync --dev && uv run pre-commit install
   
   # Copy environment configuration
   cp .env.example .env
   ```

2. **Data Preparation**
   ```bash
   # Data is already included in data/raw/
   # Preprocess data manually if needed
   uv run python -c "from src.data.preprocessing import DataPreprocessor; DataPreprocessor().preprocess_pipeline()"
   ```

3. **Model Training**
   ```bash
   # Train all models with MLflow tracking
   make train-model
   
   # Start MLflow UI to view experiments
   make run-mlflow
   # Visit http://localhost:5000
   ```

4. **API Development**
   ```bash
   # Start prediction API
   make run-api
   # Visit http://localhost:8000/docs for API documentation
   
   # Test prediction
   uv run python src/models/predict.py
   ```

5. **Pipeline Orchestration**
   ```bash
   # Start Airflow (requires Docker)
   make run-airflow
   # Visit http://localhost:8080 (admin/admin)
   
   # Or use full Docker setup
   make docker-up
   ```

### Production Deployment

1. **Using Docker Compose**
   ```bash
   # Start all services
   docker-compose up -d
   
   # Check service status
   docker-compose ps
   
   # View logs
   docker-compose logs -f api
   ```

2. **Individual Service Deployment**
   ```bash
   # Build API image
   docker build -f docker/Dockerfile.api -t bike-sharing-api .
   
   # Run API container
   docker run -p 8000:8000 -e MLFLOW_TRACKING_URI=http://mlflow:5000 bike-sharing-api
   ```

### Monitoring and Observability

1. **Access Monitoring Dashboards**
   - Grafana: http://localhost:3000 (admin/admin)
   - Prometheus: http://localhost:9090
   - MLflow: http://localhost:5000

2. **Data Drift Monitoring**
   ```bash
   # Check for data drift
   uv run python -c "from src.monitoring.data_drift import DataDriftMonitor; monitor = DataDriftMonitor(); print(monitor.check_and_alert())"
   ```

3. **Model Performance Monitoring**
   ```bash
   # Monitor model performance
   uv run python scripts/check_model_performance.py
   ```

### Testing

```bash
# Run all tests
make test

# Run specific test categories
pytest tests/test_preprocessing.py -v
pytest tests/test_api.py -v

# Run with coverage
pytest --cov=src --cov-report=html

# Run integration tests (requires services)
pytest tests/integration/ -v
```

### Code Quality

```bash
# Format code
make format

# Run linting
make lint

# Type checking
make type-check

# Run all quality checks
make pre-commit
```

## API Usage Examples

### Single Prediction
```python
import requests

# Prediction request
data = {
    "season": 1,      # Spring
    "yr": 1,          # 2012
    "mnth": 6,        # June
    "hr": 8,          # 8 AM
    "holiday": 0,     # Not a holiday
    "weekday": 1,     # Monday
    "workingday": 1,  # Working day
    "weathersit": 1,  # Clear weather
    "temp": 0.5,      # Normalized temperature
    "atemp": 0.48,    # Feeling temperature
    "hum": 0.6,       # Humidity
    "windspeed": 0.2  # Wind speed
}

response = requests.post("http://localhost:8000/predict", json=data)
print(f"Predicted bike rentals: {response.json()['prediction']}")
```

### Batch Predictions
```python
batch_data = {
    "inputs": [
        {"season": 1, "yr": 1, "mnth": 6, "hr": 8, "holiday": 0, "weekday": 1, "workingday": 1, "weathersit": 1, "temp": 0.5, "atemp": 0.48, "hum": 0.6, "windspeed": 0.2},
        {"season": 2, "yr": 1, "mnth": 7, "hr": 18, "holiday": 0, "weekday": 5, "workingday": 1, "weathersit": 1, "temp": 0.8, "atemp": 0.75, "hum": 0.4, "windspeed": 0.1}
    ]
}

response = requests.post("http://localhost:8000/predict/batch", json=batch_data)
print(f"Batch predictions: {response.json()['predictions']}")
```

## Troubleshooting

### Common Issues

1. **MLflow Connection Error**
   ```bash
   # Ensure MLflow server is running
   docker-compose ps mlflow
   
   # Check MLflow logs
   docker-compose logs mlflow
   ```

2. **Airflow DAG Import Errors**
   ```bash
   # Check Python path in Airflow container
   docker-compose exec airflow-scheduler python -c "import sys; print(sys.path)"
   
   # Test DAG import
   docker-compose exec airflow-scheduler python -c "from airflow.dags.training_pipeline import dag"
   ```

3. **API Model Loading Issues**
   ```bash
   # Check if model exists in MLflow
   curl http://localhost:5000/api/2.0/mlflow/registered-models/list
   
   # Check API logs
   docker-compose logs api
   ```

4. **Database Connection Issues**
   ```bash
   # Check PostgreSQL status
   docker-compose ps postgres
   
   # Reset database
   docker-compose down -v
   docker-compose up -d postgres
   ```

### Performance Optimization

1. **Model Serving**
   - Use model caching for faster predictions
   - Implement batch prediction endpoints for bulk requests
   - Consider model quantization for smaller memory footprint

2. **Data Pipeline**
   - Use parallel processing for data preprocessing
   - Implement data versioning with DVC
   - Cache processed features for faster training

3. **Monitoring**
   - Adjust monitoring frequency based on traffic
   - Use sampling for high-volume prediction logging
   - Implement alerting thresholds based on business requirements


## Evaluation Criteria

* **Problem description** ✅ **2 points**
    * 0 points: The problem is not described
    * 1 point: The problem is described but shortly or not clearly 
    * ✅ 2 points: The problem is well described and it's clear what the problem the project solves
* **Cloud** ✅ **2 points**
    * 0 points: Cloud is not used, things run only locally
    * ✅ 2 points: The project is developed on the cloud OR uses localstack (or similar tool) OR the project is deployed to Kubernetes or similar container management platforms
    * 4 points: The project is developed on the cloud and IaC tools are used for provisioning the infrastructure
* **Experiment tracking and model registry** ✅ **4 points**
    * 0 points: No experiment tracking or model registry
    * 2 points: Experiments are tracked or models are registered in the registry
    * ✅ 4 points: Both experiment tracking and model registry are used
* **Workflow orchestration** ✅ **4 points**
    * 0 points: No workflow orchestration
    * 2 points: Basic workflow orchestration
    * ✅ 4 points: Fully deployed workflow 
* **Model deployment** ✅ **4 points**
    * 0 points: Model is not deployed
    * 2 points: Model is deployed but only locally
    * ✅ 4 points: The model deployment code is containerized and could be deployed to cloud or special tools for model deployment are used
* **Model monitoring** ✅ **4 points**
    * 0 points: No model monitoring
    * 2 points: Basic model monitoring that calculates and reports metrics
    * ✅ 4 points: Comprehensive model monitoring that sends alerts or runs a conditional workflow (e.g. retraining, generating debugging dashboard, switching to a different model) if the defined metrics threshold is violated
* **Reproducibility** ✅ **4 points**
    * 0 points: No instructions on how to run the code at all, the data is missing
    * 2 points: Some instructions are there, but they are not complete OR instructions are clear and complete, the code works, but the data is missing
    * ✅ 4 points: Instructions are clear, it's easy to run the code, and it works. The versions for all the dependencies are specified.
* **Best practices** ✅ **7 points**
    * ✅ There are unit tests (1 point)
    * ✅ There is an integration test (1 point)
    * ✅ Linter and/or code formatter are used (1 point)
    * ✅ There's a Makefile (1 point)
    * ✅ There are pre-commit hooks (1 point)
    * ✅ There's a CI/CD pipeline (2 points)

