FROM apache/airflow:2.10.5

USER root

RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# Install required packages for our MLOps pipeline
RUN pip install --no-cache-dir \
    pandas>=2.0.0 \
    numpy>=1.24.0 \
    scikit-learn>=1.3.0 \
    mlflow>=2.8.0 \
    evidently==0.6.7 \
    psutil>=5.9.0 \
    pydantic>=2.0.0 \
    python-dotenv>=1.0.0 \
    pydantic-settings>=2.10.1 \
    requests>=2.31.0 \
    apache-airflow-providers-postgres \
    apache-airflow-providers-celery