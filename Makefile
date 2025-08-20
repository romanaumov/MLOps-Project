.PHONY: help install install-dev setup clean test lint format type-check pre-commit
.PHONY: run-mlflow run-airflow run-api run-monitoring docker-build docker-up docker-down
.PHONY: train-model predict deploy-model

# Default target
help:
	@echo "Available commands:"
	@echo "  install       Install production dependencies"
	@echo "  install-dev   Install development dependencies"
	@echo "  setup         Setup development environment"
	@echo "  clean         Clean temporary files"
	@echo "  test          Run tests"
	@echo "  lint          Run linting"
	@echo "  format        Format code"
	@echo "  type-check    Run type checking"
	@echo "  pre-commit    Run pre-commit hooks"
	@echo "  run-mlflow    Start MLflow server"
	@echo "  run-airflow   Start Airflow"
	@echo "  run-api       Start prediction API"
	@echo "  run-monitoring Start monitoring services"
	@echo "  docker-build  Build Docker images"
	@echo "  docker-up     Start all services with Docker Compose"
	@echo "  docker-down   Stop all services"
	@echo "  train-model   Train model with MLflow tracking"
	@echo "  predict       Make predictions using trained model"
	@echo "  deploy-model  Deploy model to production"

# Installation
install:
	uv sync

install-dev:
	uv sync --dev

setup: install-dev
	uv run pre-commit install
	mkdir -p data/raw data/processed models logs
	cp .env.example .env

# Cleaning
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info

# Testing and Quality
test:
	uv run pytest

lint:
	uv run flake8 src tests
	uv run black --check src tests
	uv run isort --check-only src tests

format:
	uv run black src tests
	uv run isort src tests

type-check:
	uv run mypy src

pre-commit:
	uv run pre-commit run --all-files

# Services
run-mlflow:
	uv run mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

run-airflow:
	uv run airflow webserver --port 8080 &
	uv run airflow scheduler

run-api:
	uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

run-monitoring:
	docker-compose up -d grafana prometheus

# Docker
docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

# ML Pipeline
train-model:
	uv run python src/models/train.py

predict:
	uv run python src/models/predict.py

deploy-model:
	uv run python scripts/deploy_model.py