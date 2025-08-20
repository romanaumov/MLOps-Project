#!/usr/bin/env python3
"""
Setup script for initializing the MLOps project environment.
"""

import os
import sys
import subprocess
from pathlib import Path


def run_command(command: str, check: bool = True) -> None:
    """Run a shell command."""
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, check=check)
    if result.returncode != 0 and check:
        print(f"Command failed: {command}")
        sys.exit(1)


def create_directories() -> None:
    """Create necessary directories."""
    directories = [
        "data/processed",
        "models",
        "logs",
        "monitoring/reports",
        "mlruns",
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")


def setup_environment() -> None:
    """Setup the development environment."""
    print("Setting up MLOps project environment...")
    
    # Create directories
    create_directories()
    
    # Copy environment file if it doesn't exist
    if not Path(".env").exists() and Path(".env.example").exists():
        run_command("cp .env.example .env")
        print("Created .env file from .env.example")
    
    # Install Python dependencies
    print("Installing Python dependencies...")
    run_command("uv sync --dev")
    
    # Install pre-commit hooks
    print("Installing pre-commit hooks...")
    run_command("uv run pre-commit install")
    
    # Initialize git repository if not already initialized
    if not Path(".git").exists():
        print("Initializing git repository...")
        run_command("git init")
        run_command("git add .")
        run_command("git commit -m 'Initial commit'")
    
    print("\n✅ Setup completed successfully!")
    print("\nNext steps:")
    print("1. Train your first model: make train-model")
    print("2. Start services: make docker-up")
    print("3. Visit the API docs: http://localhost:8000/docs")


if __name__ == "__main__":
    setup_environment()