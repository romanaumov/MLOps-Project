#!/usr/bin/env python3
"""
Validation script to check if the MLOps project is properly set up.
"""

import sys
import importlib
import subprocess
from pathlib import Path


def check_python_version():
    """Check Python version."""
    print("Checking Python version...")
    if sys.version_info < (3, 11):
        print("❌ Python 3.11+ required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}")
    return True


def check_dependencies():
    """Check if required dependencies are installed."""
    print("Checking dependencies...")
    required_packages = [
        'pandas', 'numpy', 'sklearn', 'mlflow', 'evidently',
        'fastapi', 'uvicorn', 'airflow', 'plotly', 'pytest'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            if package == 'sklearn':
                importlib.import_module('sklearn')
            else:
                importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing packages: {missing_packages}")
        print("Run: uv sync --dev")
        return False
    
    return True


def check_directories():
    """Check if required directories exist."""
    print("Checking directories...")
    required_dirs = [
        'src', 'data/raw', 'data/processed', 'models', 'tests',
        'airflow/dags', 'docker', 'monitoring', '.github/workflows'
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✅ {dir_path}")
        else:
            print(f"❌ {dir_path}")
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print(f"Missing directories: {missing_dirs}")
        return False
    
    return True


def check_data():
    """Check if data files exist."""
    print("Checking data files...")
    data_files = ['data/raw/hour.csv', 'data/raw/day.csv']
    
    for file_path in data_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
            return False
    
    return True


def check_docker():
    """Check if Docker is available."""
    print("Checking Docker...")
    try:
        result = subprocess.run(['docker', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Docker: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass
    
    print("❌ Docker not found or not working")
    return False


def check_git():
    """Check if Git is available."""
    print("Checking Git...")
    try:
        result = subprocess.run(['git', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Git: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass
    
    print("❌ Git not found")
    return False


def main():
    """Run all validation checks."""
    print("🔍 Validating MLOps Project Setup\n")
    
    checks = [
        check_python_version,
        check_dependencies,
        check_directories,
        check_data,
        check_docker,
        check_git
    ]
    
    passed = 0
    total = len(checks)
    
    for check in checks:
        if check():
            passed += 1
        print()
    
    print(f"📊 Results: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 All checks passed! Your MLOps project is ready to go.")
        print("\nNext steps:")
        print("1. Train a model: make train-model")
        print("2. Start services: make docker-up")
        print("3. Make predictions: curl -X POST http://localhost:8000/predict ...")
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        sys.exit(1)


if __name__ == "__main__":
    main()