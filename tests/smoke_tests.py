#!/usr/bin/env python3
"""
Smoke tests for the deployed application.

These tests verify basic functionality of the deployed API
to ensure the deployment was successful.
"""

import os
import sys
import requests
import logging
import time
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SmokeTestRunner:
    """Smoke test runner for deployed API."""
    
    def __init__(self, base_url=None, timeout=30):
        # Get base URL from parameter, environment, or default
        if base_url:
            self.base_url = base_url
        else:
            env_url = os.getenv('API_BASE_URL', '').strip()
            if env_url and not env_url.startswith(('http://', 'https://')):
                # If URL doesn't have scheme, assume http
                env_url = f'http://{env_url}'
            self.base_url = env_url or 'http://localhost:8000'
        
        # Ensure base_url doesn't end with slash
        self.base_url = self.base_url.rstrip('/')
        
        self.timeout = timeout
        self.session = requests.Session()
        
    def test_health_endpoint(self):
        """Test that the health endpoint is responding."""
        try:
            logger.info(f"Testing health endpoint: {self.base_url}/health")
            response = self.session.get(f"{self.base_url}/health", timeout=self.timeout)
            
            if response.status_code != 200:
                raise AssertionError(f"Health check failed: {response.status_code}")
            
            logger.info("✅ Health endpoint test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Health endpoint test failed: {e}")
            return False
    
    def test_prediction_endpoint(self):
        """Test that the prediction endpoint is working."""
        try:
            logger.info(f"Testing prediction endpoint: {self.base_url}/predict")
            
            # Sample prediction request
            test_data = {
                "season": 1,
                "yr": 1,
                "mnth": 1,
                "hr": 12,
                "holiday": 0,
                "weekday": 1,
                "workingday": 1,
                "weathersit": 1,
                "temp": 0.5,
                "atemp": 0.5,
                "hum": 0.6,
                "windspeed": 0.2
            }
            
            response = self.session.post(
                f"{self.base_url}/predict",
                json=test_data,
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                raise AssertionError(f"Prediction failed: {response.status_code} - {response.text}")
            
            result = response.json()
            
            # Validate response structure
            if 'prediction' not in result:
                raise AssertionError("Response missing 'prediction' field")
            
            prediction = result['prediction']
            if not isinstance(prediction, (int, float)) or prediction < 0:
                raise AssertionError(f"Invalid prediction value: {prediction}")
            
            logger.info(f"✅ Prediction endpoint test passed (prediction: {prediction})")
            return True
            
        except Exception as e:
            logger.error(f"❌ Prediction endpoint test failed: {e}")
            return False
    
    def test_metrics_endpoint(self):
        """Test that the metrics endpoint is accessible."""
        try:
            logger.info(f"Testing metrics endpoint: {self.base_url}/metrics")
            response = self.session.get(f"{self.base_url}/metrics", timeout=self.timeout)
            
            # Metrics endpoint might return 404 if not implemented, which is okay
            if response.status_code not in [200, 404]:
                raise AssertionError(f"Metrics endpoint error: {response.status_code}")
            
            if response.status_code == 200:
                logger.info("✅ Metrics endpoint test passed")
            else:
                logger.info("⚠️  Metrics endpoint not implemented (optional)")
                
            return True
            
        except Exception as e:
            logger.error(f"❌ Metrics endpoint test failed: {e}")
            return False
    
    def test_api_response_time(self):
        """Test that API response time is acceptable."""
        try:
            logger.info("Testing API response time...")
            
            test_data = {
                "season": 1, "yr": 1, "mnth": 1, "hr": 12,
                "holiday": 0, "weekday": 1, "workingday": 1,
                "weathersit": 1, "temp": 0.5, "atemp": 0.5,
                "hum": 0.6, "windspeed": 0.2
            }
            
            start_time = time.time()
            response = self.session.post(
                f"{self.base_url}/predict",
                json=test_data,
                timeout=self.timeout
            )
            response_time = time.time() - start_time
            
            if response.status_code != 200:
                raise AssertionError(f"Request failed: {response.status_code}")
            
            # API should respond within 5 seconds
            max_response_time = 5.0
            if response_time > max_response_time:
                raise AssertionError(f"Response time too slow: {response_time:.2f}s > {max_response_time}s")
            
            logger.info(f"✅ Response time test passed ({response_time:.2f}s)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Response time test failed: {e}")
            return False
    
    def run_all_tests(self):
        """Run all smoke tests."""
        logger.info(f"Starting smoke tests for: {self.base_url}")
        
        tests = [
            self.test_health_endpoint,
            self.test_prediction_endpoint,
            self.test_metrics_endpoint,
            self.test_api_response_time,
        ]
        
        results = []
        for test in tests:
            try:
                result = test()
                results.append(result)
            except Exception as e:
                logger.error(f"Test {test.__name__} crashed: {e}")
                results.append(False)
        
        passed = sum(results)
        total = len(results)
        
        logger.info(f"Smoke tests completed: {passed}/{total} passed")
        
        if passed == total:
            logger.info("🎉 All smoke tests passed!")
            return True
        else:
            logger.error(f"❌ {total - passed} smoke tests failed")
            return False


def main():
    """Main function."""
    # Get configuration from environment
    api_url = os.getenv('API_BASE_URL', '').strip()
    timeout = int(os.getenv('TIMEOUT', '30'))
    
    # Handle empty or missing API URL
    if not api_url:
        logger.warning("API_BASE_URL environment variable not set, using default: http://localhost:8000")
        api_url = 'http://localhost:8000'
    
    logger.info(f"API URL: {api_url}")
    logger.info(f"Timeout: {timeout}s")
    
    # Wait for API to be ready
    logger.info("Waiting for API to be ready...")
    time.sleep(5)
    
    # Run smoke tests
    runner = SmokeTestRunner(base_url=api_url, timeout=timeout)
    success = runner.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()