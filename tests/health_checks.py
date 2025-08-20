#!/usr/bin/env python3
"""
Production health checks for the deployed application.

These checks verify that the production deployment is healthy
and functioning correctly after deployment.
"""

import os
import sys
import requests
import logging
import time
import subprocess
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HealthChecker:
    """Production health checker."""
    
    def __init__(self, base_url=None, timeout=300):
        self.base_url = base_url or os.getenv('API_BASE_URL', 'http://localhost:8000')
        self.timeout = timeout
        self.session = requests.Session()
        
    def check_api_availability(self):
        """Check if the API is available and responding."""
        try:
            logger.info(f"Checking API availability: {self.base_url}")
            
            # Try multiple times with backoff
            max_retries = 10
            backoff_delay = 2
            
            for attempt in range(max_retries):
                try:
                    response = self.session.get(
                        f"{self.base_url}/health",
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        logger.info(f"✅ API is available (attempt {attempt + 1})")
                        return True
                        
                except requests.exceptions.RequestException:
                    if attempt < max_retries - 1:
                        logger.info(f"API not ready, retrying in {backoff_delay}s... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(backoff_delay)
                    else:
                        raise
            
            logger.error("❌ API not available after all retries")
            return False
            
        except Exception as e:
            logger.error(f"❌ API availability check failed: {e}")
            return False
    
    def check_prediction_functionality(self):
        """Check that predictions are working correctly."""
        try:
            logger.info("Checking prediction functionality...")
            
            # Test multiple prediction scenarios
            test_cases = [
                {
                    "name": "typical_workday",
                    "data": {
                        "season": 2, "yr": 1, "mnth": 6, "hr": 8,
                        "holiday": 0, "weekday": 1, "workingday": 1,
                        "weathersit": 1, "temp": 0.6, "atemp": 0.6,
                        "hum": 0.5, "windspeed": 0.3
                    }
                },
                {
                    "name": "weekend_evening",
                    "data": {
                        "season": 3, "yr": 1, "mnth": 8, "hr": 19,
                        "holiday": 0, "weekday": 6, "workingday": 0,
                        "weathersit": 1, "temp": 0.7, "atemp": 0.7,
                        "hum": 0.4, "windspeed": 0.2
                    }
                },
                {
                    "name": "bad_weather",
                    "data": {
                        "season": 4, "yr": 1, "mnth": 12, "hr": 14,
                        "holiday": 0, "weekday": 3, "workingday": 1,
                        "weathersit": 3, "temp": 0.2, "atemp": 0.2,
                        "hum": 0.9, "windspeed": 0.8
                    }
                }
            ]
            
            for test_case in test_cases:
                response = self.session.post(
                    f"{self.base_url}/predict",
                    json=test_case["data"],
                    timeout=10
                )
                
                if response.status_code != 200:
                    raise AssertionError(f"Prediction failed for {test_case['name']}: {response.status_code}")
                
                result = response.json()
                prediction = result.get('prediction')
                
                if not isinstance(prediction, (int, float)) or prediction < 0:
                    raise AssertionError(f"Invalid prediction for {test_case['name']}: {prediction}")
                
                logger.info(f"✅ Prediction test '{test_case['name']}' passed (prediction: {prediction:.1f})")
            
            logger.info("✅ All prediction functionality checks passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Prediction functionality check failed: {e}")
            return False
    
    def check_performance_metrics(self):
        """Check API performance metrics."""
        try:
            logger.info("Checking API performance...")
            
            # Test response times
            test_data = {
                "season": 1, "yr": 1, "mnth": 1, "hr": 12,
                "holiday": 0, "weekday": 1, "workingday": 1,
                "weathersit": 1, "temp": 0.5, "atemp": 0.5,
                "hum": 0.6, "windspeed": 0.2
            }
            
            response_times = []
            num_requests = 5
            
            for i in range(num_requests):
                start_time = time.time()
                response = self.session.post(
                    f"{self.base_url}/predict",
                    json=test_data,
                    timeout=10
                )
                response_time = time.time() - start_time
                response_times.append(response_time)
                
                if response.status_code != 200:
                    raise AssertionError(f"Request {i+1} failed: {response.status_code}")
            
            avg_response_time = sum(response_times) / len(response_times)
            max_response_time = max(response_times)
            
            # Performance thresholds
            max_avg_time = 2.0  # 2 seconds average
            max_single_time = 5.0  # 5 seconds max for any single request
            
            if avg_response_time > max_avg_time:
                raise AssertionError(f"Average response time too slow: {avg_response_time:.2f}s > {max_avg_time}s")
            
            if max_response_time > max_single_time:
                raise AssertionError(f"Maximum response time too slow: {max_response_time:.2f}s > {max_single_time}s")
            
            logger.info(f"✅ Performance check passed (avg: {avg_response_time:.2f}s, max: {max_response_time:.2f}s)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Performance check failed: {e}")
            return False
    
    def check_error_handling(self):
        """Check that API handles errors gracefully."""
        try:
            logger.info("Checking error handling...")
            
            # Test invalid input
            invalid_cases = [
                {"data": {}, "expected_status": 422, "name": "empty_data"},
                {"data": {"invalid": "field"}, "expected_status": 422, "name": "invalid_fields"},
                {"data": {"season": "invalid"}, "expected_status": 422, "name": "invalid_types"},
            ]
            
            for case in invalid_cases:
                response = self.session.post(
                    f"{self.base_url}/predict",
                    json=case["data"],
                    timeout=10
                )
                
                if response.status_code != case["expected_status"]:
                    logger.warning(f"⚠️  Unexpected status for {case['name']}: {response.status_code} (expected {case['expected_status']})")
                else:
                    logger.info(f"✅ Error handling test '{case['name']}' passed")
            
            logger.info("✅ Error handling checks completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error handling check failed: {e}")
            return False
    
    def check_monitoring_endpoints(self):
        """Check monitoring and observability endpoints."""
        try:
            logger.info("Checking monitoring endpoints...")
            
            # Check metrics endpoint (optional)
            try:
                response = self.session.get(f"{self.base_url}/metrics", timeout=5)
                if response.status_code == 200:
                    logger.info("✅ Metrics endpoint available")
                elif response.status_code == 404:
                    logger.info("⚠️  Metrics endpoint not implemented (optional)")
                else:
                    logger.warning(f"⚠️  Metrics endpoint returned {response.status_code}")
            except:
                logger.info("⚠️  Metrics endpoint not accessible (optional)")
            
            # Check docs endpoint
            try:
                response = self.session.get(f"{self.base_url}/docs", timeout=5)
                if response.status_code == 200:
                    logger.info("✅ API documentation available")
                else:
                    logger.warning(f"⚠️  API docs returned {response.status_code}")
            except:
                logger.warning("⚠️  API documentation not accessible")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Monitoring endpoints check failed: {e}")
            return False
    
    def run_all_checks(self):
        """Run all health checks."""
        logger.info(f"Starting production health checks for: {self.base_url}")
        
        checks = [
            ("API Availability", self.check_api_availability),
            ("Prediction Functionality", self.check_prediction_functionality),
            ("Performance Metrics", self.check_performance_metrics),
            ("Error Handling", self.check_error_handling),
            ("Monitoring Endpoints", self.check_monitoring_endpoints),
        ]
        
        results = []
        for check_name, check_func in checks:
            logger.info(f"Running check: {check_name}")
            try:
                result = check_func()
                results.append((check_name, result))
            except Exception as e:
                logger.error(f"Check '{check_name}' crashed: {e}")
                results.append((check_name, False))
        
        # Report results
        passed = sum(1 for _, result in results if result)
        total = len(results)
        
        logger.info(f"\nHealth check summary:")
        for check_name, result in results:
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"  {check_name}: {status}")
        
        logger.info(f"\nOverall: {passed}/{total} checks passed")
        
        if passed == total:
            logger.info("🎉 All health checks passed!")
            return True
        else:
            logger.error(f"❌ {total - passed} health checks failed")
            return False


def main():
    """Main function."""
    # Get configuration from environment
    api_url = os.getenv('API_BASE_URL', 'http://localhost:8000')
    timeout = int(os.getenv('TIMEOUT', '300'))
    
    logger.info(f"Production health checks starting...")
    logger.info(f"API URL: {api_url}")
    logger.info(f"Timeout: {timeout}s")
    
    # Run health checks
    checker = HealthChecker(base_url=api_url, timeout=timeout)
    success = checker.run_all_checks()
    
    # Exit with appropriate code
    if success:
        logger.info("🎉 Production deployment is healthy!")
        sys.exit(0)
    else:
        logger.error("❌ Production deployment has health issues!")
        sys.exit(1)


if __name__ == "__main__":
    main()