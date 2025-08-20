#!/usr/bin/env python3
"""
Simple notification script for CI/CD pipeline events.

This script provides logging-based notifications when email modules are not available.
Falls back to printing structured notification messages that can be captured in logs.
"""

import os
import sys
import argparse
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def send_notification(notification_type: str, status: str, **kwargs) -> bool:
    """Send a structured notification via logging."""
    try:
        admin_email = 'admin@bikesharing.com'
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
        
        # Create notification based on type
        if notification_type == 'deployment':
            environment = kwargs.get('environment', 'unknown')
            commit_sha = kwargs.get('commit_sha', 'unknown')
            
            title = f"🚀 {environment.title()} Deployment {status.title()}"
            message = f"""
=== NOTIFICATION ===
Type: Deployment
Environment: {environment}
Status: {status}
Commit: {commit_sha}
Time: {timestamp}
Recipient: {admin_email}
==================
"""
            
        elif notification_type == 'rollback':
            title = "🔄 Production Rollback Initiated"
            message = f"""
=== NOTIFICATION ===
Type: Rollback
Status: {status}
Reason: Deployment failure
Time: {timestamp}
Recipient: {admin_email}
==================
"""
            
        elif notification_type == 'training':
            reason = kwargs.get('reason', 'Automated retraining')
            drift_score = kwargs.get('drift_score', 'N/A')
            deployed = kwargs.get('deployed', 'N/A')
            
            title = f"📊 Scheduled Model Training {status.title()}"
            message = f"""
=== NOTIFICATION ===
Type: Training
Status: {status}
Reason: {reason}
Drift Score: {drift_score}
New Model Deployed: {deployed}
Time: {timestamp}
Recipient: {admin_email}
==================
"""
            
        else:
            title = f"🔔 General Notification - {status.title()}"
            message = f"""
=== NOTIFICATION ===
Type: {notification_type}
Status: {status}
Time: {timestamp}
Recipient: {admin_email}
Details: {kwargs}
==================
"""
        
        # Log the notification
        logger.info(f"NOTIFICATION SENT: {title}")
        logger.info(message)
        
        # Also print to stdout for CI/CD visibility
        print(f"\n{'='*50}")
        print(f"EMAIL NOTIFICATION TO: {admin_email}")
        print(f"SUBJECT: {title}")
        print(f"{'='*50}")
        print(message)
        print(f"{'='*50}\n")
        
        return True
        
    except Exception as e:
        logger.error(f"Error sending notification: {e}")
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Send simple notifications for CI/CD pipeline')
    parser.add_argument('--type', choices=['deployment', 'rollback', 'training'], 
                       required=True, help='Type of notification')
    parser.add_argument('--environment', default='staging', 
                       help='Deployment environment (staging/production)')
    parser.add_argument('--status', default='success', 
                       help='Status (success/failure)')
    parser.add_argument('--commit', help='Commit SHA')
    parser.add_argument('--reason', help='Training reason')
    parser.add_argument('--drift-score', help='Data drift score')
    parser.add_argument('--deployed', help='Whether new model was deployed')
    
    args = parser.parse_args()
    
    # Get commit SHA from environment if not provided
    commit_sha = args.commit or os.getenv('GITHUB_SHA', 'unknown')
    
    logger.info(f"Sending {args.type} notification...")
    
    # Prepare kwargs for the notification
    kwargs = {
        'environment': args.environment,
        'commit_sha': commit_sha,
        'reason': args.reason,
        'drift_score': args.drift_score,
        'deployed': args.deployed
    }
    
    # Remove None values
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    
    success = send_notification(args.type, args.status, **kwargs)
    
    if success:
        logger.info("Notification sent successfully")
        sys.exit(0)
    else:
        logger.error("Failed to send notification")
        sys.exit(1)


if __name__ == "__main__":
    main()