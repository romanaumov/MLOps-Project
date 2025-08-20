#!/usr/bin/env python3
"""
Send email notifications for CI/CD pipeline events.

This script sends email notifications to admin@bikesharing.com for various
pipeline events like deployment completion, failures, etc.
"""

import os
import sys
import argparse
import smtplib
import logging
from datetime import datetime
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from email.utils import formataddr

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def send_email(subject: str, body: str, html_body: str = None) -> bool:
    """Send email notification to admin@bikesharing.com."""
    try:
        # Email configuration
        admin_email = 'admin@bikesharing.com'
        sender_email = os.getenv('SENDER_EMAIL', 'mlops@bikesharing.com')
        sender_name = 'BikeSharing MLOps Pipeline'
        
        # SMTP configuration (using environment variables)
        smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        smtp_port = int(os.getenv('SMTP_PORT', '587'))
        smtp_username = os.getenv('SMTP_USERNAME', sender_email)
        smtp_password = os.getenv('SMTP_PASSWORD', '')
        
        # Create message
        msg = MimeMultipart('alternative')
        msg['From'] = formataddr((sender_name, sender_email))
        msg['To'] = admin_email
        msg['Subject'] = subject
        
        # Add plain text part
        text_part = MimeText(body, 'plain')
        msg.attach(text_part)
        
        # Add HTML part if provided
        if html_body:
            html_part = MimeText(html_body, 'html')
            msg.attach(html_part)
        
        # Send email (skip if no password configured)
        if not smtp_password:
            logger.warning("SMTP password not configured, printing notification instead")
            logger.info(f"Email notification would be sent to {admin_email}")
            logger.info(f"Subject: {subject}")
            logger.info(f"Body: {body}")
            return True
        
        # Send via SMTP
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_username, smtp_password)
            server.send_message(msg)
        
        logger.info(f"Email sent successfully to {admin_email}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        logger.info(f"Fallback: Notification would be sent to {admin_email}")
        logger.info(f"Subject: {subject}")
        logger.info(f"Body: {body}")
        return False


def send_deployment_notification(environment: str, status: str, commit_sha: str) -> bool:
    """Send deployment notification."""
    try:
        # Determine status icon and color
        if status.lower() in ['success', 'completed']:
            status_icon = '✅'
            status_color = 'green'
            status_text = 'Successful'
        elif status.lower() in ['failure', 'failed']:
            status_icon = '❌'
            status_color = 'red'
            status_text = 'Failed'
        else:
            status_icon = '⚠️'
            status_color = 'orange'
            status_text = status.title()
        
        subject = f"{status_icon} {environment.title()} Deployment {status_text} - BikeSharing MLOps"
        
        # Create plain text body
        body = f"""
{environment.title()} Deployment {status_text}

Deployment Details:
- Environment: {environment}
- Status: {status_text}
- Commit: {commit_sha}
- Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}

Repository: MLOps-Project
Branch: main

Best regards,
BikeSharing MLOps Pipeline
"""
        
        # Create HTML body
        html_body = f"""
<html>
<head></head>
<body>
    <h2 style="color: {status_color};">{status_icon} {environment.title()} Deployment {status_text}</h2>
    
    <h3>Deployment Details</h3>
    <ul>
        <li><strong>Environment:</strong> {environment}</li>
        <li><strong>Status:</strong> <span style="color: {status_color};">{status_text}</span></li>
        <li><strong>Commit:</strong> <code>{commit_sha}</code></li>
        <li><strong>Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</li>
    </ul>
    
    <h3>Repository Information</h3>
    <ul>
        <li><strong>Repository:</strong> MLOps-Project</li>
        <li><strong>Branch:</strong> main</li>
    </ul>
    
    <p>Best regards,<br>BikeSharing MLOps Pipeline</p>
</body>
</html>
"""
        
        return send_email(subject, body, html_body)
        
    except Exception as e:
        logger.error(f"Error sending deployment notification: {e}")
        return False


def send_rollback_notification() -> bool:
    """Send rollback notification."""
    try:
        subject = "🔄 Production Rollback Initiated - BikeSharing MLOps"
        
        body = f"""
Production Rollback Initiated

A production rollback has been initiated due to deployment failure.

Details:
- Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
- Reason: Deployment failure
- Action: Rolling back to previous version

Please investigate the deployment issues and ensure system stability.

Best regards,
BikeSharing MLOps Pipeline
"""
        
        html_body = f"""
<html>
<head></head>
<body>
    <h2 style="color: red;">🔄 Production Rollback Initiated</h2>
    
    <p>A production rollback has been initiated due to deployment failure.</p>
    
    <h3>Details</h3>
    <ul>
        <li><strong>Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</li>
        <li><strong>Reason:</strong> Deployment failure</li>
        <li><strong>Action:</strong> Rolling back to previous version</li>
    </ul>
    
    <p style="color: red;"><strong>Please investigate the deployment issues and ensure system stability.</strong></p>
    
    <p>Best regards,<br>BikeSharing MLOps Pipeline</p>
</body>
</html>
"""
        
        return send_email(subject, body, html_body)
        
    except Exception as e:
        logger.error(f"Error sending rollback notification: {e}")
        return False


def send_training_notification(status: str, reason: str = None, drift_score: str = None, deployed: str = None) -> bool:
    """Send training completion notification."""
    try:
        # Determine status icon and color
        if status.lower() in ['success', 'completed']:
            status_icon = '✅'
            status_color = 'green'
            status_text = 'Successful'
        elif status.lower() in ['failure', 'failed']:
            status_icon = '❌'
            status_color = 'red'
            status_text = 'Failed'
        else:
            status_icon = '⚠️'
            status_color = 'orange'
            status_text = status.title()
        
        subject = f"{status_icon} Scheduled Model Training {status_text} - BikeSharing MLOps"
        
        # Create plain text body
        body = f"""
Scheduled Model Training {status_text}

Training Details:
- Status: {status_text}
- Reason: {reason or 'Automated retraining'}
- Drift Score: {drift_score or 'N/A'}
- New Model Deployed: {deployed or 'N/A'}
- Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}

Best regards,
BikeSharing MLOps Pipeline
"""
        
        # Create HTML body
        html_body = f"""
<html>
<head></head>
<body>
    <h2 style="color: {status_color};">{status_icon} Scheduled Model Training {status_text}</h2>
    
    <h3>Training Details</h3>
    <ul>
        <li><strong>Status:</strong> <span style="color: {status_color};">{status_text}</span></li>
        <li><strong>Reason:</strong> {reason or 'Automated retraining'}</li>
        <li><strong>Drift Score:</strong> {drift_score or 'N/A'}</li>
        <li><strong>New Model Deployed:</strong> {deployed or 'N/A'}</li>
        <li><strong>Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</li>
    </ul>
    
    <p>Best regards,<br>BikeSharing MLOps Pipeline</p>
</body>
</html>
"""
        
        return send_email(subject, body, html_body)
        
    except Exception as e:
        logger.error(f"Error sending training notification: {e}")
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Send email notifications for CI/CD pipeline')
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
    
    success = False
    if args.type == 'deployment':
        success = send_deployment_notification(args.environment, args.status, commit_sha)
    elif args.type == 'rollback':
        success = send_rollback_notification()
    elif args.type == 'training':
        success = send_training_notification(args.status, args.reason, args.drift_score, args.deployed)
    
    if success:
        logger.info("Notification sent successfully")
        sys.exit(0)
    else:
        logger.error("Failed to send notification")
        sys.exit(1)


if __name__ == "__main__":
    main()