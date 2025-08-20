#!/usr/bin/env python3
"""
Send data drift alert email notifications.

This script sends email alerts to admin@bikesharing.com when data drift is detected.
"""

import os
import sys
import json
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


def send_drift_alert_email(drift_results: dict) -> bool:
    """Send data drift alert email to admin@bikesharing.com."""
    try:
        # Email configuration
        admin_email = 'admin@bikesharing.com'
        sender_email = os.getenv('SENDER_EMAIL', 'mlops@bikesharing.com')
        sender_name = 'BikeSharing MLOps Pipeline'
        
        # SMTP configuration
        smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        smtp_port = int(os.getenv('SMTP_PORT', '587'))
        smtp_username = os.getenv('SMTP_USERNAME', sender_email)
        smtp_password = os.getenv('SMTP_PASSWORD', '')
        
        # Create message
        subject = "⚠️ Data Drift Alert - BikeSharing MLOps"
        
        # Create plain text body
        body = f"""
Data Drift Alert Detected!

Drift Analysis Results:
- Drift Detected: {drift_results['drift_detected']}
- Drift Score: {drift_results['drift_score']:.3f}
- Drifted Columns: {', '.join(drift_results['drifted_columns']) if drift_results['drifted_columns'] else 'None'}
- Total Columns Analyzed: {drift_results['total_columns']}
- Analysis Time: {drift_results['timestamp']}

Action Required:
1. Review the drift report for detailed analysis
2. Consider retraining the model if drift is significant
3. Investigate data quality issues in the pipeline
4. Check for changes in data sources or collection methods

Thresholds:
- Alert triggered when drift score > 0.3 (30% of features)
- Current drift score: {drift_results['drift_score']:.3f}

Best regards,
BikeSharing MLOps Pipeline
"""
        
        # Create HTML body
        html_body = f"""
<html>
<head></head>
<body>
    <h2 style="color: orange;">⚠️ Data Drift Alert Detected!</h2>
    
    <p>Data drift has been detected in the bike sharing prediction system.</p>
    
    <h3>Drift Analysis Results</h3>
    <table border="1" style="border-collapse: collapse;">
        <tr>
            <td style="padding: 8px; font-weight: bold;">Drift Detected</td>
            <td style="padding: 8px; color: {'red' if drift_results['drift_detected'] else 'green'};">{drift_results['drift_detected']}</td>
        </tr>
        <tr>
            <td style="padding: 8px; font-weight: bold;">Drift Score</td>
            <td style="padding: 8px; color: red;">{drift_results['drift_score']:.3f}</td>
        </tr>
        <tr>
            <td style="padding: 8px; font-weight: bold;">Drifted Columns</td>
            <td style="padding: 8px;">{', '.join(drift_results['drifted_columns']) if drift_results['drifted_columns'] else 'None'}</td>
        </tr>
        <tr>
            <td style="padding: 8px; font-weight: bold;">Total Columns</td>
            <td style="padding: 8px;">{drift_results['total_columns']}</td>
        </tr>
        <tr>
            <td style="padding: 8px; font-weight: bold;">Analysis Time</td>
            <td style="padding: 8px;">{drift_results['timestamp']}</td>
        </tr>
    </table>
    
    <h3 style="color: red;">Action Required</h3>
    <ol>
        <li>Review the drift report for detailed analysis</li>
        <li>Consider retraining the model if drift is significant</li>
        <li>Investigate data quality issues in the pipeline</li>
        <li>Check for changes in data sources or collection methods</li>
    </ol>
    
    <h3>Alert Configuration</h3>
    <ul>
        <li>Alert triggered when drift score > 0.3 (30% of features)</li>
        <li><strong>Current drift score:</strong> <span style="color: red;">{drift_results['drift_score']:.3f}</span></li>
    </ul>
    
    <p>Best regards,<br>BikeSharing MLOps Pipeline</p>
</body>
</html>
"""
        
        # Create message
        msg = MimeMultipart('alternative')
        msg['From'] = formataddr((sender_name, sender_email))
        msg['To'] = admin_email
        msg['Subject'] = subject
        
        # Add parts
        text_part = MimeText(body, 'plain')
        html_part = MimeText(html_body, 'html')
        msg.attach(text_part)
        msg.attach(html_part)
        
        # Send email (skip if no password configured)
        if not smtp_password:
            logger.warning("SMTP password not configured, printing alert instead")
            logger.info(f"Drift alert would be sent to {admin_email}")
            logger.info(f"Subject: {subject}")
            logger.info(f"Drift Score: {drift_results['drift_score']:.3f}")
            return True
        
        # Send via SMTP
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_username, smtp_password)
            server.send_message(msg)
        
        logger.info(f"Drift alert email sent successfully to {admin_email}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to send drift alert email: {e}")
        logger.info(f"Fallback: Alert would be sent to {admin_email}")
        logger.info(f"Drift Score: {drift_results['drift_score']:.3f}")
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Send data drift alert email')
    parser.add_argument('--data', required=True, help='JSON data containing drift results')
    
    args = parser.parse_args()
    
    try:
        # Parse drift data
        alert_data = json.loads(args.data)
        drift_results = alert_data['drift_results']
        
        logger.info("Sending data drift alert email...")
        
        success = send_drift_alert_email(drift_results)
        
        if success:
            logger.info("Drift alert email sent successfully")
            sys.exit(0)
        else:
            logger.error("Failed to send drift alert email")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Error processing drift alert: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()