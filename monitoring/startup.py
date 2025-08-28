"""
Monitoring system startup and initialization.
"""

import os
import atexit
from typing import Optional

from monitoring.metrics_collector import start_metrics_collection, stop_metrics_collection
from monitoring.alerting import (
    start_alert_monitoring, stop_alert_monitoring, get_alert_manager,
    EmailNotificationHandler, SlackNotificationHandler
)
from utils.logging import get_structured_logger


logger = get_structured_logger('monitoring_startup')


def initialize_monitoring_system(
    enable_email_alerts: bool = False,
    email_config: Optional[dict] = None,
    enable_slack_alerts: bool = False,
    slack_webhook_url: Optional[str] = None
) -> None:
    """
    Initialize the complete monitoring system.
    
    Args:
        enable_email_alerts: Whether to enable email notifications
        email_config: Email configuration dictionary
        enable_slack_alerts: Whether to enable Slack notifications
        slack_webhook_url: Slack webhook URL for notifications
    """
    try:
        logger.info("Initializing monitoring system")
        
        # Start metrics collection
        start_metrics_collection()
        logger.info("Metrics collection started")
        
        # Start alert monitoring
        start_alert_monitoring()
        logger.info("Alert monitoring started")
        
        # Configure notification handlers
        alert_manager = get_alert_manager()
        
        if enable_email_alerts and email_config:
            try:
                email_handler = EmailNotificationHandler(
                    smtp_server=email_config['smtp_server'],
                    smtp_port=email_config['smtp_port'],
                    username=email_config['username'],
                    password=email_config['password'],
                    from_email=email_config['from_email'],
                    to_emails=email_config['to_emails']
                )
                alert_manager.add_notification_handler(email_handler)
                logger.info("Email notifications enabled")
            except Exception as e:
                logger.error("Failed to setup email notifications", error=str(e))
        
        if enable_slack_alerts and slack_webhook_url:
            try:
                slack_handler = SlackNotificationHandler(slack_webhook_url)
                alert_manager.add_notification_handler(slack_handler)
                logger.info("Slack notifications enabled")
            except Exception as e:
                logger.error("Failed to setup Slack notifications", error=str(e))
        
        # Register cleanup on exit
        atexit.register(shutdown_monitoring_system)
        
        logger.info("Monitoring system initialized successfully")
        
    except Exception as e:
        logger.error("Failed to initialize monitoring system", error=str(e))
        raise


def shutdown_monitoring_system() -> None:
    """Shutdown the monitoring system gracefully."""
    try:
        logger.info("Shutting down monitoring system")
        
        stop_alert_monitoring()
        stop_metrics_collection()
        
        logger.info("Monitoring system shutdown complete")
        
    except Exception as e:
        logger.error("Error during monitoring system shutdown", error=str(e))


def configure_from_environment() -> dict:
    """Configure monitoring system from environment variables."""
    config = {
        'enable_email_alerts': os.getenv('ENABLE_EMAIL_ALERTS', 'false').lower() == 'true',
        'enable_slack_alerts': os.getenv('ENABLE_SLACK_ALERTS', 'false').lower() == 'true',
        'slack_webhook_url': os.getenv('SLACK_WEBHOOK_URL'),
        'email_config': None
    }
    
    # Email configuration
    if config['enable_email_alerts']:
        email_config = {
            'smtp_server': os.getenv('SMTP_SERVER'),
            'smtp_port': int(os.getenv('SMTP_PORT', '587')),
            'username': os.getenv('SMTP_USERNAME'),
            'password': os.getenv('SMTP_PASSWORD'),
            'from_email': os.getenv('ALERT_FROM_EMAIL'),
            'to_emails': os.getenv('ALERT_TO_EMAILS', '').split(',')
        }
        
        # Validate required email config
        required_fields = ['smtp_server', 'username', 'password', 'from_email']
        if all(email_config.get(field) for field in required_fields):
            config['email_config'] = email_config
        else:
            logger.warning("Incomplete email configuration, disabling email alerts")
            config['enable_email_alerts'] = False
    
    return config


def start_monitoring_with_config() -> None:
    """Start monitoring system with configuration from environment."""
    config = configure_from_environment()
    
    initialize_monitoring_system(
        enable_email_alerts=config['enable_email_alerts'],
        email_config=config['email_config'],
        enable_slack_alerts=config['enable_slack_alerts'],
        slack_webhook_url=config['slack_webhook_url']
    )


if __name__ == '__main__':
    # Allow running this script directly to start monitoring
    start_monitoring_with_config()
    
    # Keep the script running
    try:
        import time
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        logger.info("Monitoring system stopped by user")
        shutdown_monitoring_system()