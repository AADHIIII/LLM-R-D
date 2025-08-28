"""
Alerting system for monitoring failures and anomalies.
"""

import smtplib
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from enum import Enum
import threading
import time
import sqlite3
import os

from utils.logging import get_structured_logger
from monitoring.metrics_collector import MetricsCollector, get_metrics_collector


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Alert status."""
    ACTIVE = "active"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"


@dataclass
class Alert:
    """Alert data structure."""
    id: str
    name: str
    description: str
    severity: AlertSeverity
    status: AlertStatus
    created_at: datetime
    updated_at: datetime
    resolved_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class AlertRule:
    """Alert rule configuration."""
    name: str
    description: str
    severity: AlertSeverity
    condition: Callable[[Dict[str, Any]], bool]
    threshold_duration: int = 300  # 5 minutes in seconds
    cooldown_period: int = 1800    # 30 minutes in seconds
    enabled: bool = True


class AlertManager:
    """Manages alerts and notifications."""
    
    def __init__(self, db_path: str = "monitoring/alerts.db"):
        self.db_path = db_path
        self.logger = get_structured_logger('alert_manager')
        self.rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.notification_handlers: List[Callable[[Alert], None]] = []
        self.metrics_collector = get_metrics_collector()
        
        # Alert checking thread
        self.running = False
        self.check_thread = None
        self.check_interval = 60  # Check every minute
        
        self._init_database()
        self._register_default_rules()
    
    def _init_database(self) -> None:
        """Initialize SQLite database for alert storage."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    severity TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    resolved_at TEXT,
                    acknowledged_at TEXT,
                    metadata TEXT
                )
            """)
            
            conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_created_at ON alerts(created_at)")
    
    def _register_default_rules(self) -> None:
        """Register default monitoring rules."""
        
        # High CPU usage alert
        self.add_rule(AlertRule(
            name="high_cpu_usage",
            description="CPU usage is consistently high",
            severity=AlertSeverity.HIGH,
            condition=lambda metrics: metrics.get('system', {}).get('avg_cpu_percent', 0) > 80,
            threshold_duration=300  # 5 minutes
        ))
        
        # High memory usage alert
        self.add_rule(AlertRule(
            name="high_memory_usage",
            description="Memory usage is critically high",
            severity=AlertSeverity.CRITICAL,
            condition=lambda metrics: metrics.get('system', {}).get('avg_memory_percent', 0) > 90,
            threshold_duration=180  # 3 minutes
        ))
        
        # High error rate alert
        self.add_rule(AlertRule(
            name="high_error_rate",
            description="API error rate is above threshold",
            severity=AlertSeverity.HIGH,
            condition=lambda metrics: metrics.get('api', {}).get('error_rate_percent', 0) > 10,
            threshold_duration=300  # 5 minutes
        ))
        
        # Slow response time alert
        self.add_rule(AlertRule(
            name="slow_response_time",
            description="API response times are consistently slow",
            severity=AlertSeverity.MEDIUM,
            condition=lambda metrics: metrics.get('api', {}).get('avg_response_time_ms', 0) > 5000,
            threshold_duration=600  # 10 minutes
        ))
        
        # Disk space alert
        self.add_rule(AlertRule(
            name="low_disk_space",
            description="Disk space is running low",
            severity=AlertSeverity.HIGH,
            condition=lambda metrics: metrics.get('system', {}).get('avg_disk_percent', 0) > 85,
            threshold_duration=300  # 5 minutes
        ))
        
        # No recent requests alert (system might be down)
        self.add_rule(AlertRule(
            name="no_recent_requests",
            description="No API requests received recently",
            severity=AlertSeverity.CRITICAL,
            condition=lambda metrics: metrics.get('api', {}).get('total_requests', 0) == 0,
            threshold_duration=900  # 15 minutes
        ))
    
    def add_rule(self, rule: AlertRule) -> None:
        """Add an alert rule."""
        self.rules[rule.name] = rule
        self.logger.info(
            f"Added alert rule: {rule.name}",
            rule_name=rule.name,
            severity=rule.severity.value
        )
    
    def remove_rule(self, rule_name: str) -> None:
        """Remove an alert rule."""
        if rule_name in self.rules:
            del self.rules[rule_name]
            self.logger.info(f"Removed alert rule: {rule_name}")
    
    def add_notification_handler(self, handler: Callable[[Alert], None]) -> None:
        """Add a notification handler for alerts."""
        self.notification_handlers.append(handler)
    
    def start_monitoring(self) -> None:
        """Start alert monitoring in background thread."""
        if self.running:
            self.logger.warning("Alert monitoring already running")
            return
        
        self.running = True
        self.check_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.check_thread.start()
        
        self.logger.info("Started alert monitoring")
    
    def stop_monitoring(self) -> None:
        """Stop alert monitoring."""
        self.running = False
        if self.check_thread:
            self.check_thread.join(timeout=5)
        
        self.logger.info("Stopped alert monitoring")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop running in background thread."""
        while self.running:
            try:
                self._check_alert_conditions()
                time.sleep(self.check_interval)
            except Exception as e:
                self.logger.error(
                    "Error in alert monitoring loop",
                    error=str(e)
                )
                time.sleep(self.check_interval)
    
    def _check_alert_conditions(self) -> None:
        """Check all alert rule conditions."""
        # Get recent metrics summary
        metrics_summary = self.metrics_collector.get_metrics_summary(hours=1)
        
        for rule_name, rule in self.rules.items():
            if not rule.enabled:
                continue
            
            try:
                # Check if condition is met
                condition_met = rule.condition(metrics_summary)
                
                if condition_met:
                    self._handle_condition_triggered(rule_name, rule, metrics_summary)
                else:
                    self._handle_condition_resolved(rule_name, rule)
                    
            except Exception as e:
                self.logger.error(
                    f"Error checking alert rule: {rule_name}",
                    rule_name=rule_name,
                    error=str(e)
                )
    
    def _handle_condition_triggered(self, rule_name: str, rule: AlertRule, metrics: Dict[str, Any]) -> None:
        """Handle when an alert condition is triggered."""
        alert_id = f"{rule_name}_{datetime.utcnow().strftime('%Y%m%d')}"
        
        # Check if alert already exists and is active
        if alert_id in self.active_alerts:
            # Update existing alert
            alert = self.active_alerts[alert_id]
            alert.updated_at = datetime.utcnow()
            self._update_alert_in_db(alert)
        else:
            # Create new alert
            alert = Alert(
                id=alert_id,
                name=rule.name,
                description=rule.description,
                severity=rule.severity,
                status=AlertStatus.ACTIVE,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                metadata={'metrics': metrics}
            )
            
            self.active_alerts[alert_id] = alert
            self._store_alert_in_db(alert)
            
            # Send notifications
            self._send_notifications(alert)
            
            self.logger.warning(
                f"Alert triggered: {rule.name}",
                alert_id=alert_id,
                severity=rule.severity.value,
                description=rule.description
            )
    
    def _handle_condition_resolved(self, rule_name: str, rule: AlertRule) -> None:
        """Handle when an alert condition is resolved."""
        alert_id = f"{rule_name}_{datetime.utcnow().strftime('%Y%m%d')}"
        
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = datetime.utcnow()
            alert.updated_at = datetime.utcnow()
            
            self._update_alert_in_db(alert)
            del self.active_alerts[alert_id]
            
            self.logger.info(
                f"Alert resolved: {rule.name}",
                alert_id=alert_id
            )
    
    def _send_notifications(self, alert: Alert) -> None:
        """Send notifications for an alert."""
        for handler in self.notification_handlers:
            try:
                handler(alert)
            except Exception as e:
                self.logger.error(
                    f"Error sending notification for alert {alert.id}",
                    alert_id=alert.id,
                    error=str(e)
                )
    
    def _store_alert_in_db(self, alert: Alert) -> None:
        """Store alert in database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO alerts (
                    id, name, description, severity, status, created_at,
                    updated_at, resolved_at, acknowledged_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                alert.id,
                alert.name,
                alert.description,
                alert.severity.value,
                alert.status.value,
                alert.created_at.isoformat(),
                alert.updated_at.isoformat(),
                alert.resolved_at.isoformat() if alert.resolved_at else None,
                alert.acknowledged_at.isoformat() if alert.acknowledged_at else None,
                json.dumps(alert.metadata) if alert.metadata else None
            ))
    
    def _update_alert_in_db(self, alert: Alert) -> None:
        """Update alert in database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE alerts SET
                    status = ?, updated_at = ?, resolved_at = ?,
                    acknowledged_at = ?, metadata = ?
                WHERE id = ?
            """, (
                alert.status.value,
                alert.updated_at.isoformat(),
                alert.resolved_at.isoformat() if alert.resolved_at else None,
                alert.acknowledged_at.isoformat() if alert.acknowledged_at else None,
                json.dumps(alert.metadata) if alert.metadata else None,
                alert.id
            ))
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_at = datetime.utcnow()
            alert.updated_at = datetime.utcnow()
            
            self._update_alert_in_db(alert)
            
            self.logger.info(f"Alert acknowledged: {alert_id}")
            return True
        
        return False
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get all active alerts."""
        return [asdict(alert) for alert in self.active_alerts.values()]
    
    def get_alert_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get alert history for the specified time period."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM alerts 
                WHERE created_at >= ? 
                ORDER BY created_at DESC
            """, (cutoff_time.isoformat(),))
            
            alerts = []
            for row in cursor.fetchall():
                alert_dict = dict(row)
                if alert_dict['metadata']:
                    alert_dict['metadata'] = json.loads(alert_dict['metadata'])
                alerts.append(alert_dict)
            
            return alerts


class EmailNotificationHandler:
    """Email notification handler for alerts."""
    
    def __init__(self, smtp_server: str, smtp_port: int, username: str, 
                 password: str, from_email: str, to_emails: List[str]):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_email = from_email
        self.to_emails = to_emails
        self.logger = get_structured_logger('email_notifications')
    
    def __call__(self, alert: Alert) -> None:
        """Send email notification for alert."""
        try:
            subject = f"[{alert.severity.value.upper()}] {alert.name}"
            
            body = f"""
Alert: {alert.name}
Severity: {alert.severity.value.upper()}
Description: {alert.description}
Created: {alert.created_at.isoformat()}
Status: {alert.status.value}

Metadata:
{json.dumps(alert.metadata, indent=2) if alert.metadata else 'None'}
"""
            
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = ', '.join(self.to_emails)
            msg['Subject'] = subject
            
            msg.attach(MIMEText(body, 'plain'))
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)
            
            self.logger.info(
                f"Email notification sent for alert {alert.id}",
                alert_id=alert.id,
                recipients=self.to_emails
            )
            
        except Exception as e:
            self.logger.error(
                f"Failed to send email notification for alert {alert.id}",
                alert_id=alert.id,
                error=str(e)
            )


class SlackNotificationHandler:
    """Slack notification handler for alerts."""
    
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
        self.logger = get_structured_logger('slack_notifications')
    
    def __call__(self, alert: Alert) -> None:
        """Send Slack notification for alert."""
        try:
            import requests
            
            color_map = {
                AlertSeverity.LOW: "#36a64f",      # Green
                AlertSeverity.MEDIUM: "#ff9500",   # Orange
                AlertSeverity.HIGH: "#ff0000",     # Red
                AlertSeverity.CRITICAL: "#8b0000"  # Dark Red
            }
            
            payload = {
                "attachments": [
                    {
                        "color": color_map.get(alert.severity, "#ff0000"),
                        "title": f"Alert: {alert.name}",
                        "text": alert.description,
                        "fields": [
                            {
                                "title": "Severity",
                                "value": alert.severity.value.upper(),
                                "short": True
                            },
                            {
                                "title": "Status",
                                "value": alert.status.value,
                                "short": True
                            },
                            {
                                "title": "Created",
                                "value": alert.created_at.isoformat(),
                                "short": True
                            }
                        ],
                        "footer": "LLM Optimization Platform",
                        "ts": int(alert.created_at.timestamp())
                    }
                ]
            }
            
            response = requests.post(self.webhook_url, json=payload)
            response.raise_for_status()
            
            self.logger.info(
                f"Slack notification sent for alert {alert.id}",
                alert_id=alert.id
            )
            
        except Exception as e:
            self.logger.error(
                f"Failed to send Slack notification for alert {alert.id}",
                alert_id=alert.id,
                error=str(e)
            )


# Global alert manager instance
_alert_manager = None


def get_alert_manager() -> AlertManager:
    """Get global alert manager instance."""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager


def start_alert_monitoring() -> None:
    """Start global alert monitoring."""
    manager = get_alert_manager()
    manager.start_monitoring()


def stop_alert_monitoring() -> None:
    """Stop global alert monitoring."""
    global _alert_manager
    if _alert_manager:
        _alert_manager.stop_monitoring()