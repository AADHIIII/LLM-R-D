"""
Tests for monitoring system functionality including metrics collection and alerting.
"""

import unittest
import tempfile
import os
import time
import threading
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import sqlite3
import json

from monitoring.metrics_collector import (
    MetricsCollector, SystemMetrics, APIMetrics, ApplicationMetrics,
    get_metrics_collector, start_metrics_collection, stop_metrics_collection
)
from monitoring.alerting import (
    AlertManager, Alert, AlertRule, AlertSeverity, AlertStatus,
    EmailNotificationHandler, SlackNotificationHandler,
    get_alert_manager, start_alert_monitoring, stop_alert_monitoring
)


class TestMetricsCollector(unittest.TestCase):
    """Test metrics collection functionality."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, 'test_metrics.db')
        self.collector = MetricsCollector(self.db_path, collection_interval=1)
    
    def tearDown(self):
        self.collector.stop_collection()
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_database_initialization(self):
        """Test database schema creation."""
        self.assertTrue(os.path.exists(self.db_path))
        
        with sqlite3.connect(self.db_path) as conn:
            # Check if tables exist
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            
            table_names = [table[0] for table in tables]
            self.assertIn('system_metrics', table_names)
            self.assertIn('api_metrics', table_names)
            self.assertIn('app_metrics', table_names)
    
    @patch('monitoring.metrics_collector.psutil')
    def test_system_metrics_collection(self, mock_psutil):
        """Test system metrics collection."""
        # Mock psutil responses
        mock_psutil.cpu_percent.return_value = 45.5
        
        mock_memory = Mock()
        mock_memory.percent = 60.2
        mock_memory.used = 8 * 1024 * 1024 * 1024  # 8GB
        mock_memory.available = 4 * 1024 * 1024 * 1024  # 4GB
        mock_psutil.virtual_memory.return_value = mock_memory
        
        mock_disk = Mock()
        mock_disk.percent = 75.0
        mock_disk.free = 100 * 1024 * 1024 * 1024  # 100GB
        mock_psutil.disk_usage.return_value = mock_disk
        
        mock_network = Mock()
        mock_network.bytes_sent = 1000000
        mock_network.bytes_recv = 2000000
        mock_psutil.net_io_counters.return_value = mock_network
        
        mock_psutil.net_connections.return_value = [Mock()] * 50  # 50 connections
        
        # Collect metrics
        metrics = self.collector._collect_system_metrics()
        
        self.assertEqual(metrics.cpu_percent, 45.5)
        self.assertEqual(metrics.memory_percent, 60.2)
        self.assertEqual(metrics.disk_usage_percent, 75.0)
        self.assertEqual(metrics.network_bytes_sent, 1000000)
        self.assertEqual(metrics.network_bytes_recv, 2000000)
        self.assertEqual(metrics.active_connections, 50)
    
    def test_api_metrics_recording(self):
        """Test API metrics recording."""
        api_metrics = APIMetrics(
            timestamp=datetime.utcnow(),
            endpoint='/api/v1/generate',
            method='POST',
            status_code=200,
            response_time_ms=150.5,
            request_size_bytes=1024,
            response_size_bytes=2048
        )
        
        self.collector.record_api_metrics(api_metrics)
        
        # Verify stored in memory
        self.assertEqual(len(self.collector.api_metrics), 1)
        stored_metric = self.collector.api_metrics[0]
        self.assertEqual(stored_metric.endpoint, '/api/v1/generate')
        self.assertEqual(stored_metric.status_code, 200)
        
        # Verify stored in database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM api_metrics")
            self.assertEqual(cursor.fetchone()[0], 1)
    
    def test_metrics_summary(self):
        """Test metrics summary generation."""
        # Insert test data
        now = datetime.utcnow()
        
        with sqlite3.connect(self.db_path) as conn:
            # Insert system metrics
            conn.execute("""
                INSERT INTO system_metrics (
                    timestamp, cpu_percent, memory_percent, disk_usage_percent
                ) VALUES (?, ?, ?, ?)
            """, (now.isoformat(), 50.0, 70.0, 80.0))
            
            # Insert API metrics
            conn.execute("""
                INSERT INTO api_metrics (
                    timestamp, endpoint, method, status_code, response_time_ms
                ) VALUES (?, ?, ?, ?, ?)
            """, (now.isoformat(), '/test', 'GET', 200, 100.0))
            
            conn.execute("""
                INSERT INTO api_metrics (
                    timestamp, endpoint, method, status_code, response_time_ms
                ) VALUES (?, ?, ?, ?, ?)
            """, (now.isoformat(), '/test', 'GET', 500, 200.0))
        
        summary = self.collector.get_metrics_summary(hours=1)
        
        self.assertEqual(summary['system']['avg_cpu_percent'], 50.0)
        self.assertEqual(summary['system']['avg_memory_percent'], 70.0)
        self.assertEqual(summary['api']['total_requests'], 2)
        self.assertEqual(summary['api']['error_count'], 1)
        self.assertEqual(summary['api']['error_rate_percent'], 50.0)
    
    def test_metrics_cleanup(self):
        """Test old metrics cleanup."""
        old_time = datetime.utcnow() - timedelta(days=10)
        recent_time = datetime.utcnow()
        
        with sqlite3.connect(self.db_path) as conn:
            # Insert old metrics
            conn.execute("""
                INSERT INTO system_metrics (timestamp, cpu_percent)
                VALUES (?, ?)
            """, (old_time.isoformat(), 50.0))
            
            # Insert recent metrics
            conn.execute("""
                INSERT INTO system_metrics (timestamp, cpu_percent)
                VALUES (?, ?)
            """, (recent_time.isoformat(), 60.0))
        
        deleted_counts = self.collector.cleanup_old_metrics(days=7)
        
        self.assertEqual(deleted_counts['system_metrics'], 1)
        
        # Verify only recent metrics remain
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM system_metrics")
            self.assertEqual(cursor.fetchone()[0], 1)


class TestAlertManager(unittest.TestCase):
    """Test alert management functionality."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, 'test_alerts.db')
        self.alert_manager = AlertManager(self.db_path)
    
    def tearDown(self):
        self.alert_manager.stop_monitoring()
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_database_initialization(self):
        """Test alert database schema creation."""
        self.assertTrue(os.path.exists(self.db_path))
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='alerts'"
            )
            self.assertIsNotNone(cursor.fetchone())
    
    def test_alert_rule_management(self):
        """Test adding and removing alert rules."""
        rule = AlertRule(
            name="test_rule",
            description="Test alert rule",
            severity=AlertSeverity.HIGH,
            condition=lambda metrics: metrics.get('test_value', 0) > 100
        )
        
        self.alert_manager.add_rule(rule)
        self.assertIn("test_rule", self.alert_manager.rules)
        
        self.alert_manager.remove_rule("test_rule")
        self.assertNotIn("test_rule", self.alert_manager.rules)
    
    def test_alert_condition_triggering(self):
        """Test alert condition triggering."""
        # Add a test rule
        rule = AlertRule(
            name="high_cpu",
            description="High CPU usage",
            severity=AlertSeverity.HIGH,
            condition=lambda metrics: metrics.get('system', {}).get('avg_cpu_percent', 0) > 80
        )
        
        self.alert_manager.add_rule(rule)
        
        # Mock metrics that should trigger the alert
        mock_metrics = {
            'system': {
                'avg_cpu_percent': 85.0
            }
        }
        
        # Manually trigger condition check
        self.alert_manager._handle_condition_triggered("high_cpu", rule, mock_metrics)
        
        # Verify alert was created
        self.assertEqual(len(self.alert_manager.active_alerts), 1)
        
        # Verify alert is stored in database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM alerts")
            self.assertEqual(cursor.fetchone()[0], 1)
    
    def test_alert_acknowledgment(self):
        """Test alert acknowledgment."""
        # Create a test alert
        alert = Alert(
            id="test_alert",
            name="test_alert",
            description="Test alert",
            severity=AlertSeverity.HIGH,
            status=AlertStatus.ACTIVE,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        self.alert_manager.active_alerts["test_alert"] = alert
        self.alert_manager._store_alert_in_db(alert)
        
        # Acknowledge the alert
        success = self.alert_manager.acknowledge_alert("test_alert")
        self.assertTrue(success)
        
        # Verify alert status changed
        acknowledged_alert = self.alert_manager.active_alerts["test_alert"]
        self.assertEqual(acknowledged_alert.status, AlertStatus.ACKNOWLEDGED)
        self.assertIsNotNone(acknowledged_alert.acknowledged_at)
    
    def test_notification_handlers(self):
        """Test notification handler registration and execution."""
        notifications_sent = []
        
        def test_handler(alert: Alert):
            notifications_sent.append(alert.id)
        
        self.alert_manager.add_notification_handler(test_handler)
        
        # Create and trigger an alert
        alert = Alert(
            id="test_notification",
            name="test_notification",
            description="Test notification",
            severity=AlertSeverity.HIGH,
            status=AlertStatus.ACTIVE,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        self.alert_manager._send_notifications(alert)
        
        # Verify notification was sent
        self.assertIn("test_notification", notifications_sent)
    
    def test_alert_history(self):
        """Test alert history retrieval."""
        # Create test alerts with different timestamps
        old_alert = Alert(
            id="old_alert",
            name="old_alert",
            description="Old alert",
            severity=AlertSeverity.LOW,
            status=AlertStatus.RESOLVED,
            created_at=datetime.utcnow() - timedelta(hours=25),
            updated_at=datetime.utcnow() - timedelta(hours=25),
            resolved_at=datetime.utcnow() - timedelta(hours=24)
        )
        
        recent_alert = Alert(
            id="recent_alert",
            name="recent_alert",
            description="Recent alert",
            severity=AlertSeverity.HIGH,
            status=AlertStatus.ACTIVE,
            created_at=datetime.utcnow() - timedelta(hours=1),
            updated_at=datetime.utcnow() - timedelta(hours=1)
        )
        
        self.alert_manager._store_alert_in_db(old_alert)
        self.alert_manager._store_alert_in_db(recent_alert)
        
        # Get recent history (last 24 hours)
        history = self.alert_manager.get_alert_history(hours=24)
        
        # Should only include recent alert
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]['id'], 'recent_alert')


class TestEmailNotificationHandler(unittest.TestCase):
    """Test email notification handler."""
    
    @patch('monitoring.alerting.smtplib.SMTP')
    def test_email_notification(self, mock_smtp):
        """Test email notification sending."""
        mock_server = Mock()
        mock_smtp.return_value.__enter__.return_value = mock_server
        
        handler = EmailNotificationHandler(
            smtp_server='smtp.test.com',
            smtp_port=587,
            username='test@test.com',
            password='password',
            from_email='alerts@test.com',
            to_emails=['admin@test.com']
        )
        
        alert = Alert(
            id="test_email_alert",
            name="test_email_alert",
            description="Test email alert",
            severity=AlertSeverity.CRITICAL,
            status=AlertStatus.ACTIVE,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        handler(alert)
        
        # Verify SMTP methods were called
        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_once_with('test@test.com', 'password')
        mock_server.send_message.assert_called_once()


class TestSlackNotificationHandler(unittest.TestCase):
    """Test Slack notification handler."""
    
    @patch('requests.post')
    def test_slack_notification(self, mock_post):
        """Test Slack notification sending."""
        mock_post.return_value.raise_for_status = Mock()
        
        handler = SlackNotificationHandler('https://hooks.slack.com/test')
        
        alert = Alert(
            id="test_slack_alert",
            name="test_slack_alert",
            description="Test Slack alert",
            severity=AlertSeverity.HIGH,
            status=AlertStatus.ACTIVE,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        handler(alert)
        
        # Verify POST request was made
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        
        self.assertEqual(call_args[0][0], 'https://hooks.slack.com/test')
        self.assertIn('json', call_args[1])
        
        # Verify payload structure
        payload = call_args[1]['json']
        self.assertIn('attachments', payload)
        self.assertEqual(len(payload['attachments']), 1)
        
        attachment = payload['attachments'][0]
        self.assertEqual(attachment['title'], 'Alert: test_slack_alert')
        self.assertEqual(attachment['text'], 'Test Slack alert')


class TestMonitoringIntegration(unittest.TestCase):
    """Test integration between metrics collection and alerting."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.metrics_db = os.path.join(self.temp_dir, 'metrics.db')
        self.alerts_db = os.path.join(self.temp_dir, 'alerts.db')
        
        self.metrics_collector = MetricsCollector(self.metrics_db, collection_interval=1)
        self.alert_manager = AlertManager(self.alerts_db)
        
        # Mock the global instances
        import monitoring.metrics_collector
        import monitoring.alerting
        monitoring.metrics_collector._metrics_collector = self.metrics_collector
        monitoring.alerting._alert_manager = self.alert_manager
    
    def tearDown(self):
        self.metrics_collector.stop_collection()
        self.alert_manager.stop_monitoring()
        
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @patch('monitoring.metrics_collector.psutil')
    def test_end_to_end_monitoring(self, mock_psutil):
        """Test end-to-end monitoring workflow."""
        # Mock high CPU usage
        mock_psutil.cpu_percent.return_value = 95.0
        
        mock_memory = Mock()
        mock_memory.percent = 50.0
        mock_memory.used = 4 * 1024 * 1024 * 1024
        mock_memory.available = 4 * 1024 * 1024 * 1024
        mock_psutil.virtual_memory.return_value = mock_memory
        
        mock_disk = Mock()
        mock_disk.percent = 50.0
        mock_disk.free = 100 * 1024 * 1024 * 1024
        mock_psutil.disk_usage.return_value = mock_disk
        
        mock_network = Mock()
        mock_network.bytes_sent = 1000000
        mock_network.bytes_recv = 2000000
        mock_psutil.net_io_counters.return_value = mock_network
        
        mock_psutil.net_connections.return_value = []
        
        # Collect metrics
        system_metrics = self.metrics_collector._collect_system_metrics()
        self.metrics_collector._store_system_metrics(system_metrics)
        
        # Manually trigger the high CPU alert condition
        high_cpu_rule = self.alert_manager.rules['high_cpu_usage']
        mock_metrics = {
            'system': {
                'avg_cpu_percent': 95.0,
                'avg_memory_percent': 50.0,
                'avg_disk_percent': 50.0
            },
            'api': {
                'total_requests': 10,
                'error_rate_percent': 5.0
            }
        }
        
        # Trigger the alert condition
        self.alert_manager._handle_condition_triggered('high_cpu_usage', high_cpu_rule, mock_metrics)
        
        # Verify high CPU alert was triggered
        active_alerts = self.alert_manager.get_active_alerts()
        cpu_alerts = [alert for alert in active_alerts if 'cpu' in alert['name']]
        
        self.assertGreater(len(cpu_alerts), 0)
        self.assertEqual(cpu_alerts[0]['severity'].value, 'high')


class TestMonitoringPerformance(unittest.TestCase):
    """Test monitoring system performance."""
    
    def test_metrics_collection_performance(self):
        """Test metrics collection performance under load."""
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, 'perf_test.db')
        
        try:
            collector = MetricsCollector(db_path)
            
            start_time = time.time()
            
            # Record many API metrics
            for i in range(1000):
                api_metrics = APIMetrics(
                    timestamp=datetime.utcnow(),
                    endpoint=f'/api/endpoint/{i % 10}',
                    method='GET',
                    status_code=200 if i % 10 != 0 else 500,
                    response_time_ms=100 + (i % 50),
                    request_size_bytes=1024,
                    response_size_bytes=2048
                )
                collector.record_api_metrics(api_metrics)
            
            duration = time.time() - start_time
            
            # Should record 1000 metrics in less than 2 seconds
            self.assertLess(duration, 2.0)
            
            # Verify all metrics were stored
            with sqlite3.connect(db_path) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM api_metrics")
                self.assertEqual(cursor.fetchone()[0], 1000)
            
        finally:
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
    
    def test_alert_checking_performance(self):
        """Test alert checking performance with many rules."""
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, 'alert_perf_test.db')
        
        try:
            alert_manager = AlertManager(db_path)
            
            # Add many alert rules
            for i in range(100):
                rule = AlertRule(
                    name=f"test_rule_{i}",
                    description=f"Test rule {i}",
                    severity=AlertSeverity.LOW,
                    condition=lambda metrics, threshold=i: metrics.get('test_value', 0) > threshold
                )
                alert_manager.add_rule(rule)
            
            # Mock metrics
            test_metrics = {'test_value': 50}
            
            start_time = time.time()
            
            # Check all conditions
            for rule_name, rule in alert_manager.rules.items():
                try:
                    rule.condition(test_metrics)
                except Exception:
                    pass  # Ignore condition errors for performance test
            
            duration = time.time() - start_time
            
            # Should check 100 rules in less than 0.1 seconds
            self.assertLess(duration, 0.1)
            
        finally:
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)


if __name__ == '__main__':
    unittest.main()