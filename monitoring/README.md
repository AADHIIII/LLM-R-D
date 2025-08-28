# Monitoring System

Enterprise-grade monitoring and logging system for the LLM Optimization Platform.

## 🎯 Overview

This monitoring system provides comprehensive observability with:

- **📊 Real-time Metrics**: System and API performance monitoring
- **📝 Structured Logging**: JSON-formatted logs with context correlation  
- **🚨 Intelligent Alerting**: Configurable rules with multi-channel notifications
- **📈 Interactive Dashboard**: Web-based monitoring interface
- **🔍 Log Aggregation**: Centralized log collection and search

## 🚀 Quick Start

### 1. Initialize Monitoring
```python
from monitoring.startup import start_monitoring_with_config
start_monitoring_with_config()
```

### 2. Access Dashboard
```
http://localhost:3000/monitoring
```

### 3. Check Health
```bash
curl http://localhost:5000/api/v1/monitoring/health
```

## 📁 Directory Structure

```
monitoring/
├── README.md                 # This file
├── startup.py               # System initialization
├── metrics_collector.py     # Metrics collection service
├── alerting.py             # Alert management system
└── data/                   # Database storage
    ├── metrics.db          # Metrics database
    └── alerts.db           # Alerts database
```

## 🔧 Components

### MetricsCollector
Collects system and application metrics in real-time.

```python
from monitoring.metrics_collector import get_metrics_collector

collector = get_metrics_collector()
collector.start_collection()

# Get recent metrics
metrics = collector.get_recent_system_metrics(minutes=60)
summary = collector.get_metrics_summary(hours=24)
```

### AlertManager  
Manages alert rules and notifications.

```python
from monitoring.alerting import get_alert_manager, AlertRule, AlertSeverity

alert_manager = get_alert_manager()
alert_manager.start_monitoring()

# Add custom rule
rule = AlertRule(
    name="high_error_rate",
    description="API error rate is too high",
    severity=AlertSeverity.HIGH,
    condition=lambda metrics: metrics.get('api', {}).get('error_rate_percent', 0) > 5
)
alert_manager.add_rule(rule)
```

### Startup Manager
Handles system initialization and configuration.

```python
from monitoring.startup import initialize_monitoring_system

initialize_monitoring_system(
    enable_email_alerts=True,
    email_config={
        'smtp_server': 'smtp.gmail.com',
        'smtp_port': 587,
        'username': 'alerts@company.com',
        'password': 'app_password',
        'from_email': 'alerts@company.com',
        'to_emails': ['admin@company.com']
    }
)
```

## 📊 Metrics Collected

### System Metrics
- CPU usage percentage
- Memory usage and availability  
- Disk usage and free space
- Network I/O statistics
- Active network connections

### API Metrics
- Request rate and throughput
- Response times (avg, percentiles)
- Error rates by endpoint
- Request/response sizes
- Status code distribution

### Application Metrics
- Fine-tuning job status
- Evaluation completion rates
- Cache hit rates
- Database connection pools
- Background queue sizes

## 🚨 Default Alert Rules

| Alert | Condition | Severity | Duration |
|-------|-----------|----------|----------|
| High CPU | >80% | HIGH | 5 min |
| Critical Memory | >90% | CRITICAL | 3 min |
| High Error Rate | >10% | HIGH | 5 min |
| Slow Responses | >5s avg | MEDIUM | 10 min |
| Low Disk Space | >85% | HIGH | 5 min |
| System Down | No requests | CRITICAL | 15 min |

## 🔔 Notification Channels

### Email Notifications
Configure SMTP settings for email alerts:

```bash
export ENABLE_EMAIL_ALERTS=true
export SMTP_SERVER=smtp.gmail.com
export SMTP_USERNAME=alerts@company.com
export SMTP_PASSWORD=app_password
export ALERT_TO_EMAILS=admin@company.com,ops@company.com
```

### Slack Notifications
Configure Slack webhook for team notifications:

```bash
export ENABLE_SLACK_ALERTS=true
export SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
```

## 🗄️ Database Schema

### system_metrics
```sql
CREATE TABLE system_metrics (
    id INTEGER PRIMARY KEY,
    timestamp TEXT NOT NULL,
    cpu_percent REAL,
    memory_percent REAL,
    memory_used_mb REAL,
    memory_available_mb REAL,
    disk_usage_percent REAL,
    disk_free_gb REAL,
    network_bytes_sent INTEGER,
    network_bytes_recv INTEGER,
    active_connections INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
```

### api_metrics
```sql
CREATE TABLE api_metrics (
    id INTEGER PRIMARY KEY,
    timestamp TEXT NOT NULL,
    endpoint TEXT,
    method TEXT,
    status_code INTEGER,
    response_time_ms REAL,
    request_size_bytes INTEGER,
    response_size_bytes INTEGER,
    error_count INTEGER DEFAULT 0,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
```

### alerts
```sql
CREATE TABLE alerts (
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
);
```

## 🔧 Configuration

### Environment Variables
```bash
# Metrics Collection
METRICS_COLLECTION_INTERVAL=30     # seconds
METRICS_DB_PATH=monitoring/metrics.db

# Alert Management  
ALERT_CHECK_INTERVAL=60           # seconds
ALERT_COOLDOWN_PERIOD=1800        # seconds
ALERTS_DB_PATH=monitoring/alerts.db

# Notifications
ENABLE_EMAIL_ALERTS=false
ENABLE_SLACK_ALERTS=false
```

### Programmatic Configuration
```python
from monitoring.startup import initialize_monitoring_system

config = {
    'metrics': {
        'collection_interval': 30,
        'retention_days': 7
    },
    'alerts': {
        'check_interval': 60,
        'cooldown_period': 1800
    }
}

initialize_monitoring_system(**config)
```

## 🧪 Testing

Run the monitoring system tests:

```bash
# Run all monitoring tests
python -m pytest tests/test_monitoring_system.py -v

# Run specific test categories
python -m pytest tests/test_monitoring_system.py::TestMetricsCollector -v
python -m pytest tests/test_monitoring_system.py::TestAlertManager -v
python -m pytest tests/test_monitoring_system.py::TestMonitoringPerformance -v
```

### Test Coverage
- ✅ Metrics collection accuracy
- ✅ Alert rule evaluation  
- ✅ Notification delivery
- ✅ Database operations
- ✅ Performance benchmarks
- ✅ Error handling
- ✅ Integration workflows

## 🚀 Performance

### Benchmarks
- **Metrics Collection**: <1ms overhead per request
- **Database Writes**: 1000+ metrics per second
- **Alert Evaluation**: <10ms per rule check
- **Memory Usage**: <50MB for monitoring services
- **Query Performance**: <100ms for dashboard queries

### Optimization Tips
- Adjust collection intervals for production load
- Use database cleanup for long-term deployments
- Configure appropriate retention policies
- Monitor the monitoring system itself

## 🔍 Troubleshooting

### Common Issues

#### Metrics Not Collecting
```python
from monitoring.metrics_collector import get_metrics_collector

collector = get_metrics_collector()
print(f"Running: {collector.running}")
print(f"Recent metrics: {len(collector.system_metrics)}")
```

#### Alerts Not Firing
```python
from monitoring.alerting import get_alert_manager

alert_manager = get_alert_manager()
print(f"Running: {alert_manager.running}")
print(f"Active rules: {len(alert_manager.rules)}")
print(f"Active alerts: {len(alert_manager.active_alerts)}")
```

#### Database Issues
```bash
# Check database files
ls -la monitoring/*.db

# Check database integrity
sqlite3 monitoring/metrics.db "PRAGMA integrity_check;"
```

### Debug Mode
```python
import logging
from utils.logging import setup_logging

# Enable debug logging
logger = setup_logging(level='DEBUG')
```

## 📚 Documentation

- **[Complete Guide](../docs/MONITORING_AND_LOGGING_GUIDE.md)**: Comprehensive documentation
- **[Quick Reference](../docs/MONITORING_QUICK_REFERENCE.md)**: Developer quick reference
- **[API Reference](../docs/MONITORING_AND_LOGGING_GUIDE.md#api-reference)**: REST API documentation

## 🤝 Contributing

### Adding Custom Metrics
```python
from monitoring.metrics_collector import get_metrics_collector, ApplicationMetrics
from datetime import datetime

collector = get_metrics_collector()

# Record custom application metrics
app_metrics = ApplicationMetrics(
    timestamp=datetime.utcnow(),
    active_fine_tuning_jobs=5,
    completed_evaluations=120,
    total_api_calls=1500,
    cache_hit_rate=0.85,
    database_connections=10,
    queue_size=25
)

collector._store_app_metrics(app_metrics)
```

### Adding Custom Alert Rules
```python
from monitoring.alerting import AlertRule, AlertSeverity, get_alert_manager

def custom_condition(metrics):
    """Custom alert condition logic."""
    app_metrics = metrics.get('app', {})
    return app_metrics.get('queue_size', 0) > 100

rule = AlertRule(
    name="high_queue_size",
    description="Background queue is backing up",
    severity=AlertSeverity.MEDIUM,
    condition=custom_condition,
    threshold_duration=300,  # 5 minutes
    cooldown_period=1800     # 30 minutes
)

alert_manager = get_alert_manager()
alert_manager.add_rule(rule)
```

### Adding Notification Channels
```python
from monitoring.alerting import get_alert_manager

def custom_notification_handler(alert):
    """Custom notification handler."""
    # Send to custom service (PagerDuty, Teams, etc.)
    print(f"Custom notification: {alert.name} - {alert.description}")

alert_manager = get_alert_manager()
alert_manager.add_notification_handler(custom_notification_handler)
```

## 📄 License

This monitoring system is part of the LLM Optimization Platform and follows the same licensing terms.

## 🆘 Support

For issues and questions:

1. Check the [troubleshooting guide](../docs/MONITORING_AND_LOGGING_GUIDE.md#troubleshooting)
2. Review the [complete documentation](../docs/MONITORING_AND_LOGGING_GUIDE.md)
3. Run the test suite to verify functionality
4. Check system logs for error details

---

**Built with ❤️ for reliable LLM operations**