# Monitoring & Logging Quick Reference

## 🚀 Quick Start

### Initialize Monitoring
```python
from monitoring.startup import start_monitoring_with_config
start_monitoring_with_config()
```

### Basic Logging
```python
from utils.logging import get_structured_logger

logger = get_structured_logger('my_service')
logger.info('Operation completed', user_id='123', duration_ms=150)
```

### Check System Health
```bash
curl http://localhost:5000/api/v1/monitoring/health
```

## 📊 Key Endpoints

| Endpoint | Purpose | Example |
|----------|---------|---------|
| `/api/v1/monitoring/health` | System health check | `GET /api/v1/monitoring/health` |
| `/api/v1/monitoring/metrics/system` | System metrics | `GET /api/v1/monitoring/metrics/system?minutes=60` |
| `/api/v1/monitoring/metrics/api` | API performance | `GET /api/v1/monitoring/metrics/api?minutes=60` |
| `/api/v1/monitoring/alerts` | Active alerts | `GET /api/v1/monitoring/alerts` |
| `/api/v1/monitoring/dashboard/overview` | Dashboard data | `GET /api/v1/monitoring/dashboard/overview` |

## 🔍 Logging Examples

### Structured Logging
```python
from utils.logging import StructuredLogger

logger = StructuredLogger('api')
logger.set_context(request_id='req-123', user_id='user-456')
logger.info('Processing request', endpoint='/generate', method='POST')
```

### Function Logging
```python
from utils.logging import log_function_call

@log_function_call
def process_data(data):
    return processed_data
```

### Error Logging
```python
from utils.logging import log_error_with_context

try:
    risky_operation()
except Exception as e:
    log_error_with_context(logger, e, {'operation': 'data_processing'})
```

## 📈 Metrics Collection

### Record API Metrics
```python
from monitoring.metrics_collector import get_metrics_collector, APIMetrics
from datetime import datetime

collector = get_metrics_collector()
metrics = APIMetrics(
    timestamp=datetime.utcnow(),
    endpoint='/api/v1/generate',
    method='POST',
    status_code=200,
    response_time_ms=150.5,
    request_size_bytes=1024,
    response_size_bytes=2048
)
collector.record_api_metrics(metrics)
```

### Get Recent Metrics
```python
# System metrics for last hour
system_metrics = collector.get_recent_system_metrics(minutes=60)

# API metrics for last hour  
api_metrics = collector.get_recent_api_metrics(minutes=60)

# Summary for last 24 hours
summary = collector.get_metrics_summary(hours=24)
```

## 🚨 Alerting

### Custom Alert Rule
```python
from monitoring.alerting import AlertRule, AlertSeverity, get_alert_manager

rule = AlertRule(
    name="custom_alert",
    description="Custom condition alert",
    severity=AlertSeverity.HIGH,
    condition=lambda metrics: metrics.get('custom_value', 0) > 100,
    threshold_duration=300  # 5 minutes
)

alert_manager = get_alert_manager()
alert_manager.add_rule(rule)
```

### Acknowledge Alert
```python
alert_manager.acknowledge_alert('alert_id_here')
```

## 🔧 Configuration

### Environment Variables
```bash
# Logging
LOG_FORMAT=json
LOG_LEVEL=INFO

# Email Alerts
ENABLE_EMAIL_ALERTS=true
SMTP_SERVER=smtp.gmail.com
SMTP_USERNAME=alerts@company.com
SMTP_PASSWORD=app_password
ALERT_TO_EMAILS=admin@company.com

# Slack Alerts  
ENABLE_SLACK_ALERTS=true
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
```

### Programmatic Configuration
```python
from monitoring.startup import initialize_monitoring_system

initialize_monitoring_system(
    enable_email_alerts=True,
    email_config={
        'smtp_server': 'smtp.gmail.com',
        'smtp_port': 587,
        'username': 'alerts@company.com',
        'password': 'password',
        'from_email': 'alerts@company.com',
        'to_emails': ['admin@company.com']
    }
)
```

## 🔍 Log Search

### Search Logs
```python
from utils.log_aggregator import LogAggregator, LogSearchQuery
from datetime import datetime, timedelta

aggregator = LogAggregator()

# Search for errors in last hour
query = LogSearchQuery(
    start_time=datetime.now() - timedelta(hours=1),
    level='ERROR',
    message_pattern='database'
)
results = aggregator.search_logs(query)
```

### Log Statistics
```python
# Get log statistics for last 24 hours
stats = aggregator.get_log_statistics(hours=24)
print(f"Total entries: {stats['total_entries']}")
print(f"Error rate: {stats['error_rate_percent']}%")
```

## 🐛 Troubleshooting

### Check Service Status
```python
from monitoring.metrics_collector import get_metrics_collector
from monitoring.alerting import get_alert_manager

collector = get_metrics_collector()
alert_manager = get_alert_manager()

print(f"Metrics collection running: {collector.running}")
print(f"Alert monitoring running: {alert_manager.running}")
print(f"Active alerts: {len(alert_manager.active_alerts)}")
```

### Debug Logging
```python
import logging
from utils.logging import setup_logging

# Enable debug logging
logger = setup_logging(level='DEBUG')
```

### Test Notifications
```bash
# Test email configuration
python -c "
from monitoring.alerting import EmailNotificationHandler, Alert, AlertSeverity, AlertStatus
from datetime import datetime

handler = EmailNotificationHandler(
    smtp_server='smtp.gmail.com',
    smtp_port=587,
    username='test@gmail.com',
    password='password',
    from_email='test@gmail.com',
    to_emails=['admin@gmail.com']
)

test_alert = Alert(
    id='test',
    name='test_alert',
    description='Test alert',
    severity=AlertSeverity.LOW,
    status=AlertStatus.ACTIVE,
    created_at=datetime.utcnow(),
    updated_at=datetime.utcnow()
)

handler(test_alert)
print('Email test completed')
"
```

## 📱 Dashboard Access

### Local Development
```
http://localhost:3000/monitoring
```

### Production
```
https://your-domain.com/monitoring
```

### API Access
```bash
# Get dashboard overview
curl -H "Content-Type: application/json" \
     http://localhost:5000/api/v1/monitoring/dashboard/overview

# Acknowledge alert
curl -X POST \
     -H "Content-Type: application/json" \
     http://localhost:5000/api/v1/monitoring/alerts/alert_id/acknowledge
```

## 🔄 Maintenance

### Cleanup Old Data
```python
from monitoring.metrics_collector import get_metrics_collector
from utils.log_aggregator import LogAggregator

# Cleanup metrics older than 7 days
collector = get_metrics_collector()
deleted_counts = collector.cleanup_old_metrics(days=7)

# Cleanup logs older than 30 days
aggregator = LogAggregator()
deleted_logs = aggregator.cleanup_old_logs(days=30)
```

### Database Maintenance
```bash
# Check database sizes
sqlite3 monitoring/metrics.db "SELECT COUNT(*) FROM system_metrics;"
sqlite3 monitoring/alerts.db "SELECT COUNT(*) FROM alerts;"
sqlite3 logs/log_aggregator.db "SELECT COUNT(*) FROM log_entries;"
```

## 📊 Performance Metrics

### Key Metrics to Monitor

| Metric | Threshold | Severity |
|--------|-----------|----------|
| CPU Usage | >80% for 5min | HIGH |
| Memory Usage | >90% for 3min | CRITICAL |
| API Error Rate | >10% for 5min | HIGH |
| Response Time | >5s for 10min | MEDIUM |
| Disk Usage | >85% for 5min | HIGH |

### Performance Benchmarks

- **Logging Overhead**: <1ms per request
- **Metrics Collection**: <50MB memory usage
- **Database Queries**: <100ms for dashboard
- **Alert Processing**: <10ms per rule check
- **Throughput**: 1000+ operations per second

## 🔐 Security Notes

### Sensitive Data Protection
- Passwords, API keys, tokens automatically redacted
- Use structured logging to avoid accidental exposure
- Configure proper access controls for monitoring endpoints

### Best Practices
- Use HTTPS in production
- Implement authentication for monitoring endpoints
- Regularly rotate notification credentials
- Monitor access to monitoring data

## 📞 Support

### Log Locations
- Application logs: `/app/logs/app.log`
- Metrics database: `/app/monitoring/metrics.db`
- Alerts database: `/app/monitoring/alerts.db`

### Common Commands
```bash
# View recent logs
tail -f logs/app.log | jq '.'

# Check system metrics
curl -s http://localhost:5000/api/v1/monitoring/metrics/summary | jq '.'

# List active alerts
curl -s http://localhost:5000/api/v1/monitoring/alerts | jq '.data[].name'
```