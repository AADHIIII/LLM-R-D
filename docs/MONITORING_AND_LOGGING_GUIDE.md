# Monitoring and Logging System Documentation

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Logging System](#logging-system)
4. [Monitoring System](#monitoring-system)
5. [Alerting System](#alerting-system)
6. [Dashboard](#dashboard)
7. [Configuration](#configuration)
8. [API Reference](#api-reference)
9. [Deployment](#deployment)
10. [Troubleshooting](#troubleshooting)

## Overview

The LLM Optimization Platform includes a comprehensive monitoring and logging system that provides:

- **Structured JSON Logging**: Machine-readable logs with context correlation
- **Real-time Metrics Collection**: System and application performance monitoring
- **Intelligent Alerting**: Configurable rules with multi-channel notifications
- **Interactive Dashboard**: Web-based monitoring interface
- **Performance Analytics**: Historical data analysis and trend detection

### Key Features
- 🔍 **Observability**: Complete visibility into system behavior
- 📊 **Real-time Monitoring**: Live metrics and performance tracking
- 🚨 **Proactive Alerting**: Early warning system for issues
- 📈 **Analytics**: Historical trends and performance analysis
- 🔒 **Security**: Automatic sensitive data redaction
- ⚡ **Performance**: High-throughput processing (1000+ ops/second)

## Architecture

### System Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Application   │    │   Monitoring    │    │   Dashboard     │
│                 │    │   Services      │    │                 │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │   Logging   │ │───▶│ │   Metrics   │ │───▶│ │    Web UI   │ │
│ │ Middleware  │ │    │ │ Collector   │ │    │ │             │ │
│ └─────────────┘ │    │ └─────────────┘ │    │ └─────────────┘ │
│                 │    │ ┌─────────────┐ │    │ ┌─────────────┐ │
│ ┌─────────────┐ │    │ │   Alert     │ │    │ │   API       │ │
│ │   API       │ │───▶│ │  Manager    │ │───▶│ │ Endpoints   │ │
│ │ Endpoints   │ │    │ └─────────────┘ │    │ └─────────────┘ │
│ └─────────────┘ │    │ ┌─────────────┐ │    └─────────────────┘
└─────────────────┘    │ │    Log      │ │              │
                       │ │ Aggregator  │ │              │
                       │ └─────────────┘ │              │
                       └─────────────────┘              │
                                │                       │
                                ▼                       │
                       ┌─────────────────┐              │
                       │   Notifications │              │
                       │                 │              │
                       │ ┌─────────────┐ │              │
                       │ │    Email    │ │              │
                       │ └─────────────┘ │              │
                       │ ┌─────────────┐ │              │
                       │ │    Slack    │ │              │
                       │ └─────────────┘ │              │
                       └─────────────────┘              │
                                                        │
                       ┌─────────────────┐              │
                       │    Database     │◀─────────────┘
                       │                 │
                       │ ┌─────────────┐ │
                       │ │   Metrics   │ │
                       │ │   Storage   │ │
                       │ └─────────────┘ │
                       │ ┌─────────────┐ │
                       │ │    Logs     │ │
                       │ │   Storage   │ │
                       │ └─────────────┘ │
                       │ ┌─────────────┐ │
                       │ │   Alerts    │ │
                       │ │   Storage   │ │
                       │ └─────────────┘ │
                       └─────────────────┘
```

### Data Flow

1. **Request Processing**: API requests generate logs and metrics
2. **Collection**: Background services collect system and application metrics
3. **Storage**: Data is stored in SQLite databases with proper indexing
4. **Analysis**: Alert rules evaluate metrics for threshold violations
5. **Notification**: Alerts trigger notifications via configured channels
6. **Visualization**: Dashboard displays real-time data and historical trends

## Logging System

### Structured Logging

The system uses structured JSON logging for machine readability and efficient processing.

#### JSONFormatter

```python
from utils.logging import JSONFormatter, setup_logging

# Configure JSON logging
logger = setup_logging(json_format=True)
```

**Example JSON Log Entry:**
```json
{
  "timestamp": "2024-01-15T10:30:45.123Z",
  "level": "INFO",
  "logger": "api.requests",
  "message": "Request completed",
  "module": "logging_middleware",
  "function": "log_request_end",
  "line": 85,
  "thread": 140234567890,
  "process": 12345,
  "request_id": "req-abc123",
  "method": "POST",
  "path": "/api/v1/generate",
  "status_code": 200,
  "duration_ms": 150.5,
  "response_size": 2048
}
```

#### StructuredLogger

Enhanced logger with context management:

```python
from utils.logging import StructuredLogger

logger = StructuredLogger('my_service')

# Set context for all subsequent logs
logger.set_context(user_id='user123', session_id='sess456')

# Log with additional fields
logger.info('User action completed', action='file_upload', file_size=1024)

# Clear context
logger.clear_context()
```

#### Function Call Logging

Automatic logging decorator for function execution:

```python
from utils.logging import log_function_call

@log_function_call
def process_data(data):
    # Function implementation
    return processed_data
```

### Request/Response Logging

The logging middleware automatically captures:

- **Request Details**: Method, path, headers, body (sanitized)
- **Response Details**: Status code, size, duration
- **Performance Metrics**: Response times, slow request detection
- **Error Context**: Full stack traces with request correlation

#### Sensitive Data Protection

Automatic redaction of sensitive fields:

```python
# These fields are automatically redacted in logs
SENSITIVE_FIELDS = {
    'password', 'api_key', 'token', 'secret', 'authorization',
    'credit_card', 'ssn', 'social_security', 'private_key'
}
```

### Log Aggregation

#### LogAggregator Class

Centralized log collection and search:

```python
from utils.log_aggregator import LogAggregator, LogSearchQuery

aggregator = LogAggregator()

# Ingest log files
aggregator.ingest_log_file('logs/app.log')

# Search logs
query = LogSearchQuery(
    start_time=datetime.now() - timedelta(hours=1),
    level='ERROR',
    message_pattern='database'
)
results = aggregator.search_logs(query)

# Get statistics
stats = aggregator.get_log_statistics(hours=24)
```

#### Search Capabilities

- **Time Range Filtering**: Search within specific time periods
- **Level Filtering**: Filter by log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- **Pattern Matching**: Search message content with regex patterns
- **Request Correlation**: Find all logs for a specific request ID
- **Pagination**: Handle large result sets efficiently

## Monitoring System

### Metrics Collection

#### System Metrics

Automatically collected every 30 seconds:

- **CPU Usage**: Percentage utilization
- **Memory Usage**: Used/available memory and percentage
- **Disk Usage**: Used space and percentage
- **Network I/O**: Bytes sent/received
- **Active Connections**: Number of network connections

#### API Metrics

Collected for every request:

- **Request Rate**: Requests per second
- **Response Time**: Average and percentile response times
- **Error Rate**: Percentage of failed requests
- **Throughput**: Requests processed per time period
- **Endpoint Usage**: Most frequently used endpoints

#### Application Metrics

Domain-specific metrics:

- **Fine-tuning Jobs**: Active and completed jobs
- **Evaluations**: Completed evaluation runs
- **Cache Performance**: Hit rates and efficiency
- **Database Connections**: Active connection pool usage
- **Queue Size**: Background task queue depth

### MetricsCollector Class

```python
from monitoring.metrics_collector import MetricsCollector, APIMetrics

collector = MetricsCollector()

# Start background collection
collector.start_collection()

# Record API metrics
api_metrics = APIMetrics(
    timestamp=datetime.utcnow(),
    endpoint='/api/v1/generate',
    method='POST',
    status_code=200,
    response_time_ms=150.5,
    request_size_bytes=1024,
    response_size_bytes=2048
)
collector.record_api_metrics(api_metrics)

# Get recent metrics
recent_metrics = collector.get_recent_system_metrics(minutes=60)
summary = collector.get_metrics_summary(hours=24)
```

### Performance Characteristics

- **Collection Overhead**: <1ms per request
- **Storage Efficiency**: Compressed time-series data
- **Query Performance**: <100ms for typical dashboard queries
- **Memory Usage**: <50MB for monitoring services
- **Throughput**: 1000+ metrics per second

## Alerting System

### Alert Rules

Pre-configured alert rules monitor system health:

#### Default Alert Rules

1. **High CPU Usage**
   - Condition: CPU > 80% for 5 minutes
   - Severity: HIGH
   - Cooldown: 30 minutes

2. **Critical Memory Usage**
   - Condition: Memory > 90% for 3 minutes
   - Severity: CRITICAL
   - Cooldown: 30 minutes

3. **High API Error Rate**
   - Condition: Error rate > 10% for 5 minutes
   - Severity: HIGH
   - Cooldown: 30 minutes

4. **Slow Response Times**
   - Condition: Avg response time > 5s for 10 minutes
   - Severity: MEDIUM
   - Cooldown: 30 minutes

5. **Low Disk Space**
   - Condition: Disk usage > 85% for 5 minutes
   - Severity: HIGH
   - Cooldown: 30 minutes

6. **System Unavailability**
   - Condition: No requests for 15 minutes
   - Severity: CRITICAL
   - Cooldown: 30 minutes

### Custom Alert Rules

```python
from monitoring.alerting import AlertRule, AlertSeverity, get_alert_manager

# Define custom rule
custom_rule = AlertRule(
    name="high_fine_tuning_failures",
    description="High rate of fine-tuning job failures",
    severity=AlertSeverity.HIGH,
    condition=lambda metrics: metrics.get('app', {}).get('failed_jobs', 0) > 5,
    threshold_duration=600,  # 10 minutes
    cooldown_period=1800     # 30 minutes
)

# Add to alert manager
alert_manager = get_alert_manager()
alert_manager.add_rule(custom_rule)
```

### Alert Lifecycle

1. **Triggered**: Condition met for threshold duration
2. **Active**: Alert is active and notifications sent
3. **Acknowledged**: Human acknowledges awareness
4. **Resolved**: Condition no longer met

### Notification Channels

#### Email Notifications

```python
from monitoring.alerting import EmailNotificationHandler

email_handler = EmailNotificationHandler(
    smtp_server='smtp.gmail.com',
    smtp_port=587,
    username='alerts@company.com',
    password='app_password',
    from_email='alerts@company.com',
    to_emails=['admin@company.com', 'ops@company.com']
)

alert_manager.add_notification_handler(email_handler)
```

#### Slack Notifications

```python
from monitoring.alerting import SlackNotificationHandler

slack_handler = SlackNotificationHandler(
    webhook_url='https://hooks.slack.com/services/...'
)

alert_manager.add_notification_handler(slack_handler)
```

## Dashboard

### Web Interface

The React-based dashboard provides:

- **Real-time Updates**: Auto-refresh every 30 seconds
- **System Overview**: Key metrics and health status
- **Performance Charts**: Interactive visualizations
- **Alert Management**: View and acknowledge alerts
- **Historical Data**: Trend analysis and reporting

### Dashboard Components

#### System Health Cards

```tsx
// CPU Usage Card
<Card>
  <div className="p-6">
    <div className="flex items-center">
      <div className="w-8 h-8 bg-blue-100 rounded-full">
        <span className="text-blue-600">CPU</span>
      </div>
      <div className="ml-5">
        <dt className="text-gray-500">CPU Usage</dt>
        <dd className="text-gray-900">{cpuUsage.toFixed(1)}%</dd>
      </div>
    </div>
  </div>
</Card>
```

#### Metrics Visualization

```tsx
// System Metrics Chart
<div className="h-64 flex items-end space-x-1">
  {systemMetrics.slice(-20).map((metric, index) => (
    <div key={index} className="flex-1 flex flex-col">
      <div 
        className="bg-blue-500 rounded-full"
        style={{ height: `${metric.cpu_percent}%` }}
        title={`CPU: ${metric.cpu_percent.toFixed(1)}%`}
      />
    </div>
  ))}
</div>
```

#### Alert Management

```tsx
// Alert List Component
{alerts.map((alert) => (
  <div key={alert.id} className="border rounded-lg p-4">
    <div className="flex justify-between">
      <div>
        <span className={getSeverityColor(alert.severity)}>
          {alert.severity.toUpperCase()}
        </span>
        <h4>{alert.name}</h4>
        <p>{alert.description}</p>
      </div>
      <button onClick={() => acknowledgeAlert(alert.id)}>
        Acknowledge
      </button>
    </div>
  </div>
))}
```

### Dashboard Features

- **Responsive Design**: Works on desktop and mobile
- **Real-time Data**: Live updates without page refresh
- **Interactive Charts**: Hover for detailed information
- **Alert Actions**: Acknowledge alerts directly from dashboard
- **Export Capabilities**: Download metrics data
- **Customizable Views**: Filter and sort data

## Configuration

### Environment Variables

```bash
# Logging Configuration
LOG_FORMAT=json                    # Enable JSON logging
LOG_LEVEL=INFO                     # Set logging level
LOG_FILE=/app/logs/app.log         # Log file path

# Metrics Collection
METRICS_COLLECTION_INTERVAL=30     # Collection interval in seconds
METRICS_DB_PATH=/app/data/metrics.db

# Email Alerts
ENABLE_EMAIL_ALERTS=true
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=alerts@company.com
SMTP_PASSWORD=app_password
ALERT_FROM_EMAIL=alerts@company.com
ALERT_TO_EMAILS=admin@company.com,ops@company.com

# Slack Alerts
ENABLE_SLACK_ALERTS=true
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...

# Alert Configuration
ALERT_CHECK_INTERVAL=60            # Alert check interval in seconds
ALERT_COOLDOWN_PERIOD=1800         # Default cooldown in seconds
```

### Application Configuration

```python
# config/monitoring.py
MONITORING_CONFIG = {
    'metrics': {
        'collection_interval': 30,
        'retention_days': 7,
        'db_path': 'monitoring/metrics.db'
    },
    'logging': {
        'level': 'INFO',
        'format': 'json',
        'file': 'logs/app.log',
        'max_size': '10MB',
        'backup_count': 5
    },
    'alerts': {
        'check_interval': 60,
        'default_cooldown': 1800,
        'db_path': 'monitoring/alerts.db'
    }
}
```

### Startup Configuration

```python
from monitoring.startup import initialize_monitoring_system

# Initialize with configuration
initialize_monitoring_system(
    enable_email_alerts=True,
    email_config={
        'smtp_server': 'smtp.gmail.com',
        'smtp_port': 587,
        'username': 'alerts@company.com',
        'password': 'app_password',
        'from_email': 'alerts@company.com',
        'to_emails': ['admin@company.com']
    },
    enable_slack_alerts=True,
    slack_webhook_url='https://hooks.slack.com/services/...'
)
```

## API Reference

### Monitoring Endpoints

#### Health Check
```http
GET /api/v1/monitoring/health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:45.123Z",
  "services": {
    "metrics_collector": {
      "running": true,
      "last_collection": "2024-01-15T10:30:00.000Z"
    },
    "alert_manager": {
      "running": true,
      "active_alerts": 2
    }
  }
}
```

#### System Metrics
```http
GET /api/v1/monitoring/metrics/system?minutes=60
```

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "timestamp": "2024-01-15T10:30:00.000Z",
      "cpu_percent": 45.2,
      "memory_percent": 67.8,
      "disk_usage_percent": 23.4,
      "network_bytes_sent": 1048576,
      "network_bytes_recv": 2097152,
      "active_connections": 42
    }
  ],
  "count": 120,
  "time_range_minutes": 60
}
```

#### API Metrics
```http
GET /api/v1/monitoring/metrics/api?minutes=60
```

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "timestamp": "2024-01-15T10:30:00.000Z",
      "endpoint": "/api/v1/generate",
      "method": "POST",
      "status_code": 200,
      "response_time_ms": 150.5,
      "request_size_bytes": 1024,
      "response_size_bytes": 2048,
      "error_count": 0
    }
  ],
  "count": 500,
  "time_range_minutes": 60
}
```

#### Metrics Summary
```http
GET /api/v1/monitoring/metrics/summary?hours=24
```

**Response:**
```json
{
  "success": true,
  "data": {
    "time_period_hours": 24,
    "system": {
      "avg_cpu_percent": 45.2,
      "max_cpu_percent": 78.9,
      "avg_memory_percent": 67.8,
      "max_memory_percent": 89.1,
      "avg_disk_percent": 23.4,
      "data_points": 2880
    },
    "api": {
      "total_requests": 15420,
      "avg_response_time_ms": 145.7,
      "max_response_time_ms": 2340.1,
      "error_count": 23,
      "error_rate_percent": 0.15,
      "unique_endpoints": 12
    }
  }
}
```

#### Active Alerts
```http
GET /api/v1/monitoring/alerts
```

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "id": "high_cpu_usage_20240115",
      "name": "high_cpu_usage",
      "description": "CPU usage is consistently high",
      "severity": "high",
      "status": "active",
      "created_at": "2024-01-15T10:25:00.000Z",
      "updated_at": "2024-01-15T10:30:00.000Z",
      "metadata": {
        "metrics": {
          "system": {
            "avg_cpu_percent": 85.2
          }
        }
      }
    }
  ],
  "count": 1
}
```

#### Acknowledge Alert
```http
POST /api/v1/monitoring/alerts/{alert_id}/acknowledge
```

**Response:**
```json
{
  "success": true,
  "message": "Alert high_cpu_usage_20240115 acknowledged"
}
```

#### Dashboard Overview
```http
GET /api/v1/monitoring/dashboard/overview
```

**Response:**
```json
{
  "success": true,
  "data": {
    "timestamp": "2024-01-15T10:30:45.123Z",
    "system_health": {
      "status": "healthy",
      "cpu_usage": 45.2,
      "memory_usage": 67.8,
      "disk_usage": 23.4
    },
    "api_performance": {
      "total_requests": 1542,
      "avg_response_time": 145.7,
      "error_rate": 0.15,
      "unique_endpoints": 12
    },
    "alerts": {
      "total_active": 1,
      "by_severity": {
        "high": 1
      },
      "critical_count": 0,
      "high_count": 1
    },
    "trends": {
      "cpu_trend": "stable",
      "memory_trend": "increasing",
      "avg_cpu_last_hour": 45.2,
      "avg_memory_last_hour": 67.8
    }
  }
}
```

## Deployment

### Docker Configuration

#### Dockerfile Additions

```dockerfile
# Install monitoring dependencies
RUN pip install psutil

# Create monitoring directories
RUN mkdir -p /app/monitoring /app/logs

# Copy monitoring configuration
COPY monitoring/ /app/monitoring/
COPY config/monitoring.py /app/config/

# Expose monitoring port (if separate service)
EXPOSE 8080

# Health check for monitoring
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:5000/api/v1/monitoring/health || exit 1
```

#### Docker Compose

```yaml
version: '3.8'
services:
  llm-platform:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./logs:/app/logs
      - ./monitoring/data:/app/monitoring/data
    environment:
      - LOG_FORMAT=json
      - LOG_LEVEL=INFO
      - ENABLE_EMAIL_ALERTS=true
      - SMTP_SERVER=smtp.gmail.com
      - SMTP_USERNAME=${SMTP_USERNAME}
      - SMTP_PASSWORD=${SMTP_PASSWORD}
      - ALERT_TO_EMAILS=${ALERT_TO_EMAILS}
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/api/v1/monitoring/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

### Kubernetes Deployment

#### Monitoring ConfigMap

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: monitoring-config
data:
  monitoring.yaml: |
    metrics:
      collection_interval: 30
      retention_days: 7
    logging:
      level: INFO
      format: json
    alerts:
      check_interval: 60
      default_cooldown: 1800
```

#### Deployment with Monitoring

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-platform
spec:
  replicas: 3
  selector:
    matchLabels:
      app: llm-platform
  template:
    metadata:
      labels:
        app: llm-platform
    spec:
      containers:
      - name: llm-platform
        image: llm-platform:latest
        ports:
        - containerPort: 5000
        env:
        - name: LOG_FORMAT
          value: "json"
        - name: LOG_LEVEL
          value: "INFO"
        - name: ENABLE_EMAIL_ALERTS
          value: "true"
        - name: SMTP_USERNAME
          valueFrom:
            secretKeyRef:
              name: email-credentials
              key: username
        - name: SMTP_PASSWORD
          valueFrom:
            secretKeyRef:
              name: email-credentials
              key: password
        volumeMounts:
        - name: logs
          mountPath: /app/logs
        - name: monitoring-data
          mountPath: /app/monitoring/data
        - name: config
          mountPath: /app/config/monitoring.yaml
          subPath: monitoring.yaml
        livenessProbe:
          httpGet:
            path: /api/v1/monitoring/health
            port: 5000
          initialDelaySeconds: 30
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /api/v1/monitoring/health
            port: 5000
          initialDelaySeconds: 5
          periodSeconds: 10
      volumes:
      - name: logs
        persistentVolumeClaim:
          claimName: logs-pvc
      - name: monitoring-data
        persistentVolumeClaim:
          claimName: monitoring-data-pvc
      - name: config
        configMap:
          name: monitoring-config
```

### Production Considerations

#### Performance Tuning

```python
# config/production.py
MONITORING_CONFIG = {
    'metrics': {
        'collection_interval': 60,  # Reduce frequency in production
        'retention_days': 30,       # Longer retention
        'batch_size': 100          # Batch database writes
    },
    'logging': {
        'level': 'WARNING',        # Reduce log volume
        'async_logging': True,     # Non-blocking logging
        'buffer_size': 1000       # Buffer log writes
    },
    'alerts': {
        'check_interval': 120,     # Less frequent checks
        'rate_limiting': True,     # Prevent alert spam
        'max_alerts_per_hour': 10 # Rate limit
    }
}
```

#### Security Configuration

```python
# Secure configuration
MONITORING_SECURITY = {
    'api_authentication': True,
    'encrypt_sensitive_data': True,
    'audit_logging': True,
    'access_control': {
        'read_metrics': ['admin', 'operator'],
        'acknowledge_alerts': ['admin', 'operator'],
        'manage_rules': ['admin']
    }
}
```

## Troubleshooting

### Common Issues

#### High Memory Usage

**Symptoms:**
- Memory usage alerts
- Slow response times
- Application crashes

**Diagnosis:**
```bash
# Check memory metrics
curl http://localhost:5000/api/v1/monitoring/metrics/system?minutes=60

# Check for memory leaks in logs
grep -i "memory\|leak\|oom" logs/app.log
```

**Solutions:**
- Increase memory limits
- Optimize data retention policies
- Check for memory leaks in application code

#### Missing Metrics

**Symptoms:**
- Empty dashboard
- No recent metrics data

**Diagnosis:**
```python
from monitoring.metrics_collector import get_metrics_collector

collector = get_metrics_collector()
print(f"Collection running: {collector.running}")
print(f"Recent metrics count: {len(collector.system_metrics)}")
```

**Solutions:**
- Restart metrics collection service
- Check database permissions
- Verify system dependencies (psutil)

#### Alert Notification Failures

**Symptoms:**
- Alerts not received via email/Slack
- Error logs about notification failures

**Diagnosis:**
```bash
# Check alert manager logs
grep -i "notification\|email\|slack" logs/app.log

# Test SMTP connection
python -c "
import smtplib
server = smtplib.SMTP('smtp.gmail.com', 587)
server.starttls()
server.login('user', 'pass')
print('SMTP connection successful')
"
```

**Solutions:**
- Verify SMTP credentials
- Check firewall rules
- Test webhook URLs
- Review notification handler configuration

#### Database Performance Issues

**Symptoms:**
- Slow dashboard loading
- High CPU usage from database queries

**Diagnosis:**
```sql
-- Check database size
SELECT 
  name,
  COUNT(*) as record_count,
  SUM(LENGTH(data)) as size_bytes
FROM (
  SELECT 'system_metrics' as name, * FROM system_metrics
  UNION ALL
  SELECT 'api_metrics' as name, * FROM api_metrics
  UNION ALL
  SELECT 'alerts' as name, * FROM alerts
) GROUP BY name;

-- Check query performance
EXPLAIN QUERY PLAN SELECT * FROM system_metrics 
WHERE timestamp >= datetime('now', '-1 hour');
```

**Solutions:**
- Run database cleanup
- Optimize retention policies
- Add missing indexes
- Consider database partitioning

### Debug Mode

Enable debug logging for troubleshooting:

```python
import logging
from utils.logging import setup_logging

# Enable debug logging
logger = setup_logging(level='DEBUG')

# Enable SQL query logging
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)
```

### Health Checks

Verify system health:

```bash
# Check monitoring service health
curl http://localhost:5000/api/v1/monitoring/health

# Check individual components
curl http://localhost:5000/api/v1/monitoring/status

# Verify database connectivity
python -c "
from monitoring.metrics_collector import MetricsCollector
collector = MetricsCollector()
print('Database connection successful')
"
```

### Performance Monitoring

Monitor the monitoring system itself:

```python
import time
from monitoring.metrics_collector import get_metrics_collector

collector = get_metrics_collector()

# Measure collection performance
start_time = time.time()
metrics = collector._collect_system_metrics()
collection_time = time.time() - start_time

print(f"Collection time: {collection_time:.3f}s")
print(f"Memory usage: {len(collector.system_metrics)} entries")
```

## Best Practices

### Logging Best Practices

1. **Use Structured Logging**: Always use JSON format for production
2. **Include Context**: Add request IDs and user context
3. **Sanitize Sensitive Data**: Never log passwords or API keys
4. **Log at Appropriate Levels**: Use DEBUG for development, INFO+ for production
5. **Include Timing Information**: Log execution times for performance analysis

### Monitoring Best Practices

1. **Monitor Key Metrics**: Focus on user-impacting metrics
2. **Set Meaningful Thresholds**: Avoid alert fatigue with proper thresholds
3. **Use Percentiles**: Monitor 95th/99th percentiles, not just averages
4. **Implement Health Checks**: Provide endpoints for external monitoring
5. **Plan for Scale**: Design for high-volume environments

### Alerting Best Practices

1. **Alert on Symptoms**: Alert on user impact, not just technical metrics
2. **Provide Context**: Include relevant information in alert messages
3. **Implement Escalation**: Have multiple notification channels
4. **Document Runbooks**: Provide clear resolution steps
5. **Review and Tune**: Regularly review alert effectiveness

### Security Best Practices

1. **Encrypt Sensitive Data**: Protect credentials and personal information
2. **Implement Access Control**: Restrict monitoring data access
3. **Audit Access**: Log who accesses monitoring data
4. **Secure Communications**: Use HTTPS for all monitoring endpoints
5. **Regular Updates**: Keep monitoring dependencies updated

This comprehensive documentation provides everything needed to understand, deploy, and maintain the monitoring and logging system for the LLM Optimization Platform.