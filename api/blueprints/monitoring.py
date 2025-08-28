"""
Monitoring dashboard API endpoints.
"""

from flask import Blueprint, jsonify, request
from datetime import datetime, timedelta
from typing import Dict, Any

from monitoring.metrics_collector import get_metrics_collector
from monitoring.alerting import get_alert_manager
from utils.logging import get_structured_logger

monitoring_bp = Blueprint('monitoring', __name__)
logger = get_structured_logger('monitoring_api')


@monitoring_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring dashboard."""
    try:
        metrics_collector = get_metrics_collector()
        alert_manager = get_alert_manager()
        
        # Check if services are running
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'services': {
                'metrics_collector': {
                    'running': metrics_collector.running,
                    'last_collection': None
                },
                'alert_manager': {
                    'running': alert_manager.running,
                    'active_alerts': len(alert_manager.active_alerts)
                }
            }
        }
        
        # Get last system metrics to check if collection is working
        recent_metrics = metrics_collector.get_recent_system_metrics(minutes=5)
        if recent_metrics:
            health_status['services']['metrics_collector']['last_collection'] = recent_metrics[-1]['timestamp']
        
        return jsonify(health_status)
        
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500


@monitoring_bp.route('/metrics/system', methods=['GET'])
def get_system_metrics():
    """Get system performance metrics."""
    try:
        minutes = request.args.get('minutes', 60, type=int)
        
        if minutes > 1440:  # Limit to 24 hours
            minutes = 1440
        
        metrics_collector = get_metrics_collector()
        metrics = metrics_collector.get_recent_system_metrics(minutes=minutes)
        
        return jsonify({
            'success': True,
            'data': metrics,
            'count': len(metrics),
            'time_range_minutes': minutes
        })
        
    except Exception as e:
        logger.error("Failed to get system metrics", error=str(e))
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@monitoring_bp.route('/metrics/api', methods=['GET'])
def get_api_metrics():
    """Get API performance metrics."""
    try:
        minutes = request.args.get('minutes', 60, type=int)
        
        if minutes > 1440:  # Limit to 24 hours
            minutes = 1440
        
        metrics_collector = get_metrics_collector()
        metrics = metrics_collector.get_recent_api_metrics(minutes=minutes)
        
        return jsonify({
            'success': True,
            'data': metrics,
            'count': len(metrics),
            'time_range_minutes': minutes
        })
        
    except Exception as e:
        logger.error("Failed to get API metrics", error=str(e))
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@monitoring_bp.route('/metrics/summary', methods=['GET'])
def get_metrics_summary():
    """Get metrics summary for dashboard overview."""
    try:
        hours = request.args.get('hours', 24, type=int)
        
        if hours > 168:  # Limit to 7 days
            hours = 168
        
        metrics_collector = get_metrics_collector()
        summary = metrics_collector.get_metrics_summary(hours=hours)
        
        return jsonify({
            'success': True,
            'data': summary
        })
        
    except Exception as e:
        logger.error("Failed to get metrics summary", error=str(e))
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@monitoring_bp.route('/alerts', methods=['GET'])
def get_alerts():
    """Get active alerts."""
    try:
        alert_manager = get_alert_manager()
        active_alerts = alert_manager.get_active_alerts()
        
        return jsonify({
            'success': True,
            'data': active_alerts,
            'count': len(active_alerts)
        })
        
    except Exception as e:
        logger.error("Failed to get alerts", error=str(e))
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@monitoring_bp.route('/alerts/history', methods=['GET'])
def get_alert_history():
    """Get alert history."""
    try:
        hours = request.args.get('hours', 24, type=int)
        
        if hours > 168:  # Limit to 7 days
            hours = 168
        
        alert_manager = get_alert_manager()
        history = alert_manager.get_alert_history(hours=hours)
        
        return jsonify({
            'success': True,
            'data': history,
            'count': len(history),
            'time_range_hours': hours
        })
        
    except Exception as e:
        logger.error("Failed to get alert history", error=str(e))
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@monitoring_bp.route('/alerts/<alert_id>/acknowledge', methods=['POST'])
def acknowledge_alert(alert_id: str):
    """Acknowledge an alert."""
    try:
        alert_manager = get_alert_manager()
        success = alert_manager.acknowledge_alert(alert_id)
        
        if success:
            logger.info(f"Alert acknowledged via API", alert_id=alert_id)
            return jsonify({
                'success': True,
                'message': f'Alert {alert_id} acknowledged'
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Alert {alert_id} not found or already resolved'
            }), 404
        
    except Exception as e:
        logger.error("Failed to acknowledge alert", alert_id=alert_id, error=str(e))
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@monitoring_bp.route('/dashboard/overview', methods=['GET'])
def get_dashboard_overview():
    """Get comprehensive dashboard overview."""
    try:
        metrics_collector = get_metrics_collector()
        alert_manager = get_alert_manager()
        
        # Get metrics summary
        metrics_summary = metrics_collector.get_metrics_summary(hours=1)
        
        # Get active alerts by severity
        active_alerts = alert_manager.get_active_alerts()
        alerts_by_severity = {}
        for alert in active_alerts:
            severity = alert['severity']
            if severity not in alerts_by_severity:
                alerts_by_severity[severity] = 0
            alerts_by_severity[severity] += 1
        
        # Get recent system metrics for trend analysis
        recent_system_metrics = metrics_collector.get_recent_system_metrics(minutes=60)
        
        # Calculate trends (last hour vs previous hour)
        current_hour_metrics = [m for m in recent_system_metrics if 
                              datetime.fromisoformat(m['timestamp'].replace('Z', '')) >= 
                              datetime.utcnow() - timedelta(hours=1)]
        
        trends = {}
        if current_hour_metrics:
            avg_cpu = sum(m['cpu_percent'] for m in current_hour_metrics) / len(current_hour_metrics)
            avg_memory = sum(m['memory_percent'] for m in current_hour_metrics) / len(current_hour_metrics)
            
            trends = {
                'cpu_trend': 'stable',  # Would calculate actual trend
                'memory_trend': 'stable',
                'avg_cpu_last_hour': round(avg_cpu, 2),
                'avg_memory_last_hour': round(avg_memory, 2)
            }
        
        overview = {
            'timestamp': datetime.utcnow().isoformat(),
            'system_health': {
                'status': 'healthy' if len(active_alerts) == 0 else 'warning',
                'cpu_usage': metrics_summary['system']['avg_cpu_percent'],
                'memory_usage': metrics_summary['system']['avg_memory_percent'],
                'disk_usage': metrics_summary['system']['avg_disk_percent']
            },
            'api_performance': {
                'total_requests': metrics_summary['api']['total_requests'],
                'avg_response_time': metrics_summary['api']['avg_response_time_ms'],
                'error_rate': metrics_summary['api']['error_rate_percent'],
                'unique_endpoints': metrics_summary['api']['unique_endpoints']
            },
            'alerts': {
                'total_active': len(active_alerts),
                'by_severity': alerts_by_severity,
                'critical_count': alerts_by_severity.get('critical', 0),
                'high_count': alerts_by_severity.get('high', 0)
            },
            'trends': trends
        }
        
        return jsonify({
            'success': True,
            'data': overview
        })
        
    except Exception as e:
        logger.error("Failed to get dashboard overview", error=str(e))
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@monitoring_bp.route('/metrics/cleanup', methods=['POST'])
def cleanup_old_metrics():
    """Clean up old metrics data."""
    try:
        days = request.json.get('days', 7) if request.is_json else 7
        
        if days < 1 or days > 30:
            return jsonify({
                'success': False,
                'error': 'Days must be between 1 and 30'
            }), 400
        
        metrics_collector = get_metrics_collector()
        deleted_counts = metrics_collector.cleanup_old_metrics(days=days)
        
        logger.info("Metrics cleanup completed via API", deleted_counts=deleted_counts)
        
        return jsonify({
            'success': True,
            'message': f'Cleaned up metrics older than {days} days',
            'deleted_counts': deleted_counts
        })
        
    except Exception as e:
        logger.error("Failed to cleanup metrics", error=str(e))
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@monitoring_bp.route('/status', methods=['GET'])
def get_monitoring_status():
    """Get monitoring system status."""
    try:
        metrics_collector = get_metrics_collector()
        alert_manager = get_alert_manager()
        
        status = {
            'monitoring_active': True,
            'metrics_collection': {
                'running': metrics_collector.running,
                'collection_interval': metrics_collector.collection_interval,
                'recent_metrics_count': len(metrics_collector.system_metrics)
            },
            'alerting': {
                'running': alert_manager.running,
                'check_interval': alert_manager.check_interval,
                'active_rules': len(alert_manager.rules),
                'notification_handlers': len(alert_manager.notification_handlers)
            },
            'database': {
                'metrics_db': metrics_collector.db_path,
                'alerts_db': alert_manager.db_path
            }
        }
        
        return jsonify({
            'success': True,
            'data': status
        })
        
    except Exception as e:
        logger.error("Failed to get monitoring status", error=str(e))
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500