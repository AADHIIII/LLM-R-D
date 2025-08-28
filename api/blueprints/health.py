"""
Health check and status endpoints.
"""
from flask import Blueprint, jsonify
from datetime import datetime
import psutil
import os
from typing import Dict, Any

health_bp = Blueprint('health', __name__)


@health_bp.route('/health', methods=['GET'])
def health_check() -> Dict[str, Any]:
    """
    Basic health check endpoint.
    
    Returns:
        JSON response with health status
    """
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '1.0.0'
    })


@health_bp.route('/status', methods=['GET'])
def detailed_status() -> Dict[str, Any]:
    """
    Detailed system status endpoint.
    
    Returns:
        JSON response with detailed system information
    """
    try:
        # Get system metrics
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        cpu_percent = psutil.cpu_percent(interval=1)
        
        status_data = {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'version': '1.0.0',
            'system': {
                'memory': {
                    'total': memory.total,
                    'available': memory.available,
                    'percent': memory.percent,
                    'used': memory.used
                },
                'disk': {
                    'total': disk.total,
                    'free': disk.free,
                    'used': disk.used,
                    'percent': (disk.used / disk.total) * 100
                },
                'cpu': {
                    'percent': cpu_percent,
                    'count': psutil.cpu_count()
                }
            },
            'environment': {
                'python_version': os.sys.version,
                'platform': os.name
            }
        }
        
        # Check if system is under stress
        if memory.percent > 90 or cpu_percent > 90:
            status_data['status'] = 'degraded'
            status_data['warnings'] = []
            
            if memory.percent > 90:
                status_data['warnings'].append('High memory usage detected')
            
            if cpu_percent > 90:
                status_data['warnings'].append('High CPU usage detected')
        
        return jsonify(status_data)
        
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'timestamp': datetime.utcnow().isoformat(),
            'error': str(e)
        }), 500


@health_bp.route('/ready', methods=['GET'])
def readiness_check() -> Dict[str, Any]:
    """
    Readiness check for Kubernetes/container orchestration.
    
    Returns:
        JSON response indicating if service is ready to accept traffic
    """
    try:
        # Check if essential services are available
        checks = {
            'memory_available': psutil.virtual_memory().percent < 95,
            'disk_available': psutil.disk_usage('/').free > 1024 * 1024 * 100,  # 100MB free
        }
        
        all_ready = all(checks.values())
        
        return jsonify({
            'ready': all_ready,
            'timestamp': datetime.utcnow().isoformat(),
            'checks': checks
        }), 200 if all_ready else 503
        
    except Exception as e:
        return jsonify({
            'ready': False,
            'timestamp': datetime.utcnow().isoformat(),
            'error': str(e)
        }), 503