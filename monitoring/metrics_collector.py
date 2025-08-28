"""
System metrics collection for monitoring dashboard.
"""

import psutil
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from collections import deque
import json
import sqlite3
import os

from utils.logging import get_structured_logger


@dataclass
class SystemMetrics:
    """System performance metrics."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    disk_free_gb: float
    network_bytes_sent: int
    network_bytes_recv: int
    active_connections: int


@dataclass
class APIMetrics:
    """API performance metrics."""
    timestamp: datetime
    endpoint: str
    method: str
    status_code: int
    response_time_ms: float
    request_size_bytes: int
    response_size_bytes: int
    error_count: int = 0


@dataclass
class ApplicationMetrics:
    """Application-specific metrics."""
    timestamp: datetime
    active_fine_tuning_jobs: int
    completed_evaluations: int
    total_api_calls: int
    cache_hit_rate: float
    database_connections: int
    queue_size: int


class MetricsCollector:
    """Collects and stores system and application metrics."""
    
    def __init__(self, db_path: str = "monitoring/metrics.db", collection_interval: int = 30):
        self.db_path = db_path
        self.collection_interval = collection_interval
        self.logger = get_structured_logger('metrics_collector')
        self.running = False
        self.collection_thread = None
        
        # In-memory storage for recent metrics (last 1000 points)
        self.system_metrics = deque(maxlen=1000)
        self.api_metrics = deque(maxlen=5000)
        self.app_metrics = deque(maxlen=1000)
        
        # Network baseline for calculating deltas
        self.last_network_stats = None
        
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize SQLite database for metrics storage."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            # System metrics table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
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
                )
            """)
            
            # API metrics table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS api_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    endpoint TEXT,
                    method TEXT,
                    status_code INTEGER,
                    response_time_ms REAL,
                    request_size_bytes INTEGER,
                    response_size_bytes INTEGER,
                    error_count INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Application metrics table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS app_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    active_fine_tuning_jobs INTEGER,
                    completed_evaluations INTEGER,
                    total_api_calls INTEGER,
                    cache_hit_rate REAL,
                    database_connections INTEGER,
                    queue_size INTEGER,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for better query performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_system_timestamp ON system_metrics(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_api_timestamp ON api_metrics(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_app_timestamp ON app_metrics(timestamp)")
    
    def start_collection(self) -> None:
        """Start metrics collection in background thread."""
        if self.running:
            self.logger.warning("Metrics collection already running")
            return
        
        self.running = True
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()
        
        self.logger.info(
            "Started metrics collection",
            interval_seconds=self.collection_interval
        )
    
    def stop_collection(self) -> None:
        """Stop metrics collection."""
        self.running = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
        
        self.logger.info("Stopped metrics collection")
    
    def _collection_loop(self) -> None:
        """Main collection loop running in background thread."""
        while self.running:
            try:
                # Collect system metrics
                system_metrics = self._collect_system_metrics()
                self.system_metrics.append(system_metrics)
                self._store_system_metrics(system_metrics)
                
                # Collect application metrics
                app_metrics = self._collect_app_metrics()
                self.app_metrics.append(app_metrics)
                self._store_app_metrics(app_metrics)
                
                time.sleep(self.collection_interval)
                
            except Exception as e:
                self.logger.error(
                    "Error in metrics collection loop",
                    error=str(e)
                )
                time.sleep(self.collection_interval)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system performance metrics."""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        
        # Disk usage (root partition)
        disk = psutil.disk_usage('/')
        
        # Network statistics
        network = psutil.net_io_counters()
        
        # Network connections
        connections = len(psutil.net_connections())
        
        return SystemMetrics(
            timestamp=datetime.utcnow(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_mb=memory.used / (1024 * 1024),
            memory_available_mb=memory.available / (1024 * 1024),
            disk_usage_percent=disk.percent,
            disk_free_gb=disk.free / (1024 * 1024 * 1024),
            network_bytes_sent=network.bytes_sent,
            network_bytes_recv=network.bytes_recv,
            active_connections=connections
        )
    
    def _collect_app_metrics(self) -> ApplicationMetrics:
        """Collect application-specific metrics."""
        # These would be populated by the application
        # For now, using placeholder values
        return ApplicationMetrics(
            timestamp=datetime.utcnow(),
            active_fine_tuning_jobs=0,  # Would be set by fine-tuning service
            completed_evaluations=0,    # Would be set by evaluation engine
            total_api_calls=0,          # Would be set by API gateway
            cache_hit_rate=0.0,         # Would be set by caching layer
            database_connections=0,     # Would be set by database layer
            queue_size=0                # Would be set by task queue
        )
    
    def record_api_metrics(self, metrics: APIMetrics) -> None:
        """Record API request metrics."""
        self.api_metrics.append(metrics)
        self._store_api_metrics(metrics)
    
    def _store_system_metrics(self, metrics: SystemMetrics) -> None:
        """Store system metrics in database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO system_metrics (
                    timestamp, cpu_percent, memory_percent, memory_used_mb,
                    memory_available_mb, disk_usage_percent, disk_free_gb,
                    network_bytes_sent, network_bytes_recv, active_connections
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics.timestamp.isoformat(),
                metrics.cpu_percent,
                metrics.memory_percent,
                metrics.memory_used_mb,
                metrics.memory_available_mb,
                metrics.disk_usage_percent,
                metrics.disk_free_gb,
                metrics.network_bytes_sent,
                metrics.network_bytes_recv,
                metrics.active_connections
            ))
    
    def _store_api_metrics(self, metrics: APIMetrics) -> None:
        """Store API metrics in database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO api_metrics (
                    timestamp, endpoint, method, status_code, response_time_ms,
                    request_size_bytes, response_size_bytes, error_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics.timestamp.isoformat(),
                metrics.endpoint,
                metrics.method,
                metrics.status_code,
                metrics.response_time_ms,
                metrics.request_size_bytes,
                metrics.response_size_bytes,
                metrics.error_count
            ))
    
    def _store_app_metrics(self, metrics: ApplicationMetrics) -> None:
        """Store application metrics in database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO app_metrics (
                    timestamp, active_fine_tuning_jobs, completed_evaluations,
                    total_api_calls, cache_hit_rate, database_connections, queue_size
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics.timestamp.isoformat(),
                metrics.active_fine_tuning_jobs,
                metrics.completed_evaluations,
                metrics.total_api_calls,
                metrics.cache_hit_rate,
                metrics.database_connections,
                metrics.queue_size
            ))
    
    def get_recent_system_metrics(self, minutes: int = 60) -> List[Dict[str, Any]]:
        """Get recent system metrics."""
        cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)
        
        return [
            asdict(metric) for metric in self.system_metrics
            if metric.timestamp >= cutoff_time
        ]
    
    def get_recent_api_metrics(self, minutes: int = 60) -> List[Dict[str, Any]]:
        """Get recent API metrics."""
        cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)
        
        return [
            asdict(metric) for metric in self.api_metrics
            if metric.timestamp >= cutoff_time
        ]
    
    def get_metrics_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get metrics summary for the specified time period."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        with sqlite3.connect(self.db_path) as conn:
            # System metrics summary
            system_summary = conn.execute("""
                SELECT 
                    AVG(cpu_percent) as avg_cpu,
                    MAX(cpu_percent) as max_cpu,
                    AVG(memory_percent) as avg_memory,
                    MAX(memory_percent) as max_memory,
                    AVG(disk_usage_percent) as avg_disk,
                    COUNT(*) as data_points
                FROM system_metrics 
                WHERE timestamp >= ?
            """, (cutoff_time.isoformat(),)).fetchone()
            
            # API metrics summary
            api_summary = conn.execute("""
                SELECT 
                    COUNT(*) as total_requests,
                    AVG(response_time_ms) as avg_response_time,
                    MAX(response_time_ms) as max_response_time,
                    SUM(CASE WHEN status_code >= 400 THEN 1 ELSE 0 END) as error_count,
                    COUNT(DISTINCT endpoint) as unique_endpoints
                FROM api_metrics 
                WHERE timestamp >= ?
            """, (cutoff_time.isoformat(),)).fetchone()
            
            # Error rate calculation
            error_rate = 0.0
            if api_summary[0] > 0:  # total_requests
                error_rate = (api_summary[3] / api_summary[0]) * 100  # error_count / total_requests
            
            return {
                'time_period_hours': hours,
                'system': {
                    'avg_cpu_percent': round(system_summary[0] or 0, 2),
                    'max_cpu_percent': round(system_summary[1] or 0, 2),
                    'avg_memory_percent': round(system_summary[2] or 0, 2),
                    'max_memory_percent': round(system_summary[3] or 0, 2),
                    'avg_disk_percent': round(system_summary[4] or 0, 2),
                    'data_points': system_summary[5] or 0
                },
                'api': {
                    'total_requests': api_summary[0] or 0,
                    'avg_response_time_ms': round(api_summary[1] or 0, 2),
                    'max_response_time_ms': round(api_summary[2] or 0, 2),
                    'error_count': api_summary[3] or 0,
                    'error_rate_percent': round(error_rate, 2),
                    'unique_endpoints': api_summary[4] or 0
                }
            }
    
    def cleanup_old_metrics(self, days: int = 7) -> Dict[str, int]:
        """Remove metrics older than specified days."""
        cutoff_time = datetime.utcnow() - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            # Clean up system metrics
            system_deleted = conn.execute(
                "DELETE FROM system_metrics WHERE timestamp < ?",
                (cutoff_time.isoformat(),)
            ).rowcount
            
            # Clean up API metrics
            api_deleted = conn.execute(
                "DELETE FROM api_metrics WHERE timestamp < ?",
                (cutoff_time.isoformat(),)
            ).rowcount
            
            # Clean up app metrics
            app_deleted = conn.execute(
                "DELETE FROM app_metrics WHERE timestamp < ?",
                (cutoff_time.isoformat(),)
            ).rowcount
        
        deleted_counts = {
            'system_metrics': system_deleted,
            'api_metrics': api_deleted,
            'app_metrics': app_deleted
        }
        
        self.logger.info(
            "Cleaned up old metrics",
            cutoff_days=days,
            deleted_counts=deleted_counts
        )
        
        return deleted_counts


# Global metrics collector instance
_metrics_collector = None


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector instance."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def start_metrics_collection() -> None:
    """Start global metrics collection."""
    collector = get_metrics_collector()
    collector.start_collection()


def stop_metrics_collection() -> None:
    """Stop global metrics collection."""
    global _metrics_collector
    if _metrics_collector:
        _metrics_collector.stop_collection()