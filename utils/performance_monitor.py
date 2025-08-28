"""
Performance monitoring and benchmarking utilities.
"""
import time
import psutil
import logging
import threading
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from contextlib import contextmanager
from functools import wraps
import json
import os

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""
    operation: str
    duration: float
    cpu_usage: float
    memory_usage: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemMetrics:
    """System-wide performance metrics."""
    cpu_percent: float
    memory_percent: float
    memory_available: int
    disk_usage: float
    network_io: Dict[str, int]
    timestamp: datetime = field(default_factory=datetime.utcnow)


class PerformanceMonitor:
    """Performance monitoring and profiling system."""
    
    def __init__(self, max_metrics: int = 10000):
        self.max_metrics = max_metrics
        self.metrics: List[PerformanceMetrics] = []
        self.system_metrics: List[SystemMetrics] = []
        self.lock = threading.Lock()
        
        # Start system monitoring thread
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._system_monitor, daemon=True)
        self.monitor_thread.start()
    
    def _system_monitor(self) -> None:
        """Background system monitoring."""
        while self.monitoring:
            try:
                # Collect system metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                network = psutil.net_io_counters()
                
                system_metric = SystemMetrics(
                    cpu_percent=cpu_percent,
                    memory_percent=memory.percent,
                    memory_available=memory.available,
                    disk_usage=disk.percent,
                    network_io={
                        'bytes_sent': network.bytes_sent,
                        'bytes_recv': network.bytes_recv
                    }
                )
                
                with self.lock:
                    self.system_metrics.append(system_metric)
                    
                    # Keep only recent metrics
                    if len(self.system_metrics) > self.max_metrics:
                        self.system_metrics = self.system_metrics[-self.max_metrics:]
                
                time.sleep(5)  # Collect every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in system monitoring: {e}")
                time.sleep(10)
    
    @contextmanager
    def measure_operation(self, operation: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Context manager to measure operation performance.
        
        Args:
            operation: Operation name
            metadata: Additional metadata
        """
        start_time = time.time()
        start_cpu = psutil.cpu_percent()
        start_memory = psutil.virtual_memory().percent
        
        try:
            yield
        finally:
            end_time = time.time()
            end_cpu = psutil.cpu_percent()
            end_memory = psutil.virtual_memory().percent
            
            duration = end_time - start_time
            avg_cpu = (start_cpu + end_cpu) / 2
            avg_memory = (start_memory + end_memory) / 2
            
            metric = PerformanceMetrics(
                operation=operation,
                duration=duration,
                cpu_usage=avg_cpu,
                memory_usage=avg_memory,
                metadata=metadata or {}
            )
            
            self.record_metric(metric)
    
    def record_metric(self, metric: PerformanceMetrics) -> None:
        """Record a performance metric."""
        with self.lock:
            self.metrics.append(metric)
            
            # Keep only recent metrics
            if len(self.metrics) > self.max_metrics:
                self.metrics = self.metrics[-self.max_metrics:]
    
    def get_operation_stats(self, operation: str, hours: int = 24) -> Dict[str, Any]:
        """
        Get statistics for a specific operation.
        
        Args:
            operation: Operation name
            hours: Hours of history to analyze
            
        Returns:
            Statistics dictionary
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        with self.lock:
            relevant_metrics = [
                m for m in self.metrics 
                if m.operation == operation and m.timestamp > cutoff_time
            ]
        
        if not relevant_metrics:
            return {'operation': operation, 'count': 0}
        
        durations = [m.duration for m in relevant_metrics]
        cpu_usages = [m.cpu_usage for m in relevant_metrics]
        memory_usages = [m.memory_usage for m in relevant_metrics]
        
        return {
            'operation': operation,
            'count': len(relevant_metrics),
            'duration': {
                'min': min(durations),
                'max': max(durations),
                'avg': sum(durations) / len(durations),
                'total': sum(durations)
            },
            'cpu_usage': {
                'min': min(cpu_usages),
                'max': max(cpu_usages),
                'avg': sum(cpu_usages) / len(cpu_usages)
            },
            'memory_usage': {
                'min': min(memory_usages),
                'max': max(memory_usages),
                'avg': sum(memory_usages) / len(memory_usages)
            },
            'first_seen': min(m.timestamp for m in relevant_metrics),
            'last_seen': max(m.timestamp for m in relevant_metrics)
        }
    
    def get_system_stats(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get system performance statistics.
        
        Args:
            hours: Hours of history to analyze
            
        Returns:
            System statistics
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        with self.lock:
            relevant_metrics = [
                m for m in self.system_metrics 
                if m.timestamp > cutoff_time
            ]
        
        if not relevant_metrics:
            return {'count': 0}
        
        cpu_values = [m.cpu_percent for m in relevant_metrics]
        memory_values = [m.memory_percent for m in relevant_metrics]
        
        return {
            'count': len(relevant_metrics),
            'cpu': {
                'min': min(cpu_values),
                'max': max(cpu_values),
                'avg': sum(cpu_values) / len(cpu_values)
            },
            'memory': {
                'min': min(memory_values),
                'max': max(memory_values),
                'avg': sum(memory_values) / len(memory_values)
            },
            'period': {
                'start': min(m.timestamp for m in relevant_metrics),
                'end': max(m.timestamp for m in relevant_metrics)
            }
        }
    
    def get_all_operations(self) -> List[str]:
        """Get list of all monitored operations."""
        with self.lock:
            return list(set(m.operation for m in self.metrics))
    
    def export_metrics(self, filepath: str, hours: int = 24) -> None:
        """
        Export metrics to JSON file.
        
        Args:
            filepath: Output file path
            hours: Hours of history to export
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        with self.lock:
            relevant_metrics = [
                {
                    'operation': m.operation,
                    'duration': m.duration,
                    'cpu_usage': m.cpu_usage,
                    'memory_usage': m.memory_usage,
                    'timestamp': m.timestamp.isoformat(),
                    'metadata': m.metadata
                }
                for m in self.metrics 
                if m.timestamp > cutoff_time
            ]
            
            relevant_system_metrics = [
                {
                    'cpu_percent': m.cpu_percent,
                    'memory_percent': m.memory_percent,
                    'memory_available': m.memory_available,
                    'disk_usage': m.disk_usage,
                    'network_io': m.network_io,
                    'timestamp': m.timestamp.isoformat()
                }
                for m in self.system_metrics 
                if m.timestamp > cutoff_time
            ]
        
        export_data = {
            'export_time': datetime.utcnow().isoformat(),
            'period_hours': hours,
            'operation_metrics': relevant_metrics,
            'system_metrics': relevant_system_metrics
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported {len(relevant_metrics)} metrics to {filepath}")
    
    def stop_monitoring(self) -> None:
        """Stop system monitoring."""
        self.monitoring = False
        if self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)


class Benchmark:
    """Benchmarking utilities for performance testing."""
    
    def __init__(self, monitor: PerformanceMonitor):
        self.monitor = monitor
    
    def benchmark_function(self, func: Callable, *args, iterations: int = 10, **kwargs) -> Dict[str, Any]:
        """
        Benchmark a function with multiple iterations.
        
        Args:
            func: Function to benchmark
            *args: Function arguments
            iterations: Number of iterations
            **kwargs: Function keyword arguments
            
        Returns:
            Benchmark results
        """
        results = []
        
        for i in range(iterations):
            with self.monitor.measure_operation(f"benchmark_{func.__name__}_{i}"):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    success = True
                    error = None
                except Exception as e:
                    result = None
                    success = False
                    error = str(e)
                
                end_time = time.time()
                
                results.append({
                    'iteration': i,
                    'duration': end_time - start_time,
                    'success': success,
                    'error': error,
                    'result_size': len(str(result)) if result else 0
                })
        
        # Calculate statistics
        successful_results = [r for r in results if r['success']]
        durations = [r['duration'] for r in successful_results]
        
        if durations:
            return {
                'function': func.__name__,
                'iterations': iterations,
                'successful': len(successful_results),
                'failed': len(results) - len(successful_results),
                'duration': {
                    'min': min(durations),
                    'max': max(durations),
                    'avg': sum(durations) / len(durations),
                    'total': sum(durations)
                },
                'throughput': len(successful_results) / sum(durations) if durations else 0,
                'results': results
            }
        else:
            return {
                'function': func.__name__,
                'iterations': iterations,
                'successful': 0,
                'failed': len(results),
                'error': 'All iterations failed',
                'results': results
            }


# Global performance monitor
performance_monitor = PerformanceMonitor()


def monitor_performance(operation: str = None, metadata: Dict[str, Any] = None):
    """
    Decorator to monitor function performance.
    
    Args:
        operation: Operation name (defaults to function name)
        metadata: Additional metadata
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation or func.__name__
            with performance_monitor.measure_operation(op_name, metadata):
                return func(*args, **kwargs)
        return wrapper
    return decorator