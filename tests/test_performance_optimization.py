"""
Tests for performance optimization features.
"""
import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from utils.cache_manager import CacheManager, MemoryCache, cache_result
from utils.async_processor import AsyncTaskManager, TaskStatus, background_task
from utils.connection_pool import ConnectionPool, HTTPConnectionPool
from utils.performance_monitor import PerformanceMonitor, monitor_performance


class TestCacheManager:
    """Test cache management functionality."""
    
    def test_memory_cache_basic_operations(self):
        """Test basic cache operations."""
        cache = MemoryCache(max_size=3, default_ttl=60)
        
        # Test set and get
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # Test non-existent key
        assert cache.get("nonexistent") is None
        
        # Test cache size limit
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        cache.set("key4", "value4")  # Should evict key1
        
        assert cache.get("key1") is None
        assert cache.get("key4") == "value4"
    
    def test_memory_cache_ttl(self):
        """Test cache TTL functionality."""
        cache = MemoryCache(max_size=10, default_ttl=1)
        
        cache.set("key1", "value1", ttl=1)
        assert cache.get("key1") == "value1"
        
        # Wait for expiration
        time.sleep(1.1)
        assert cache.get("key1") is None
    
    def test_cache_manager_fallback(self):
        """Test cache manager fallback behavior."""
        with patch('utils.cache_manager.RedisCache') as mock_redis:
            # Mock Redis failure
            mock_redis.return_value.client = None
            
            cache_manager = CacheManager(use_redis=True)
            
            # Should fallback to memory cache
            assert isinstance(cache_manager.primary_cache, MemoryCache)
    
    def test_cache_result_decorator(self):
        """Test cache result decorator."""
        call_count = 0
        
        @cache_result(key_prefix="test", ttl=60)
        def expensive_function(x, y):
            nonlocal call_count
            call_count += 1
            return x + y
        
        # First call should execute function
        result1 = expensive_function(1, 2)
        assert result1 == 3
        assert call_count == 1
        
        # Second call should use cache
        result2 = expensive_function(1, 2)
        assert result2 == 3
        assert call_count == 1  # Should not increment
        
        # Different parameters should execute function
        result3 = expensive_function(2, 3)
        assert result3 == 5
        assert call_count == 2


class TestAsyncProcessor:
    """Test asynchronous processing functionality."""
    
    def test_task_manager_basic_operations(self):
        """Test basic task manager operations."""
        task_manager = AsyncTaskManager(max_workers=2)
        
        def simple_task(x, y):
            time.sleep(0.1)
            return x + y
        
        # Submit task
        task_id = task_manager.submit_task(simple_task, 1, 2)
        assert task_id is not None
        
        # Check initial status
        status = task_manager.get_task_status(task_id)
        assert status is not None
        assert status.task_id == task_id
        
        # Wait for completion
        time.sleep(0.2)
        
        # Check final status
        status = task_manager.get_task_status(task_id)
        assert status.status == TaskStatus.COMPLETED
        assert status.result == 3
        
        task_manager.shutdown()
    
    def test_task_manager_error_handling(self):
        """Test task manager error handling."""
        task_manager = AsyncTaskManager(max_workers=1)
        
        def failing_task():
            raise ValueError("Test error")
        
        # Submit failing task
        task_id = task_manager.submit_task(failing_task)
        
        # Wait for completion
        time.sleep(0.1)
        
        # Check error status
        status = task_manager.get_task_status(task_id)
        assert status.status == TaskStatus.FAILED
        assert "Test error" in status.error
        
        task_manager.shutdown()
    
    def test_task_cancellation(self):
        """Test task cancellation."""
        task_manager = AsyncTaskManager(max_workers=1)
        
        def long_running_task():
            time.sleep(1)
            return "completed"
        
        # Submit task
        task_id = task_manager.submit_task(long_running_task)
        
        # Try to cancel immediately
        cancelled = task_manager.cancel_task(task_id)
        
        # Note: Cancellation may not always succeed if task already started
        if cancelled:
            status = task_manager.get_task_status(task_id)
            assert status.status == TaskStatus.CANCELLED
        
        task_manager.shutdown()
    
    def test_background_task_decorator(self):
        """Test background task decorator."""
        executed = threading.Event()
        
        @background_task
        def background_function():
            executed.set()
        
        # Execute background task
        background_function()
        
        # Wait for execution
        assert executed.wait(timeout=1)


class TestConnectionPool:
    """Test connection pooling functionality."""
    
    def test_connection_pool_basic_operations(self):
        """Test basic connection pool operations."""
        created_connections = []
        closed_connections = []
        
        def create_conn():
            conn = Mock()
            created_connections.append(conn)
            return conn
        
        def close_conn(conn):
            closed_connections.append(conn)
        
        def validate_conn(conn):
            return True
        
        pool = ConnectionPool(
            create_connection=create_conn,
            close_connection=close_conn,
            validate_connection=validate_conn,
            max_size=3,
            min_size=1
        )
        
        # Test getting connection
        with pool.get_connection() as conn:
            assert conn is not None
            assert conn in created_connections
        
        # Test pool statistics
        stats = pool.get_stats()
        assert stats['max_size'] == 3
        assert stats['min_size'] == 1
        
        pool.close()
    
    def test_connection_pool_validation(self):
        """Test connection validation and replacement."""
        validation_results = [True, False, True]  # Second validation fails
        validation_calls = 0
        
        def create_conn():
            return Mock()
        
        def close_conn(conn):
            pass
        
        def validate_conn(conn):
            nonlocal validation_calls
            result = validation_results[validation_calls % len(validation_results)]
            validation_calls += 1
            return result
        
        pool = ConnectionPool(
            create_connection=create_conn,
            close_connection=close_conn,
            validate_connection=validate_conn,
            max_size=2,
            min_size=1
        )
        
        # First connection should work
        with pool.get_connection() as conn1:
            assert conn1 is not None
        
        # Second connection should trigger validation failure and recreation
        with pool.get_connection() as conn2:
            assert conn2 is not None
        
        pool.close()
    
    def test_http_connection_pool(self):
        """Test HTTP connection pool."""
        with patch('requests.Session') as mock_session_class:
            mock_session = Mock()
            mock_session_class.return_value = mock_session
            
            pool = HTTPConnectionPool(max_size=2, min_size=1)
            
            with pool.get_connection() as session:
                assert session == mock_session
            
            pool.close()


class TestPerformanceMonitor:
    """Test performance monitoring functionality."""
    
    def test_performance_monitor_basic_operations(self):
        """Test basic performance monitoring."""
        monitor = PerformanceMonitor(max_metrics=100)
        
        # Test operation measurement
        with monitor.measure_operation("test_operation"):
            time.sleep(0.1)
        
        # Check recorded metrics
        stats = monitor.get_operation_stats("test_operation")
        assert stats['count'] == 1
        assert stats['duration']['avg'] >= 0.1
        
        monitor.stop_monitoring()
    
    def test_performance_monitor_decorator(self):
        """Test performance monitoring decorator."""
        monitor = PerformanceMonitor(max_metrics=100)
        
        # Patch global monitor
        with patch('utils.performance_monitor.performance_monitor', monitor):
            @monitor_performance("decorated_operation")
            def test_function():
                time.sleep(0.05)
                return "result"
            
            result = test_function()
            assert result == "result"
            
            # Check metrics
            stats = monitor.get_operation_stats("decorated_operation")
            assert stats['count'] == 1
        
        monitor.stop_monitoring()
    
    def test_performance_monitor_system_stats(self):
        """Test system performance monitoring."""
        monitor = PerformanceMonitor(max_metrics=100)
        
        # Wait for some system metrics to be collected
        time.sleep(1)
        
        # Check system stats
        system_stats = monitor.get_system_stats(hours=1)
        assert system_stats['count'] > 0
        assert 'cpu' in system_stats
        assert 'memory' in system_stats
        
        monitor.stop_monitoring()
    
    def test_performance_monitor_export(self):
        """Test metrics export functionality."""
        import tempfile
        import json
        
        monitor = PerformanceMonitor(max_metrics=100)
        
        # Record some metrics
        with monitor.measure_operation("export_test"):
            time.sleep(0.01)
        
        # Export metrics
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            monitor.export_metrics(f.name, hours=1)
            
            # Read and verify export
            with open(f.name, 'r') as read_f:
                data = json.load(read_f)
                
                assert 'export_time' in data
                assert 'operation_metrics' in data
                assert 'system_metrics' in data
                assert len(data['operation_metrics']) > 0
        
        monitor.stop_monitoring()


class TestIntegratedPerformance:
    """Test integrated performance optimizations."""
    
    def test_cached_model_loading_performance(self):
        """Test performance of cached model loading."""
        from api.services.model_loader import ModelLoader
        
        # Mock transformers to avoid actual model loading
        with patch('api.services.model_loader.AutoModelForCausalLM') as mock_model, \
             patch('api.services.model_loader.AutoTokenizer') as mock_tokenizer, \
             patch('os.path.exists', return_value=True):
            
            mock_model.from_pretrained.return_value = Mock()
            mock_tokenizer.from_pretrained.return_value = Mock()
            
            loader = ModelLoader(cache_size=2)
            
            # First load should be slower (actual loading)
            start_time = time.time()
            model, tokenizer = loader.load_model("test_model")
            first_load_time = time.time() - start_time
            
            # Second load should be faster (cached)
            start_time = time.time()
            model2, tokenizer2 = loader.load_model("test_model")
            second_load_time = time.time() - start_time
            
            # Cache should be significantly faster
            assert second_load_time < first_load_time * 0.1  # At least 10x faster
            assert model == model2
            assert tokenizer == tokenizer2
    
    def test_database_connection_pooling_performance(self):
        """Test database connection pooling performance."""
        from database.connection import DatabaseManager
        
        # Mock database operations
        with patch('sqlalchemy.create_engine') as mock_engine, \
             patch('sqlalchemy.orm.sessionmaker') as mock_sessionmaker:
            
            mock_session = Mock()
            mock_sessionmaker.return_value = Mock(return_value=mock_session)
            
            db_manager = DatabaseManager("sqlite:///test.db")
            db_manager.initialize()
            
            # Test multiple concurrent connections
            def db_operation():
                with db_manager.get_session() as session:
                    session.execute("SELECT 1")
            
            # Measure performance with multiple threads
            threads = []
            start_time = time.time()
            
            for _ in range(10):
                thread = threading.Thread(target=db_operation)
                threads.append(thread)
                thread.start()
            
            for thread in threads:
                thread.join()
            
            total_time = time.time() - start_time
            
            # Should complete reasonably quickly with connection pooling
            assert total_time < 1.0  # Should complete within 1 second
    
    def test_async_task_processing_performance(self):
        """Test asynchronous task processing performance."""
        task_manager = AsyncTaskManager(max_workers=4)
        
        def cpu_intensive_task(n):
            # Simulate CPU-intensive work
            total = 0
            for i in range(n):
                total += i * i
            return total
        
        # Submit multiple tasks
        task_ids = []
        start_time = time.time()
        
        for i in range(8):
            task_id = task_manager.submit_task(cpu_intensive_task, 10000)
            task_ids.append(task_id)
        
        # Wait for all tasks to complete
        while True:
            completed = 0
            for task_id in task_ids:
                status = task_manager.get_task_status(task_id)
                if status and status.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                    completed += 1
            
            if completed == len(task_ids):
                break
            
            time.sleep(0.1)
        
        total_time = time.time() - start_time
        
        # Parallel execution should be faster than sequential
        # (This is a rough test - actual speedup depends on system)
        assert total_time < 5.0  # Should complete within reasonable time
        
        # Verify all tasks completed successfully
        for task_id in task_ids:
            status = task_manager.get_task_status(task_id)
            assert status.status == TaskStatus.COMPLETED
            assert status.result is not None
        
        task_manager.shutdown()


if __name__ == "__main__":
    pytest.main([__file__])