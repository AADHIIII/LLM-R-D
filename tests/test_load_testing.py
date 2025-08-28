"""
Load testing for concurrent users and requests.
"""
import pytest
import time
import threading
import concurrent.futures
from unittest.mock import Mock, patch
import statistics
from typing import List, Dict, Any
import json

from api.app import create_app
from utils.performance_monitor import PerformanceMonitor
from utils.async_processor import AsyncTaskManager


class LoadTestResults:
    """Container for load test results."""
    
    def __init__(self):
        self.response_times: List[float] = []
        self.success_count: int = 0
        self.error_count: int = 0
        self.errors: List[str] = []
        self.start_time: float = 0
        self.end_time: float = 0
    
    def add_result(self, response_time: float, success: bool, error: str = None):
        """Add a test result."""
        self.response_times.append(response_time)
        if success:
            self.success_count += 1
        else:
            self.error_count += 1
            if error:
                self.errors.append(error)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get test statistics."""
        if not self.response_times:
            return {"error": "No results recorded"}
        
        total_time = self.end_time - self.start_time
        total_requests = len(self.response_times)
        
        return {
            "total_requests": total_requests,
            "successful_requests": self.success_count,
            "failed_requests": self.error_count,
            "success_rate": self.success_count / total_requests * 100,
            "total_duration": total_time,
            "requests_per_second": total_requests / total_time if total_time > 0 else 0,
            "response_times": {
                "min": min(self.response_times),
                "max": max(self.response_times),
                "mean": statistics.mean(self.response_times),
                "median": statistics.median(self.response_times),
                "p95": statistics.quantiles(self.response_times, n=20)[18] if len(self.response_times) >= 20 else max(self.response_times),
                "p99": statistics.quantiles(self.response_times, n=100)[98] if len(self.response_times) >= 100 else max(self.response_times)
            },
            "errors": list(set(self.errors))  # Unique errors
        }


class TestLoadTesting:
    """Load testing suite for the LLM optimization platform."""
    
    @pytest.fixture
    def app(self):
        """Create test Flask app."""
        app = create_app('testing')
        app.config['TESTING'] = True
        return app
    
    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return app.test_client()
    
    def test_api_concurrent_requests(self, client):
        """Test API under concurrent request load."""
        
        # Mock external services to avoid actual API calls
        with patch('api.services.openai_client.OpenAI') as mock_openai:
            mock_openai.return_value.chat.completions.create.return_value = Mock(
                choices=[Mock(message=Mock(content="Test response"))]
            )
            
            results = LoadTestResults()
            results.start_time = time.time()
            
            def make_request():
                """Make a single API request."""
                request_start = time.time()
                try:
                    response = client.post('/api/v1/generate', 
                                         data=json.dumps({
                                             "prompt": "Test prompt",
                                             "model_type": "openai",
                                             "model_name": "gpt-4"
                                         }),
                                         content_type='application/json')
                    
                    request_time = time.time() - request_start
                    success = response.status_code in [200, 401]  # Accept auth errors
                    error = None if success else f"HTTP {response.status_code}"
                    
                    results.add_result(request_time, success, error)
                    
                except Exception as e:
                    request_time = time.time() - request_start
                    results.add_result(request_time, False, str(e))
            
            # Run concurrent requests
            num_threads = 20
            num_requests_per_thread = 5
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = []
                
                for _ in range(num_threads):
                    for _ in range(num_requests_per_thread):
                        future = executor.submit(make_request)
                        futures.append(future)
                
                # Wait for all requests to complete
                concurrent.futures.wait(futures)
            
            results.end_time = time.time()
            
            # Analyze results
            stats = results.get_statistics()
            
            # Assertions for load test
            assert stats["total_requests"] == num_threads * num_requests_per_thread
            assert stats["success_rate"] >= 80  # At least 80% success rate
            assert stats["response_times"]["mean"] < 2.0  # Average response time under 2 seconds
            assert stats["response_times"]["p95"] < 5.0  # 95th percentile under 5 seconds
            
            print(f"Load test results: {json.dumps(stats, indent=2)}")
    
    def test_database_concurrent_operations(self):
        """Test database under concurrent operation load."""
        
        from database.connection import DatabaseManager
        from database.repositories import ExperimentRepository
        
        # Use in-memory database for testing
        db_manager = DatabaseManager("sqlite:///:memory:")
        db_manager.initialize()
        
        results = LoadTestResults()
        results.start_time = time.time()
        
        def database_operation(thread_id: int):
            """Perform database operations."""
            operation_start = time.time()
            try:
                with db_manager.get_session() as session:
                    repo = ExperimentRepository(session)
                    
                    # Create experiment
                    experiment_data = {
                        "name": f"Load Test Experiment {thread_id}",
                        "description": "Load testing experiment",
                        "status": "completed",
                        "config": {"thread_id": thread_id}
                    }
                    
                    experiment = repo.create_experiment(experiment_data)
                    
                    # Query experiments
                    experiments = repo.get_experiments()
                    
                    # Update experiment
                    experiment.status = "archived"
                    repo.update_experiment(experiment.id, {"status": "archived"})
                
                operation_time = time.time() - operation_start
                results.add_result(operation_time, True)
                
            except Exception as e:
                operation_time = time.time() - operation_start
                results.add_result(operation_time, False, str(e))
        
        # Run concurrent database operations
        num_threads = 10
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(database_operation, i) for i in range(num_threads)]
            concurrent.futures.wait(futures)
        
        results.end_time = time.time()
        
        # Analyze results
        stats = results.get_statistics()
        
        # Assertions
        assert stats["total_requests"] == num_threads
        assert stats["success_rate"] >= 95  # High success rate for database operations
        assert stats["response_times"]["mean"] < 1.0  # Fast database operations
        
        print(f"Database load test results: {json.dumps(stats, indent=2)}")
    
    def test_model_loading_concurrent_access(self):
        """Test model loading under concurrent access."""
        
        from api.services.model_loader import ModelLoader
        
        # Mock transformers to avoid actual model loading
        with patch('api.services.model_loader.AutoModelForCausalLM') as mock_model, \
             patch('api.services.model_loader.AutoTokenizer') as mock_tokenizer, \
             patch('os.path.exists', return_value=True):
            
            mock_model.from_pretrained.return_value = Mock()
            mock_tokenizer.from_pretrained.return_value = Mock()
            
            loader = ModelLoader(cache_size=3)
            results = LoadTestResults()
            results.start_time = time.time()
            
            def load_model_operation(model_id: str):
                """Load a model."""
                operation_start = time.time()
                try:
                    model, tokenizer = loader.load_model(model_id)
                    operation_time = time.time() - operation_start
                    results.add_result(operation_time, True)
                    
                except Exception as e:
                    operation_time = time.time() - operation_start
                    results.add_result(operation_time, False, str(e))
            
            # Test concurrent access to same and different models
            model_ids = ["model_1", "model_2", "model_3", "model_1", "model_2"]
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = []
                
                # Submit multiple requests for each model
                for _ in range(4):  # 4 rounds
                    for model_id in model_ids:
                        future = executor.submit(load_model_operation, model_id)
                        futures.append(future)
                
                concurrent.futures.wait(futures)
            
            results.end_time = time.time()
            
            # Analyze results
            stats = results.get_statistics()
            
            # Assertions
            assert stats["success_rate"] >= 95
            assert stats["response_times"]["mean"] < 0.5  # Fast due to caching
            
            print(f"Model loading load test results: {json.dumps(stats, indent=2)}")
    
    def test_async_task_processing_load(self):
        """Test async task processing under load."""
        
        task_manager = AsyncTaskManager(max_workers=8)
        results = LoadTestResults()
        results.start_time = time.time()
        
        def cpu_task(task_id: int, duration: float = 0.1):
            """CPU-intensive task."""
            start_time = time.time()
            while time.time() - start_time < duration:
                # Simulate CPU work
                sum(i * i for i in range(1000))
            return f"Task {task_id} completed"
        
        # Submit many tasks
        num_tasks = 50
        task_ids = []
        
        submit_start = time.time()
        for i in range(num_tasks):
            task_id = task_manager.submit_task(cpu_task, i, duration=0.05)
            task_ids.append(task_id)
        
        # Wait for all tasks to complete
        completed_tasks = 0
        while completed_tasks < num_tasks:
            completed_tasks = 0
            for task_id in task_ids:
                status = task_manager.get_task_status(task_id)
                if status and status.status.value in ['completed', 'failed']:
                    completed_tasks += 1
            
            time.sleep(0.1)
        
        results.end_time = time.time()
        
        # Collect results
        for task_id in task_ids:
            status = task_manager.get_task_status(task_id)
            if status:
                duration = (status.completed_at - status.created_at).total_seconds()
                success = status.status.value == 'completed'
                error = status.error if not success else None
                results.add_result(duration, success, error)
        
        # Analyze results
        stats = results.get_statistics()
        
        # Assertions
        assert stats["total_requests"] == num_tasks
        assert stats["success_rate"] >= 95
        assert stats["response_times"]["mean"] < 1.0  # Tasks should complete quickly
        
        print(f"Async task load test results: {json.dumps(stats, indent=2)}")
        
        task_manager.shutdown()
    
    def test_cache_performance_under_load(self):
        """Test cache performance under high load."""
        
        from utils.cache_manager import CacheManager
        
        cache_manager = CacheManager(use_redis=False, memory_cache_size=1000)
        results = LoadTestResults()
        results.start_time = time.time()
        
        def cache_operation(thread_id: int):
            """Perform cache operations."""
            operation_start = time.time()
            try:
                # Mix of cache operations
                for i in range(10):
                    key = f"thread_{thread_id}_key_{i}"
                    value = f"value_{i}" * 100  # Larger values
                    
                    # Set value
                    cache_manager.set(key, value)
                    
                    # Get value
                    retrieved = cache_manager.get(key)
                    assert retrieved == value
                    
                    # Delete some values
                    if i % 3 == 0:
                        cache_manager.delete(key)
                
                operation_time = time.time() - operation_start
                results.add_result(operation_time, True)
                
            except Exception as e:
                operation_time = time.time() - operation_start
                results.add_result(operation_time, False, str(e))
        
        # Run concurrent cache operations
        num_threads = 20
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(cache_operation, i) for i in range(num_threads)]
            concurrent.futures.wait(futures)
        
        results.end_time = time.time()
        
        # Analyze results
        stats = results.get_statistics()
        
        # Assertions
        assert stats["success_rate"] >= 98  # Cache operations should be very reliable
        assert stats["response_times"]["mean"] < 0.1  # Cache should be very fast
        
        print(f"Cache load test results: {json.dumps(stats, indent=2)}")
    
    def test_memory_usage_under_load(self):
        """Test memory usage under sustained load."""
        
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Simulate memory-intensive operations
        large_objects = []
        
        def memory_intensive_operation():
            """Create and manipulate large objects."""
            # Create large data structures
            data = {f"key_{i}": [j for j in range(1000)] for i in range(100)}
            large_objects.append(data)
            
            # Process data
            processed = {}
            for key, values in data.items():
                processed[key] = sum(values)
            
            return processed
        
        # Run operations
        results = []
        for i in range(10):
            result = memory_intensive_operation()
            results.append(result)
            
            # Check memory usage
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = current_memory - initial_memory
            
            # Memory shouldn't grow excessively
            assert memory_increase < 500, f"Memory usage increased by {memory_increase}MB"
        
        # Cleanup and check for memory leaks
        large_objects.clear()
        results.clear()
        gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_leak = final_memory - initial_memory
        
        # Should return close to initial memory usage
        assert memory_leak < 50, f"Potential memory leak: {memory_leak}MB not released"
        
        print(f"Memory test: Initial={initial_memory:.1f}MB, Final={final_memory:.1f}MB, Leak={memory_leak:.1f}MB")
    
    def test_system_stability_under_stress(self):
        """Test overall system stability under stress conditions."""
        
        # Mock all external dependencies
        with patch('api.services.openai_client.OpenAI') as mock_openai, \
             patch('api.services.model_loader.AutoModelForCausalLM') as mock_model, \
             patch('api.services.model_loader.AutoTokenizer') as mock_tokenizer:
            
            mock_openai.return_value.chat.completions.create.return_value = Mock(
                choices=[Mock(message=Mock(content="Stress test response"))]
            )
            mock_model.from_pretrained.return_value = Mock()
            mock_tokenizer.from_pretrained.return_value = Mock()
            
            # Create multiple components
            from api.services.text_generator import TextGenerator
            from utils.async_processor import AsyncTaskManager
            from utils.cache_manager import CacheManager
            
            generator = TextGenerator()
            task_manager = AsyncTaskManager(max_workers=4)
            cache_manager = CacheManager(use_redis=False)
            
            # Run mixed workload
            def mixed_workload(worker_id: int):
                """Mixed operations simulating real usage."""
                try:
                    for i in range(5):
                        # Text generation
                        result = generator.generate_text(
                            f"Worker {worker_id} prompt {i}",
                            "openai"
                        )
                        
                        # Cache operations
                        cache_key = f"worker_{worker_id}_result_{i}"
                        cache_manager.set(cache_key, result)
                        cached_result = cache_manager.get(cache_key)
                        
                        # Async task
                        def simple_task():
                            return sum(range(1000))
                        
                        task_id = task_manager.submit_task(simple_task)
                        
                        # Small delay
                        time.sleep(0.01)
                    
                    return True
                    
                except Exception as e:
                    print(f"Worker {worker_id} error: {e}")
                    return False
            
            # Run stress test
            num_workers = 15
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(mixed_workload, i) for i in range(num_workers)]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
            
            # Check results
            success_count = sum(1 for result in results if result)
            success_rate = success_count / len(results) * 100
            
            # System should remain stable
            assert success_rate >= 90, f"System stability test failed: {success_rate}% success rate"
            
            print(f"Stress test completed: {success_rate}% success rate")
            
            task_manager.shutdown()


if __name__ == "__main__":
    pytest.main([__file__])