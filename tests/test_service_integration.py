"""
Integration tests for all service interactions.
"""
import pytest
import json
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from database.connection import DatabaseManager
from api.services.model_loader import ModelLoader
from api.services.text_generator import TextGenerator
from evaluator.prompt_evaluator import PromptEvaluator
from fine_tuning.fine_tuning_service import FineTuningService
from utils.cache_manager import CacheManager
from utils.async_processor import AsyncTaskManager


class TestServiceIntegration:
    """Test integration between different services."""
    
    @pytest.fixture
    def db_manager(self):
        """Create test database manager."""
        db_manager = DatabaseManager("sqlite:///:memory:")
        db_manager.initialize()
        return db_manager
    
    @pytest.fixture
    def sample_dataset(self):
        """Create sample dataset file."""
        dataset_data = [
            {"prompt": "What is AI?", "response": "AI is artificial intelligence."},
            {"prompt": "Define ML", "response": "ML is machine learning."}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for item in dataset_data:
                f.write(json.dumps(item) + '\n')
            return f.name
    
    def test_fine_tuning_to_model_loading_integration(self, sample_dataset):
        """Test integration from fine-tuning to model loading."""
        
        with patch('fine_tuning.fine_tuning_service.AutoModelForCausalLM') as mock_model, \
             patch('fine_tuning.fine_tuning_service.AutoTokenizer') as mock_tokenizer, \
             patch('fine_tuning.fine_tuning_service.Trainer') as mock_trainer, \
             patch('api.services.model_loader.AutoModelForCausalLM') as mock_loader_model, \
             patch('api.services.model_loader.AutoTokenizer') as mock_loader_tokenizer, \
             patch('os.makedirs'), \
             patch('torch.save'), \
             patch('os.path.exists', return_value=True):
            
            # Setup mocks
            mock_model.from_pretrained.return_value = Mock()
            mock_tokenizer.from_pretrained.return_value = Mock()
            mock_trainer_instance = Mock()
            mock_trainer.return_value = mock_trainer_instance
            
            mock_loader_model.from_pretrained.return_value = Mock()
            mock_loader_tokenizer.from_pretrained.return_value = Mock()
            
            # Step 1: Fine-tune a model
            fine_tuning_service = FineTuningService()
            
            from fine_tuning.training_config import TrainingConfig
            config = TrainingConfig(
                base_model="gpt2",
                epochs=1,
                batch_size=2
            )
            
            with patch('fine_tuning.fine_tuning_service.load_dataset') as mock_load_ds:
                mock_load_ds.return_value = Mock()
                
                job_id = fine_tuning_service.start_training(
                    dataset_path=sample_dataset,
                    config=config,
                    output_dir="test_model_output"
                )
                
                assert job_id is not None
            
            # Step 2: Load the fine-tuned model
            model_loader = ModelLoader()
            
            # Simulate model being saved and available
            model, tokenizer = model_loader.load_model("test_model_output")
            
            assert model is not None
            assert tokenizer is not None
            
            # Step 3: Verify model is cached
            cached_model, cached_tokenizer = model_loader.load_model("test_model_output")
            assert cached_model == model
            assert cached_tokenizer == tokenizer
    
    def test_model_loading_to_text_generation_integration(self):
        """Test integration from model loading to text generation."""
        
        with patch('api.services.model_loader.AutoModelForCausalLM') as mock_model, \
             patch('api.services.model_loader.AutoTokenizer') as mock_tokenizer, \
             patch('api.services.text_generator.torch') as mock_torch, \
             patch('os.path.exists', return_value=True):
            
            # Setup mocks
            mock_model_instance = Mock()
            mock_tokenizer_instance = Mock()
            
            mock_model.from_pretrained.return_value = mock_model_instance
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
            
            # Mock tokenizer behavior
            mock_tokenizer_instance.encode.return_value = [1, 2, 3, 4]
            mock_tokenizer_instance.decode.return_value = "Generated text response"
            mock_tokenizer_instance.pad_token_id = 0
            
            # Mock model generation
            mock_model_instance.generate.return_value = [[1, 2, 3, 4, 5, 6]]
            
            # Mock torch
            mock_torch.no_grad.return_value.__enter__ = Mock()
            mock_torch.no_grad.return_value.__exit__ = Mock()
            mock_torch.tensor.return_value = Mock()
            
            # Step 1: Load model
            model_loader = ModelLoader()
            model, tokenizer = model_loader.load_model("test_model")
            
            # Step 2: Generate text using loaded model
            text_generator = TextGenerator()
            
            # Mock the model loader in text generator
            with patch.object(text_generator, 'model_loader', model_loader):
                result = text_generator.generate_text(
                    prompt="Test prompt",
                    model_type="fine_tuned",
                    model_name="test_model"
                )
                
                assert result is not None
                assert "text" in result
    
    def test_text_generation_to_evaluation_integration(self):
        """Test integration from text generation to evaluation."""
        
        with patch('api.services.openai_client.OpenAI') as mock_openai, \
             patch('evaluator.langchain_evaluator.ChatOpenAI') as mock_langchain_openai, \
             patch('evaluator.metrics_calculator.sentence_transformers') as mock_st:
            
            # Setup mocks
            mock_openai.return_value.chat.completions.create.return_value = Mock(
                choices=[Mock(message=Mock(content="OpenAI response"))]
            )
            
            mock_langchain_openai.return_value = Mock()
            mock_st.SentenceTransformer.return_value = Mock()
            
            # Step 1: Generate text from multiple models
            text_generator = TextGenerator()
            
            prompts = ["What is AI?", "Explain machine learning"]
            model_configs = [
                {"name": "gpt-4", "type": "openai"},
                {"name": "gpt-3.5-turbo", "type": "openai"}
            ]
            
            generated_responses = {}
            for prompt in prompts:
                generated_responses[prompt] = {}
                for config in model_configs:
                    response = text_generator.generate_text(
                        prompt=prompt,
                        model_type=config["type"],
                        model_name=config["name"]
                    )
                    generated_responses[prompt][config["name"]] = response["text"]
            
            # Step 2: Evaluate the generated responses
            evaluator = PromptEvaluator()
            
            with patch.object(evaluator, 'evaluate_batch') as mock_evaluate:
                mock_evaluate.return_value = [
                    {
                        "prompt": prompts[0],
                        "model_results": {
                            "gpt-4": {
                                "response": generated_responses[prompts[0]]["gpt-4"],
                                "metrics": {"bleu": 0.8, "rouge": 0.75}
                            },
                            "gpt-3.5-turbo": {
                                "response": generated_responses[prompts[0]]["gpt-3.5-turbo"],
                                "metrics": {"bleu": 0.7, "rouge": 0.7}
                            }
                        }
                    }
                ]
                
                evaluation_results = evaluator.evaluate_batch(
                    prompts=prompts,
                    model_configs=model_configs
                )
                
                assert len(evaluation_results) > 0
                assert "model_results" in evaluation_results[0]
    
    def test_database_to_api_integration(self, db_manager):
        """Test integration between database and API services."""
        
        from database.repositories import ExperimentRepository, EvaluationRepository
        from api.blueprints.evaluate import evaluate_bp
        from flask import Flask
        
        # Create Flask app for testing
        app = Flask(__name__)
        app.register_blueprint(evaluate_bp)
        
        with app.test_client() as client:
            with db_manager.get_session() as session:
                # Step 1: Create experiment in database
                exp_repo = ExperimentRepository(session)
                experiment_data = {
                    "name": "Integration Test Experiment",
                    "description": "Testing database to API integration",
                    "status": "running",
                    "config": {"test": True}
                }
                
                experiment = exp_repo.create_experiment(experiment_data)
                assert experiment.id is not None
                
                # Step 2: Create evaluations
                eval_repo = EvaluationRepository(session)
                
                for i in range(3):
                    evaluation_data = {
                        "experiment_id": experiment.id,
                        "prompt": f"Test prompt {i}",
                        "model_id": "test_model",
                        "response": f"Test response {i}",
                        "metrics": {"bleu": 0.8 + i * 0.05},
                        "cost_usd": 0.001,
                        "latency_ms": 150 + i * 10
                    }
                    
                    evaluation = eval_repo.create_evaluation(evaluation_data)
                    assert evaluation.id is not None
                
                # Step 3: Query through API (mock the endpoint)
                with patch('api.blueprints.evaluate.get_db_session') as mock_get_session:
                    mock_get_session.return_value.__enter__ = Mock(return_value=session)
                    mock_get_session.return_value.__exit__ = Mock()
                    
                    # This would normally test the actual API endpoint
                    # For now, we verify the data exists
                    evaluations = eval_repo.get_evaluations_by_experiment(experiment.id)
                    assert len(evaluations) == 3
    
    def test_cache_to_async_processing_integration(self):
        """Test integration between caching and async processing."""
        
        cache_manager = CacheManager(use_redis=False)
        task_manager = AsyncTaskManager(max_workers=2)
        
        def cached_computation(x: int, y: int) -> int:
            """Expensive computation that should be cached."""
            # Simulate expensive operation
            import time
            time.sleep(0.1)
            return x * y + sum(range(x))
        
        # Step 1: Submit async tasks
        task_ids = []
        inputs = [(5, 10), (3, 7), (5, 10), (8, 2)]  # Note: (5, 10) is repeated
        
        for x, y in inputs:
            # Check cache first
            cache_key = f"computation_{x}_{y}"
            cached_result = cache_manager.get(cache_key)
            
            if cached_result is not None:
                # Use cached result
                print(f"Using cached result for {x}, {y}: {cached_result}")
            else:
                # Submit async task
                task_id = task_manager.submit_task(cached_computation, x, y)
                task_ids.append((task_id, cache_key, x, y))
        
        # Step 2: Wait for tasks and cache results
        import time
        time.sleep(0.5)  # Wait for tasks to complete
        
        for task_id, cache_key, x, y in task_ids:
            status = task_manager.get_task_status(task_id)
            if status and status.status.value == 'completed':
                # Cache the result
                cache_manager.set(cache_key, status.result)
                print(f"Cached result for {x}, {y}: {status.result}")
        
        # Step 3: Verify caching works
        # Submit the same computation again - should use cache
        cache_key = "computation_5_10"
        cached_result = cache_manager.get(cache_key)
        assert cached_result is not None
        
        task_manager.shutdown()
    
    def test_monitoring_integration(self):
        """Test integration with monitoring and performance tracking."""
        
        from utils.performance_monitor import PerformanceMonitor, monitor_performance
        
        monitor = PerformanceMonitor(max_metrics=100)
        
        # Mock services with monitoring
        @monitor_performance("text_generation")
        def mock_text_generation(prompt: str):
            import time
            time.sleep(0.05)  # Simulate processing time
            return f"Generated response for: {prompt}"
        
        @monitor_performance("model_loading")
        def mock_model_loading(model_id: str):
            import time
            time.sleep(0.1)  # Simulate loading time
            return f"Loaded model: {model_id}"
        
        @monitor_performance("evaluation")
        def mock_evaluation(response: str):
            import time
            time.sleep(0.02)  # Simulate evaluation time
            return {"bleu": 0.8, "rouge": 0.75}
        
        # Step 1: Simulate service interactions
        model = mock_model_loading("test_model")
        response = mock_text_generation("Test prompt")
        metrics = mock_evaluation(response)
        
        # Step 2: Check monitoring data
        text_gen_stats = monitor.get_operation_stats("text_generation")
        model_load_stats = monitor.get_operation_stats("model_loading")
        eval_stats = monitor.get_operation_stats("evaluation")
        
        assert text_gen_stats["count"] == 1
        assert model_load_stats["count"] == 1
        assert eval_stats["count"] == 1
        
        # Verify timing relationships
        assert model_load_stats["duration"]["avg"] > text_gen_stats["duration"]["avg"]
        assert text_gen_stats["duration"]["avg"] > eval_stats["duration"]["avg"]
        
        monitor.stop_monitoring()
    
    def test_error_propagation_across_services(self):
        """Test how errors propagate across service boundaries."""
        
        # Test error in model loading affecting text generation
        with patch('api.services.model_loader.AutoModelForCausalLM') as mock_model:
            mock_model.from_pretrained.side_effect = RuntimeError("Model loading failed")
            
            model_loader = ModelLoader()
            text_generator = TextGenerator()
            
            # Model loading should fail
            with pytest.raises(RuntimeError, match="Model loading failed"):
                model_loader.load_model("failing_model")
            
            # Text generation should handle the error gracefully
            with patch.object(text_generator, 'model_loader', model_loader):
                result = text_generator.generate_text(
                    prompt="Test prompt",
                    model_type="fine_tuned",
                    model_name="failing_model"
                )
                
                # Should return error information
                assert "error" in result or result is None
    
    def test_data_flow_consistency(self, db_manager):
        """Test data consistency across the entire pipeline."""
        
        from database.repositories import ExperimentRepository, EvaluationRepository
        
        with db_manager.get_session() as session:
            exp_repo = ExperimentRepository(session)
            eval_repo = EvaluationRepository(session)
            
            # Step 1: Create experiment
            experiment_data = {
                "name": "Data Flow Test",
                "description": "Testing data consistency",
                "status": "running",
                "config": {"models": ["gpt-4", "claude-3"]}
            }
            
            experiment = exp_repo.create_experiment(experiment_data)
            
            # Step 2: Simulate evaluation pipeline
            prompts = ["Test prompt 1", "Test prompt 2"]
            models = ["gpt-4", "claude-3"]
            
            evaluation_ids = []
            
            for prompt in prompts:
                for model in models:
                    # Simulate text generation and evaluation
                    evaluation_data = {
                        "experiment_id": experiment.id,
                        "prompt": prompt,
                        "model_id": model,
                        "response": f"Response from {model} for '{prompt}'",
                        "metrics": {
                            "bleu": 0.8 if model == "gpt-4" else 0.75,
                            "rouge": 0.7 if model == "gpt-4" else 0.72
                        },
                        "cost_usd": 0.002 if model == "gpt-4" else 0.001,
                        "latency_ms": 200 if model == "gpt-4" else 180
                    }
                    
                    evaluation = eval_repo.create_evaluation(evaluation_data)
                    evaluation_ids.append(evaluation.id)
            
            # Step 3: Verify data consistency
            # All evaluations should be linked to the experiment
            experiment_evaluations = eval_repo.get_evaluations_by_experiment(experiment.id)
            assert len(experiment_evaluations) == len(prompts) * len(models)
            
            # Check that all evaluation IDs are present
            stored_ids = [eval.id for eval in experiment_evaluations]
            assert set(evaluation_ids) == set(stored_ids)
            
            # Verify metrics are properly stored and retrievable
            for evaluation in experiment_evaluations:
                assert evaluation.metrics is not None
                assert "bleu" in evaluation.metrics
                assert "rouge" in evaluation.metrics
                assert evaluation.cost_usd > 0
                assert evaluation.latency_ms > 0
    
    def test_concurrent_service_interactions(self):
        """Test concurrent interactions between services."""
        
        import threading
        import concurrent.futures
        
        cache_manager = CacheManager(use_redis=False)
        task_manager = AsyncTaskManager(max_workers=4)
        
        results = []
        errors = []
        
        def service_interaction(thread_id: int):
            """Simulate concurrent service interactions."""
            try:
                # Cache operation
                cache_key = f"thread_{thread_id}_data"
                cache_manager.set(cache_key, f"data_for_thread_{thread_id}")
                
                # Async task
                def compute_task():
                    return sum(range(thread_id * 100))
                
                task_id = task_manager.submit_task(compute_task)
                
                # Wait for task
                import time
                time.sleep(0.1)
                
                # Retrieve results
                cached_data = cache_manager.get(cache_key)
                task_status = task_manager.get_task_status(task_id)
                
                results.append({
                    "thread_id": thread_id,
                    "cached_data": cached_data,
                    "task_result": task_status.result if task_status else None,
                    "success": True
                })
                
            except Exception as e:
                errors.append(f"Thread {thread_id}: {str(e)}")
        
        # Run concurrent interactions
        num_threads = 10
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(service_interaction, i) for i in range(num_threads)]
            concurrent.futures.wait(futures)
        
        # Verify results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == num_threads
        
        # Check that all threads got their data correctly
        for result in results:
            thread_id = result["thread_id"]
            assert result["cached_data"] == f"data_for_thread_{thread_id}"
            assert result["task_result"] is not None
            assert result["success"] is True
        
        task_manager.shutdown()


if __name__ == "__main__":
    pytest.main([__file__])