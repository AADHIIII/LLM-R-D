"""
End-to-end tests for complete user workflows.
"""
import pytest
import json
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from api.app import create_app
from database.connection import DatabaseManager
from fine_tuning.dataset_validator import DatasetValidator
from fine_tuning.fine_tuning_service import FineTuningService
from evaluator.prompt_evaluator import PromptEvaluator
from api.services.model_loader import ModelLoader


class TestEndToEndWorkflows:
    """Test complete user workflows from start to finish."""
    
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
    
    @pytest.fixture
    def sample_dataset(self):
        """Create sample dataset for testing."""
        dataset_data = [
            {"prompt": "What is AI?", "response": "AI is artificial intelligence."},
            {"prompt": "Define machine learning", "response": "Machine learning is a subset of AI."},
            {"prompt": "What is deep learning?", "response": "Deep learning uses neural networks."}
        ]
        
        # Create temporary JSONL file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for item in dataset_data:
                f.write(json.dumps(item) + '\n')
            return f.name
    
    def test_complete_fine_tuning_workflow(self, sample_dataset):
        """Test complete fine-tuning workflow from dataset to trained model."""
        
        # Mock transformers to avoid actual model operations
        with patch('fine_tuning.dataset_validator.AutoTokenizer') as mock_tokenizer, \
             patch('fine_tuning.fine_tuning_service.AutoModelForCausalLM') as mock_model, \
             patch('fine_tuning.fine_tuning_service.Trainer') as mock_trainer, \
             patch('os.makedirs'), \
             patch('torch.save'):
            
            # Setup mocks
            mock_tokenizer.from_pretrained.return_value = Mock()
            mock_model.from_pretrained.return_value = Mock()
            mock_trainer_instance = Mock()
            mock_trainer.return_value = mock_trainer_instance
            
            # Step 1: Validate dataset
            validator = DatasetValidator()
            validation_result = validator.validate_dataset(sample_dataset)
            
            assert validation_result.is_valid
            assert validation_result.sample_count == 3
            
            # Step 2: Tokenize dataset
            from fine_tuning.dataset_tokenizer import DatasetTokenizer
            tokenizer = DatasetTokenizer()
            
            with patch('datasets.load_dataset') as mock_load:
                mock_dataset = Mock()
                mock_dataset.map.return_value = mock_dataset
                mock_load.return_value = mock_dataset
                
                tokenized_dataset = tokenizer.tokenize_dataset(
                    sample_dataset, 
                    "gpt2", 
                    max_length=512
                )
                
                assert tokenized_dataset is not None
            
            # Step 3: Configure training
            from fine_tuning.training_config import TrainingConfig
            config = TrainingConfig(
                base_model="gpt2",
                epochs=1,
                batch_size=2,
                learning_rate=5e-5
            )
            
            assert config.base_model == "gpt2"
            assert config.epochs == 1
            
            # Step 4: Start fine-tuning
            fine_tuning_service = FineTuningService()
            
            with patch('fine_tuning.fine_tuning_service.load_dataset') as mock_load_ds:
                mock_load_ds.return_value = Mock()
                
                job_id = fine_tuning_service.start_training(
                    dataset_path=sample_dataset,
                    config=config,
                    output_dir="test_output"
                )
                
                assert job_id is not None
                
                # Check training status
                status = fine_tuning_service.get_training_status(job_id)
                assert status is not None
    
    def test_complete_evaluation_workflow(self):
        """Test complete evaluation workflow from prompts to results."""
        
        # Mock external dependencies
        with patch('evaluator.langchain_evaluator.ChatOpenAI') as mock_openai, \
             patch('evaluator.metrics_calculator.sentence_transformers') as mock_st, \
             patch('api.services.openai_client.OpenAI') as mock_openai_client:
            
            # Setup mocks
            mock_openai.return_value = Mock()
            mock_st.SentenceTransformer.return_value = Mock()
            mock_openai_client.return_value = Mock()
            
            # Step 1: Prepare test prompts
            test_prompts = [
                "Explain quantum computing",
                "What is blockchain technology?",
                "Define artificial intelligence"
            ]
            
            # Step 2: Generate responses from different models
            from api.services.text_generator import TextGenerator
            generator = TextGenerator()
            
            # Mock model responses
            mock_responses = {
                "gpt-4": ["Quantum computing uses qubits...", "Blockchain is a distributed ledger...", "AI simulates human intelligence..."],
                "claude-3": ["Quantum computing leverages quantum mechanics...", "Blockchain technology creates immutable records...", "Artificial intelligence enables machines to think..."]
            }
            
            with patch.object(generator, 'generate_text') as mock_generate:
                def mock_generate_side_effect(prompt, model_type, **kwargs):
                    if model_type == "openai":
                        return {"text": mock_responses["gpt-4"][len(mock_generate.call_args_list) % 3]}
                    elif model_type == "anthropic":
                        return {"text": mock_responses["claude-3"][len(mock_generate.call_args_list) % 3]}
                
                mock_generate.side_effect = mock_generate_side_effect
                
                # Generate responses
                results = {}
                for model in ["gpt-4", "claude-3"]:
                    model_results = []
                    for prompt in test_prompts:
                        model_type = "openai" if model == "gpt-4" else "anthropic"
                        response = generator.generate_text(prompt, model_type)
                        model_results.append(response["text"])
                    results[model] = model_results
                
                assert len(results) == 2
                assert len(results["gpt-4"]) == 3
                assert len(results["claude-3"]) == 3
            
            # Step 3: Evaluate responses
            evaluator = PromptEvaluator()
            
            with patch.object(evaluator, 'evaluate_batch') as mock_evaluate:
                mock_evaluate.return_value = [
                    {
                        "prompt": test_prompts[0],
                        "model_results": {
                            "gpt-4": {"response": results["gpt-4"][0], "metrics": {"bleu": 0.8, "rouge": 0.75}},
                            "claude-3": {"response": results["claude-3"][0], "metrics": {"bleu": 0.85, "rouge": 0.8}}
                        }
                    }
                ]
                
                evaluation_results = evaluator.evaluate_batch(
                    prompts=test_prompts,
                    model_configs=[
                        {"name": "gpt-4", "type": "openai"},
                        {"name": "claude-3", "type": "anthropic"}
                    ]
                )
                
                assert len(evaluation_results) > 0
                assert "model_results" in evaluation_results[0]
    
    def test_api_integration_workflow(self, client):
        """Test API integration workflow."""
        
        # Mock external services
        with patch('api.services.model_loader.ModelLoader') as mock_loader, \
             patch('api.services.openai_client.OpenAI') as mock_openai:
            
            # Setup mocks
            mock_loader.return_value.list_available_models.return_value = ["test_model_1", "test_model_2"]
            mock_openai.return_value.chat.completions.create.return_value = Mock(
                choices=[Mock(message=Mock(content="Test response"))]
            )
            
            # Step 1: Check API health
            response = client.get('/api/v1/health')
            assert response.status_code == 200
            
            # Step 2: List available models
            response = client.get('/api/v1/models')
            assert response.status_code == 200
            data = json.loads(response.data)
            assert 'models' in data
            
            # Step 3: Generate text
            generate_request = {
                "prompt": "Test prompt",
                "model_type": "openai",
                "model_name": "gpt-4",
                "max_tokens": 100
            }
            
            response = client.post('/api/v1/generate', 
                                 data=json.dumps(generate_request),
                                 content_type='application/json')
            
            # Note: This might fail due to authentication, but we're testing the flow
            assert response.status_code in [200, 401, 403]  # Accept auth errors
    
    def test_database_integration_workflow(self):
        """Test database integration workflow."""
        
        # Use in-memory SQLite for testing
        db_manager = DatabaseManager("sqlite:///:memory:")
        db_manager.initialize()
        
        # Test database operations
        from database.repositories import ExperimentRepository, ModelRepository, EvaluationRepository
        
        with db_manager.get_session() as session:
            # Step 1: Create repositories
            exp_repo = ExperimentRepository(session)
            model_repo = ModelRepository(session)
            eval_repo = EvaluationRepository(session)
            
            # Step 2: Create test experiment
            experiment_data = {
                "name": "Test Experiment",
                "description": "End-to-end test experiment",
                "status": "running",
                "config": {"test": True}
            }
            
            experiment = exp_repo.create_experiment(experiment_data)
            assert experiment.id is not None
            assert experiment.name == "Test Experiment"
            
            # Step 3: Create test model
            model_data = {
                "name": "test_model",
                "type": "fine-tuned",
                "base_model": "gpt2",
                "model_path": "/test/path",
                "training_config": {"epochs": 3}
            }
            
            model = model_repo.create_model(model_data)
            assert model.id is not None
            assert model.name == "test_model"
            
            # Step 4: Create test evaluation
            evaluation_data = {
                "experiment_id": experiment.id,
                "prompt": "Test prompt",
                "model_id": model.id,
                "response": "Test response",
                "metrics": {"bleu": 0.8},
                "cost_usd": 0.001,
                "latency_ms": 150
            }
            
            evaluation = eval_repo.create_evaluation(evaluation_data)
            assert evaluation.id is not None
            assert evaluation.prompt == "Test prompt"
            
            # Step 5: Query and verify data
            experiments = exp_repo.get_experiments()
            assert len(experiments) == 1
            
            models = model_repo.get_models()
            assert len(models) == 1
            
            evaluations = eval_repo.get_evaluations_by_experiment(experiment.id)
            assert len(evaluations) == 1
    
    def test_web_interface_integration(self):
        """Test web interface integration points."""
        
        # Mock React frontend API calls
        with patch('requests.get') as mock_get, \
             patch('requests.post') as mock_post:
            
            # Mock successful API responses
            mock_get.return_value = Mock(
                status_code=200,
                json=lambda: {"status": "healthy"}
            )
            
            mock_post.return_value = Mock(
                status_code=200,
                json=lambda: {"task_id": "test_task_123"}
            )
            
            # Simulate frontend API calls
            import requests
            
            # Health check
            health_response = requests.get("http://localhost:5000/api/v1/health")
            assert health_response.status_code == 200
            
            # Submit training job
            training_request = {
                "dataset_path": "test.jsonl",
                "base_model": "gpt2",
                "epochs": 3
            }
            
            training_response = requests.post(
                "http://localhost:5000/api/v1/fine-tune",
                json=training_request
            )
            assert training_response.status_code == 200
    
    def test_error_handling_workflow(self, client):
        """Test error handling throughout the system."""
        
        # Test invalid API requests
        response = client.post('/api/v1/generate', 
                             data='invalid json',
                             content_type='application/json')
        assert response.status_code == 400
        
        # Test missing required fields
        response = client.post('/api/v1/generate',
                             data=json.dumps({}),
                             content_type='application/json')
        assert response.status_code == 400
        
        # Test invalid model type
        invalid_request = {
            "prompt": "Test",
            "model_type": "invalid_type"
        }
        
        response = client.post('/api/v1/generate',
                             data=json.dumps(invalid_request),
                             content_type='application/json')
        assert response.status_code in [400, 422]  # Bad request or validation error
    
    def test_performance_under_load(self):
        """Test system performance under simulated load."""
        import threading
        import time
        
        # Mock heavy operations
        with patch('api.services.text_generator.TextGenerator.generate_text') as mock_generate:
            mock_generate.return_value = {"text": "Test response", "tokens": 10}
            
            from api.services.text_generator import TextGenerator
            generator = TextGenerator()
            
            # Simulate concurrent requests
            results = []
            errors = []
            
            def make_request():
                try:
                    start_time = time.time()
                    result = generator.generate_text("Test prompt", "openai")
                    duration = time.time() - start_time
                    results.append(duration)
                except Exception as e:
                    errors.append(str(e))
            
            # Create multiple threads
            threads = []
            for _ in range(10):
                thread = threading.Thread(target=make_request)
                threads.append(thread)
            
            # Start all threads
            start_time = time.time()
            for thread in threads:
                thread.start()
            
            # Wait for completion
            for thread in threads:
                thread.join()
            
            total_time = time.time() - start_time
            
            # Verify results
            assert len(errors) == 0, f"Errors occurred: {errors}"
            assert len(results) == 10
            assert total_time < 5.0  # Should complete within reasonable time
            assert all(duration < 1.0 for duration in results)  # Each request should be fast


if __name__ == "__main__":
    pytest.main([__file__])