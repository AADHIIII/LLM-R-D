"""
Unit tests for model inference endpoints.
"""
import pytest
import json
import os
from unittest.mock import patch, MagicMock, mock_open

from api.app import create_app


class TestModelEndpoints:
    """Test model management endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_app('testing')
        return app.test_client()
    
    def test_list_models_empty(self, client):
        """Test listing models when no models are available."""
        with patch('os.path.exists', return_value=False):
            response = client.get('/api/v1/models')
            assert response.status_code == 200
            
            data = json.loads(response.data)
            assert 'models' in data
            assert 'count' in data
            assert 'timestamp' in data
            
            # Should still have commercial models
            commercial_models = [m for m in data['models'] if m['type'] == 'commercial']
            assert len(commercial_models) > 0
    
    @patch('os.path.exists')
    @patch('os.listdir')
    @patch('glob.glob')
    def test_list_models_with_fine_tuned(self, mock_glob, mock_listdir, mock_exists, client):
        """Test listing models with fine-tuned models available."""
        # Mock file system
        mock_exists.side_effect = lambda path: path.endswith('config.json') or 'models' in path
        mock_glob.return_value = ['/path/to/models/my-model']
        mock_listdir.return_value = ['my-model']
        
        with patch('os.path.isdir', return_value=True):
            with patch('os.path.getctime', return_value=1640995200):  # 2022-01-01
                response = client.get('/api/v1/models')
                assert response.status_code == 200
                
                data = json.loads(response.data)
                assert data['count'] > 0
                
                # Check for both commercial and fine-tuned models
                model_types = [m['type'] for m in data['models']]
                assert 'commercial' in model_types
                assert 'fine-tuned' in model_types
    
    def test_get_model_info_not_found(self, client):
        """Test getting info for non-existent model."""
        response = client.get('/api/v1/models/nonexistent-model')
        assert response.status_code == 404
        
        data = json.loads(response.data)
        assert data['error'] == 'model_not_found'
        assert 'available_models' in data
    
    @patch('api.blueprints.models.get_available_models')
    def test_get_model_info_commercial(self, mock_get_models, client):
        """Test getting info for commercial model."""
        mock_get_models.return_value = [
            {
                'id': 'gpt-4',
                'name': 'GPT-4',
                'type': 'commercial',
                'provider': 'openai',
                'status': 'available'
            }
        ]
        
        response = client.get('/api/v1/models/gpt-4')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'model' in data
        assert data['model']['id'] == 'gpt-4'
        assert data['model']['type'] == 'commercial'


class TestGenerateEndpoints:
    """Test text generation endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_app('testing')
        return app.test_client()
    
    def test_generate_missing_fields(self, client):
        """Test generation with missing required fields."""
        response = client.post('/api/v1/generate', 
                             json={'prompt': 'test prompt'})  # missing model_id
        assert response.status_code == 400
        
        data = json.loads(response.data)
        assert 'validation_error' in data['error']
    
    def test_generate_empty_prompt(self, client):
        """Test generation with empty prompt."""
        response = client.post('/api/v1/generate', 
                             json={
                                 'prompt': '   ',  # empty/whitespace prompt
                                 'model_id': 'gpt-4'
                             })
        assert response.status_code == 400
        
        data = json.loads(response.data)
        assert 'validation_error' in data['error']
        assert 'empty' in data['message'].lower()
    
    def test_generate_invalid_parameters(self, client):
        """Test generation with invalid parameters."""
        # Test invalid max_tokens
        response = client.post('/api/v1/generate', 
                             json={
                                 'prompt': 'test prompt',
                                 'model_id': 'gpt-4',
                                 'max_tokens': 5000  # too high
                             })
        assert response.status_code == 400
        
        # Test invalid temperature
        response = client.post('/api/v1/generate', 
                             json={
                                 'prompt': 'test prompt',
                                 'model_id': 'gpt-4',
                                 'temperature': 3.0  # too high
                             })
        assert response.status_code == 400
    
    @patch('api.services.text_generator.TextGenerator.generate')
    def test_generate_success(self, mock_generate, client):
        """Test successful text generation."""
        # Mock successful generation
        mock_generate.return_value = {
            'text': 'Generated text response',
            'metadata': {
                'model_type': 'commercial',
                'provider': 'openai',
                'input_tokens': 10,
                'output_tokens': 15,
                'cost_usd': 0.001
            }
        }
        
        response = client.post('/api/v1/generate', 
                             json={
                                 'prompt': 'test prompt',
                                 'model_id': 'gpt-4',
                                 'max_tokens': 100,
                                 'temperature': 0.7
                             })
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['text'] == 'Generated text response'
        assert data['model_id'] == 'gpt-4'
        assert 'metrics' in data
        assert 'latency_ms' in data['metrics']
        assert 'timestamp' in data
    
    @patch('api.services.text_generator.TextGenerator.generate')
    def test_generate_error(self, mock_generate, client):
        """Test generation error handling."""
        # Mock generation error
        mock_generate.side_effect = RuntimeError("Model not available")
        
        response = client.post('/api/v1/generate', 
                             json={
                                 'prompt': 'test prompt',
                                 'model_id': 'gpt-4'
                             })
        assert response.status_code == 500
        
        data = json.loads(response.data)
        assert data['error'] == 'generation_error'
    
    def test_batch_generate_too_many_prompts(self, client):
        """Test batch generation with too many prompts."""
        prompts = ['prompt ' + str(i) for i in range(15)]  # More than limit
        
        response = client.post('/api/v1/generate/batch', 
                             json={
                                 'prompts': prompts,
                                 'model_id': 'gpt-4'
                             })
        assert response.status_code == 400
        
        data = json.loads(response.data)
        assert 'batch size' in data['message'].lower()
    
    def test_batch_generate_invalid_prompts(self, client):
        """Test batch generation with invalid prompts."""
        response = client.post('/api/v1/generate/batch', 
                             json={
                                 'prompts': ['valid prompt', '', 'another valid'],  # Empty prompt
                                 'model_id': 'gpt-4'
                             })
        assert response.status_code == 400
        
        data = json.loads(response.data)
        assert 'invalid' in data['message'].lower()
    
    @patch('api.services.text_generator.TextGenerator.generate')
    def test_batch_generate_success(self, mock_generate, client):
        """Test successful batch generation."""
        # Mock successful generation
        mock_generate.side_effect = [
            {
                'text': 'Response 1',
                'metadata': {'model_type': 'commercial'}
            },
            {
                'text': 'Response 2',
                'metadata': {'model_type': 'commercial'}
            }
        ]
        
        response = client.post('/api/v1/generate/batch', 
                             json={
                                 'prompts': ['prompt 1', 'prompt 2'],
                                 'model_id': 'gpt-4'
                             })
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert len(data['results']) == 2
        assert data['metrics']['successful_generations'] == 2
        assert data['metrics']['failed_generations'] == 0
    
    @patch('api.services.text_generator.TextGenerator.generate')
    def test_batch_generate_partial_failure(self, mock_generate, client):
        """Test batch generation with partial failures."""
        # Mock mixed success/failure
        mock_generate.side_effect = [
            {
                'text': 'Response 1',
                'metadata': {'model_type': 'commercial'}
            },
            RuntimeError("Generation failed")
        ]
        
        response = client.post('/api/v1/generate/batch', 
                             json={
                                 'prompts': ['prompt 1', 'prompt 2'],
                                 'model_id': 'gpt-4'
                             })
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert len(data['results']) == 2
        assert data['results'][0]['success'] is True
        assert data['results'][1]['success'] is False
        assert data['metrics']['successful_generations'] == 1
        assert data['metrics']['failed_generations'] == 1


class TestModelLoader:
    """Test model loader service."""
    
    def test_model_cache_basic_operations(self):
        """Test basic cache operations."""
        from api.services.model_loader import ModelCache
        
        cache = ModelCache(max_size=2)
        
        # Test put and get
        cache.put('model1', {'data': 'test1'})
        assert cache.get('model1') == {'data': 'test1'}
        
        # Test cache miss
        assert cache.get('nonexistent') is None
        
        # Test cache info
        info = cache.get_cache_info()
        assert info['size'] == 1
        assert info['max_size'] == 2
        assert 'model1' in info['models']
    
    def test_model_cache_lru_eviction(self):
        """Test LRU eviction in cache."""
        from api.services.model_loader import ModelCache
        
        cache = ModelCache(max_size=2)
        
        # Fill cache
        cache.put('model1', {'data': 'test1'})
        cache.put('model2', {'data': 'test2'})
        
        # Access model1 to make it most recent
        cache.get('model1')
        
        # Add model3, should evict model2
        cache.put('model3', {'data': 'test3'})
        
        assert cache.get('model1') is not None
        assert cache.get('model2') is None  # Evicted
        assert cache.get('model3') is not None
    
    @patch('os.path.exists')
    @patch('os.listdir')
    def test_list_available_models(self, mock_listdir, mock_exists):
        """Test listing available models."""
        from api.services.model_loader import ModelLoader
        
        # Mock file system
        mock_exists.side_effect = lambda path: (
            path.endswith('models') or 
            path.endswith('config.json')
        )
        mock_listdir.return_value = ['model1', 'model2', 'invalid_model']
        
        with patch('os.path.isdir') as mock_isdir:
            mock_isdir.side_effect = lambda path: not path.endswith('invalid_model')
            
            loader = ModelLoader()
            models = loader.list_available_models()
            
            # Should only include valid models (with config.json)
            assert 'model1' in models
            assert 'model2' in models
            assert 'invalid_model' not in models


class TestTextGenerator:
    """Test text generator service."""
    
    def test_is_commercial_model(self):
        """Test commercial model detection."""
        from api.services.text_generator import TextGenerator
        
        generator = TextGenerator()
        
        assert generator._is_commercial_model('gpt-4') is True
        assert generator._is_commercial_model('claude-3-sonnet') is True
        assert generator._is_commercial_model('my-fine-tuned-model') is False
    
    def test_estimate_openai_cost(self):
        """Test OpenAI cost estimation."""
        from api.services.text_generator import TextGenerator
        
        generator = TextGenerator()
        
        # Test GPT-4 cost
        cost = generator._estimate_openai_cost('gpt-4', 1000, 500)
        assert cost > 0
        assert isinstance(cost, float)
        
        # Test unknown model
        cost = generator._estimate_openai_cost('unknown-model', 1000, 500)
        assert cost == 0.0
    
    def test_estimate_anthropic_cost(self):
        """Test Anthropic cost estimation."""
        from api.services.text_generator import TextGenerator
        
        generator = TextGenerator()
        
        # Test Claude cost
        cost = generator._estimate_anthropic_cost('claude-3-sonnet', 1000, 500)
        assert cost > 0
        assert isinstance(cost, float)
        
        # Test unknown model
        cost = generator._estimate_anthropic_cost('unknown-model', 1000, 500)
        assert cost == 0.0