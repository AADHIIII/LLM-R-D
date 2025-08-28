"""
Integration tests for the complete API gateway.
"""
import pytest
import json
from unittest.mock import patch, MagicMock

from api.app import create_app


class TestAPIIntegration:
    """Test complete API integration."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_app('testing')
        return app.test_client()
    
    def test_health_endpoints(self, client):
        """Test all health endpoints work."""
        # Basic health check
        response = client.get('/api/v1/health')
        assert response.status_code == 200
        
        # Detailed status
        with patch('api.blueprints.health.psutil.virtual_memory') as mock_mem, \
             patch('api.blueprints.health.psutil.disk_usage') as mock_disk, \
             patch('api.blueprints.health.psutil.cpu_percent') as mock_cpu:
            
            mock_mem.return_value = MagicMock(total=8000000000, available=4000000000, percent=50.0, used=4000000000)
            mock_disk.return_value = MagicMock(total=1000000000, free=500000000, used=500000000)
            mock_cpu.return_value = 25.0
            
            response = client.get('/api/v1/status')
            assert response.status_code == 200
        
        # Readiness check
        with patch('api.blueprints.health.psutil.virtual_memory') as mock_mem, \
             patch('api.blueprints.health.psutil.disk_usage') as mock_disk:
            
            mock_mem.return_value = MagicMock(percent=50.0)
            mock_disk.return_value = MagicMock(free=200000000)  # 200MB free
            
            response = client.get('/api/v1/ready')
            assert response.status_code in [200, 503]  # Either ready or not ready
    
    def test_models_endpoints(self, client):
        """Test model management endpoints."""
        # List models
        response = client.get('/api/v1/models')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'models' in data
        assert 'count' in data
        
        # Get specific model info (should return 404 for non-existent model)
        response = client.get('/api/v1/models/nonexistent-model')
        assert response.status_code == 404
    
    def test_commercial_endpoints(self, client):
        """Test commercial API endpoints."""
        # Test commercial API connections
        response = client.get('/api/v1/commercial/test')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'tests' in data
        assert 'overall_success' in data
        
        # List commercial models
        response = client.get('/api/v1/commercial/models')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'models' in data
        assert 'count' in data
        assert 'providers' in data
    
    def test_generate_endpoint_validation(self, client):
        """Test text generation endpoint validation."""
        # Test missing required fields
        response = client.post('/api/v1/generate', json={})
        assert response.status_code == 400
        
        # Test invalid parameters
        response = client.post('/api/v1/generate', json={
            'prompt': 'test',
            'model_id': 'gpt-4',
            'max_tokens': 5000  # Too high
        })
        assert response.status_code == 400
        
        # Test empty prompt
        response = client.post('/api/v1/generate', json={
            'prompt': '   ',  # Empty/whitespace
            'model_id': 'gpt-4'
        })
        assert response.status_code == 400
    
    @patch('api.services.text_generator.TextGenerator.generate')
    def test_generate_endpoint_success(self, mock_generate, client):
        """Test successful text generation."""
        # Mock successful generation
        mock_generate.return_value = {
            'text': 'Generated response',
            'metadata': {
                'model_type': 'commercial',
                'provider': 'openai',
                'input_tokens': 5,
                'output_tokens': 10,
                'cost_usd': 0.001
            }
        }
        
        response = client.post('/api/v1/generate', json={
            'prompt': 'Hello, world!',
            'model_id': 'gpt-3.5-turbo',
            'max_tokens': 50,
            'temperature': 0.7
        })
        
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['text'] == 'Generated response'
        assert data['model_id'] == 'gpt-3.5-turbo'
        assert 'metrics' in data
        assert 'timestamp' in data
    
    def test_batch_generate_endpoint(self, client):
        """Test batch generation endpoint validation."""
        # Test too many prompts
        prompts = ['prompt ' + str(i) for i in range(15)]  # More than limit
        response = client.post('/api/v1/generate/batch', json={
            'prompts': prompts,
            'model_id': 'gpt-4'
        })
        assert response.status_code == 400
        
        # Test invalid prompts
        response = client.post('/api/v1/generate/batch', json={
            'prompts': ['valid', '', 'also valid'],  # Empty prompt in middle
            'model_id': 'gpt-4'
        })
        assert response.status_code == 400
    
    def test_error_handling(self, client):
        """Test API error handling."""
        # Test 404 for non-existent endpoint
        response = client.get('/api/v1/nonexistent')
        assert response.status_code == 404
        
        data = json.loads(response.data)
        assert data['error'] == 'not_found'
        assert 'timestamp' in data
    
    def test_security_headers(self, client):
        """Test that security headers are present."""
        response = client.get('/api/v1/health')
        
        # Check for security headers
        assert 'X-Frame-Options' in response.headers
        assert 'X-Content-Type-Options' in response.headers
        assert 'X-XSS-Protection' in response.headers
        assert 'Content-Security-Policy' in response.headers
        assert 'X-Request-ID' in response.headers
    
    def test_cors_headers(self, client):
        """Test CORS configuration."""
        # Make an OPTIONS request to check CORS
        response = client.options('/api/v1/health')
        
        # Should not error (CORS is configured)
        assert response.status_code in [200, 204]
    
    def test_request_validation_middleware(self, client):
        """Test request validation middleware."""
        # Test invalid JSON
        response = client.post('/api/v1/generate',
                             data='invalid json',
                             content_type='application/json')
        assert response.status_code == 400
        
        # Test non-JSON content type for POST
        response = client.post('/api/v1/generate',
                             data='some data',
                             content_type='text/plain')
        assert response.status_code == 400


class TestAPIConfiguration:
    """Test API configuration in different environments."""
    
    def test_development_config(self):
        """Test development configuration."""
        app = create_app('development')
        assert app.config['DEBUG'] is True
        assert app.config['TESTING'] is False
    
    def test_testing_config(self):
        """Test testing configuration."""
        app = create_app('testing')
        assert app.config['DEBUG'] is False
        assert app.config['TESTING'] is True
        assert app.config['RATE_LIMIT_ENABLED'] is False
    
    @patch.dict('os.environ', {
        'OPENAI_API_KEY': 'test-openai-key',
        'ANTHROPIC_API_KEY': 'test-anthropic-key'
    })
    def test_production_config(self):
        """Test production configuration."""
        app = create_app('production')
        assert app.config['DEBUG'] is False
        assert app.config['TESTING'] is False
        assert app.config['OPENAI_API_KEY'] == 'test-openai-key'
        assert app.config['ANTHROPIC_API_KEY'] == 'test-anthropic-key'