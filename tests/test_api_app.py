"""
Unit tests for Flask application initialization and configuration.
"""
import pytest
import json
import os
from unittest.mock import patch, MagicMock

from api.app import create_app
from api.config import DevelopmentConfig, TestingConfig, ProductionConfig


class TestFlaskAppCreation:
    """Test Flask application factory and configuration."""
    
    def test_create_app_development(self):
        """Test creating app with development configuration."""
        app = create_app('development')
        
        assert app is not None
        assert app.config['DEBUG'] is True
        assert app.config['TESTING'] is False
        assert 'CORS_ORIGINS' in app.config
    
    def test_create_app_testing(self):
        """Test creating app with testing configuration."""
        app = create_app('testing')
        
        assert app is not None
        assert app.config['DEBUG'] is False
        assert app.config['TESTING'] is True
        assert app.config['RATE_LIMIT_ENABLED'] is False
    
    @patch.dict(os.environ, {
        'OPENAI_API_KEY': 'test-openai-key',
        'ANTHROPIC_API_KEY': 'test-anthropic-key'
    })
    def test_create_app_production(self):
        """Test creating app with production configuration."""
        app = create_app('production')
        
        assert app is not None
        assert app.config['DEBUG'] is False
        assert app.config['TESTING'] is False
        assert app.config['OPENAI_API_KEY'] == 'test-openai-key'
        assert app.config['ANTHROPIC_API_KEY'] == 'test-anthropic-key'
    
    def test_blueprints_registered(self):
        """Test that all blueprints are properly registered."""
        app = create_app('testing')
        
        # Check that blueprints are registered
        blueprint_names = [bp.name for bp in app.blueprints.values()]
        expected_blueprints = ['health', 'models', 'generate', 'evaluate']
        
        for expected_bp in expected_blueprints:
            assert expected_bp in blueprint_names
    
    def test_error_handlers_registered(self):
        """Test that error handlers are properly registered."""
        app = create_app('testing')
        
        # Test 404 error handler
        with app.test_client() as client:
            response = client.get('/nonexistent-endpoint')
            assert response.status_code == 404
            
            data = json.loads(response.data)
            assert 'error' in data
            assert data['error'] == 'not_found'
            assert 'timestamp' in data


class TestHealthEndpoints:
    """Test health check endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_app('testing')
        return app.test_client()
    
    def test_health_check(self, client):
        """Test basic health check endpoint."""
        response = client.get('/api/v1/health')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['status'] == 'healthy'
        assert 'timestamp' in data
        assert 'version' in data
    
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    @patch('psutil.cpu_percent')
    @patch('psutil.cpu_count')
    def test_detailed_status(self, mock_cpu_count, mock_cpu_percent, 
                           mock_disk_usage, mock_memory, client):
        """Test detailed status endpoint."""
        # Mock system metrics
        mock_memory.return_value = MagicMock(
            total=8000000000, available=4000000000, 
            percent=50.0, used=4000000000
        )
        mock_disk_usage.return_value = MagicMock(
            total=1000000000, free=500000000, used=500000000
        )
        mock_cpu_percent.return_value = 25.0
        mock_cpu_count.return_value = 4
        
        response = client.get('/api/v1/status')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['status'] == 'healthy'
        assert 'system' in data
        assert 'memory' in data['system']
        assert 'disk' in data['system']
        assert 'cpu' in data['system']
        assert 'environment' in data
    
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    @patch('psutil.cpu_percent')
    def test_status_degraded_high_memory(self, mock_cpu_percent, 
                                       mock_disk_usage, mock_memory, client):
        """Test status endpoint with high memory usage."""
        # Mock high memory usage
        mock_memory.return_value = MagicMock(
            total=8000000000, available=400000000, 
            percent=95.0, used=7600000000
        )
        mock_disk_usage.return_value = MagicMock(
            total=1000000000, free=500000000, used=500000000
        )
        mock_cpu_percent.return_value = 25.0
        
        response = client.get('/api/v1/status')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['status'] == 'degraded'
        assert 'warnings' in data
        assert any('memory' in warning.lower() for warning in data['warnings'])
    
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    def test_readiness_check(self, mock_disk_usage, mock_memory, client):
        """Test readiness check endpoint."""
        # Mock healthy system
        mock_memory.return_value = MagicMock(percent=50.0)
        mock_disk_usage.return_value = MagicMock(free=200000000)  # 200MB free
        
        response = client.get('/api/v1/ready')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['ready'] is True
        assert 'checks' in data
    
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    def test_readiness_check_not_ready(self, mock_disk_usage, mock_memory, client):
        """Test readiness check when system is not ready."""
        # Mock unhealthy system
        mock_memory.return_value = MagicMock(percent=98.0)  # High memory usage
        mock_disk_usage.return_value = MagicMock(free=50000000)  # Low disk space
        
        response = client.get('/api/v1/ready')
        assert response.status_code == 503
        
        data = json.loads(response.data)
        assert data['ready'] is False
        assert 'checks' in data


class TestMiddleware:
    """Test middleware functionality."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_app('testing')
        return app.test_client()
    
    def test_request_id_header(self, client):
        """Test that request ID is added to response headers."""
        response = client.get('/api/v1/health')
        assert 'X-Request-ID' in response.headers
        assert len(response.headers['X-Request-ID']) == 8
    
    def test_security_headers(self, client):
        """Test that security headers are added to responses."""
        response = client.get('/api/v1/health')
        
        # Check security headers
        assert response.headers.get('X-Frame-Options') == 'DENY'
        assert response.headers.get('X-Content-Type-Options') == 'nosniff'
        assert response.headers.get('X-XSS-Protection') == '1; mode=block'
        assert 'Content-Security-Policy' in response.headers
        assert 'Referrer-Policy' in response.headers
    
    def test_rate_limiting_disabled_in_testing(self, client):
        """Test that rate limiting is disabled in testing environment."""
        # Make multiple requests quickly
        for _ in range(10):
            response = client.get('/api/v1/health')
            assert response.status_code == 200
    
    def test_json_validation_middleware(self, client):
        """Test JSON validation middleware."""
        # Test with invalid content type
        response = client.post('/api/v1/generate', 
                             data='invalid data',
                             content_type='text/plain')
        assert response.status_code == 400
        
        data = json.loads(response.data)
        assert 'validation_error' in data['error']


class TestConfiguration:
    """Test configuration classes."""
    
    def test_development_config(self):
        """Test development configuration."""
        config = DevelopmentConfig()
        assert config.DEBUG is True
        assert config.TESTING is False
        assert config.RATE_LIMIT_PER_MINUTE == 120
        assert config.LOG_LEVEL == 'DEBUG'
    
    def test_testing_config(self):
        """Test testing configuration."""
        config = TestingConfig()
        assert config.DEBUG is False
        assert config.TESTING is True
        assert config.RATE_LIMIT_ENABLED is False
        assert config.MODEL_CACHE_SIZE == 1
    
    def test_production_config(self):
        """Test production configuration."""
        config = ProductionConfig()
        assert config.DEBUG is False
        assert config.TESTING is False
        assert config.RATE_LIMIT_PER_MINUTE == 30
    
    def test_production_config_validation_in_app(self):
        """Test production configuration validation happens in app creation."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                create_app('production')