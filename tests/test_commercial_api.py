"""
Integration tests for commercial API clients.
"""
import pytest
import json
import time
from unittest.mock import patch, MagicMock

from api.services.openai_client import OpenAIClient, RateLimiter
from api.services.anthropic_client import AnthropicClient, AnthropicRateLimiter
from api.services.commercial_api_service import CommercialAPIService


class TestRateLimiter:
    """Test rate limiting functionality."""
    
    def test_rate_limiter_basic(self):
        """Test basic rate limiter functionality."""
        limiter = RateLimiter(max_requests_per_minute=2)
        
        # Should allow first two requests immediately
        start_time = time.time()
        limiter.wait_if_needed()
        limiter.wait_if_needed()
        elapsed = time.time() - start_time
        
        # Should be very fast (no waiting)
        assert elapsed < 0.1
    
    def test_rate_limiter_blocking(self):
        """Test that rate limiter blocks when limit is exceeded."""
        limiter = RateLimiter(max_requests_per_minute=1)
        
        # First request should be immediate
        limiter.wait_if_needed()
        
        # Second request should be blocked (but we won't wait for it in test)
        # Just verify the internal state
        assert len(limiter.requests) == 1


class TestOpenAIClient:
    """Test OpenAI client functionality."""
    
    def test_client_initialization_no_api_key(self):
        """Test client initialization without API key."""
        with pytest.raises(RuntimeError):
            OpenAIClient("")
    
    @patch('api.services.openai_client.OpenAI')
    def test_client_initialization_success(self, mock_openai):
        """Test successful client initialization."""
        mock_openai.return_value = MagicMock()
        
        client = OpenAIClient("test-api-key")
        assert client.api_key == "test-api-key"
        assert client._client is not None
    
    @patch('api.services.openai_client.OpenAI')
    def test_generate_text_success(self, mock_openai):
        """Test successful text generation."""
        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Generated text"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15
        
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        client = OpenAIClient("test-api-key")
        result = client.generate_text("Test prompt", "gpt-3.5-turbo")
        
        assert result['text'] == "Generated text"
        assert result['finish_reason'] == "stop"
        assert result['usage']['input_tokens'] == 10
        assert result['usage']['output_tokens'] == 5
        assert result['cost_usd'] > 0
        assert 'latency_ms' in result
    
    @patch('api.services.openai_client.OpenAI')
    def test_generate_text_error_handling(self, mock_openai):
        """Test error handling in text generation."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_openai.return_value = mock_client
        
        client = OpenAIClient("test-api-key")
        
        with pytest.raises(RuntimeError, match="OpenAI API error"):
            client.generate_text("Test prompt", "gpt-3.5-turbo")
    
    @patch('api.services.openai_client.OpenAI')
    def test_rate_limit_error_handling(self, mock_openai):
        """Test rate limit error handling."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("rate_limit exceeded")
        mock_openai.return_value = mock_client
        
        client = OpenAIClient("test-api-key")
        
        with pytest.raises(RuntimeError, match="rate limit exceeded"):
            client.generate_text("Test prompt", "gpt-3.5-turbo")
    
    def test_cost_calculation(self):
        """Test cost calculation for different models."""
        client = OpenAIClient.__new__(OpenAIClient)  # Create without initialization
        
        # Test GPT-4 cost
        cost = client._calculate_cost("gpt-4", 1000, 500)
        assert cost > 0
        assert isinstance(cost, float)
        
        # Test GPT-3.5 cost (should be cheaper)
        cost_35 = client._calculate_cost("gpt-3.5-turbo", 1000, 500)
        assert cost_35 > 0
        assert cost_35 < cost
        
        # Test unknown model
        cost_unknown = client._calculate_cost("unknown-model", 1000, 500)
        assert cost_unknown == 0.0
    
    @patch('api.services.openai_client.OpenAI')
    def test_batch_generation(self, mock_openai):
        """Test batch text generation."""
        # Mock successful responses
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Generated text"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15
        
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        client = OpenAIClient("test-api-key")
        results = client.generate_batch(["prompt1", "prompt2"], "gpt-3.5-turbo")
        
        assert len(results) == 2
        assert all(r['success'] for r in results)
        assert all('result' in r for r in results)


class TestAnthropicClient:
    """Test Anthropic client functionality."""
    
    def test_client_initialization_no_api_key(self):
        """Test client initialization without API key."""
        with pytest.raises(RuntimeError):
            AnthropicClient("")
    
    @patch('api.services.anthropic_client.anthropic.Anthropic')
    def test_client_initialization_success(self, mock_anthropic):
        """Test successful client initialization."""
        mock_anthropic.return_value = MagicMock()
        
        client = AnthropicClient("test-api-key")
        assert client.api_key == "test-api-key"
        assert client._client is not None
    
    @patch('api.services.anthropic_client.anthropic.Anthropic')
    def test_generate_text_success(self, mock_anthropic):
        """Test successful text generation."""
        # Mock Anthropic response
        mock_response = MagicMock()
        mock_response.content[0].text = "Generated text"
        mock_response.stop_reason = "end_turn"
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5
        
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client
        
        client = AnthropicClient("test-api-key")
        result = client.generate_text("Test prompt", "claude-3-sonnet-20240229")
        
        assert result['text'] == "Generated text"
        assert result['stop_reason'] == "end_turn"
        assert result['usage']['input_tokens'] == 10
        assert result['usage']['output_tokens'] == 5
        assert result['cost_usd'] > 0
        assert 'latency_ms' in result
    
    def test_cost_calculation(self):
        """Test cost calculation for different Claude models."""
        client = AnthropicClient.__new__(AnthropicClient)  # Create without initialization
        
        # Test Claude 3 Sonnet cost
        cost = client._calculate_cost("claude-3-sonnet-20240229", 1000, 500)
        assert cost > 0
        assert isinstance(cost, float)
        
        # Test Claude 3 Haiku cost (should be cheaper)
        cost_haiku = client._calculate_cost("claude-3-haiku-20240307", 1000, 500)
        assert cost_haiku > 0
        assert cost_haiku < cost
        
        # Test unknown model
        cost_unknown = client._calculate_cost("unknown-model", 1000, 500)
        assert cost_unknown == 0.0


class TestCommercialAPIService:
    """Test unified commercial API service."""
    
    def test_service_initialization_no_keys(self):
        """Test service initialization without API keys."""
        service = CommercialAPIService()
        assert service.openai_client is None
        assert service.anthropic_client is None
    
    @patch('api.services.commercial_api_service.OpenAIClient')
    @patch('api.services.commercial_api_service.AnthropicClient')
    def test_service_initialization_with_keys(self, mock_anthropic_client, mock_openai_client):
        """Test service initialization with API keys."""
        mock_openai_client.return_value = MagicMock()
        mock_anthropic_client.return_value = MagicMock()
        
        service = CommercialAPIService(
            openai_api_key="openai-key",
            anthropic_api_key="anthropic-key"
        )
        
        assert service.openai_client is not None
        assert service.anthropic_client is not None
    
    def test_model_type_detection(self):
        """Test model type detection."""
        service = CommercialAPIService()
        
        assert service._is_openai_model("gpt-4") is True
        assert service._is_openai_model("gpt-3.5-turbo") is True
        assert service._is_openai_model("claude-3-sonnet") is False
        
        assert service._is_anthropic_model("claude-3-sonnet") is True
        assert service._is_anthropic_model("claude-instant") is True
        assert service._is_anthropic_model("gpt-4") is False
    
    @patch('api.services.commercial_api_service.OpenAIClient')
    def test_generate_text_openai(self, mock_openai_client):
        """Test text generation with OpenAI model."""
        mock_client = MagicMock()
        mock_client.generate_text.return_value = {
            'text': 'Generated text',
            'usage': {'input_tokens': 10, 'output_tokens': 5, 'total_tokens': 15},
            'cost_usd': 0.001,
            'latency_ms': 500,
            'finish_reason': 'stop',
            'timestamp': '2024-01-01T00:00:00'
        }
        mock_openai_client.return_value = mock_client
        
        service = CommercialAPIService(openai_api_key="test-key")
        result = service.generate_text("Test prompt", "gpt-4")
        
        assert result['text'] == 'Generated text'
        assert result['metadata']['provider'] == 'openai'
        assert result['metadata']['model_type'] == 'commercial'
    
    def test_generate_text_unsupported_model(self):
        """Test text generation with unsupported model."""
        service = CommercialAPIService()
        
        with pytest.raises(ValueError, match="Unsupported commercial model"):
            service.generate_text("Test prompt", "unsupported-model")
    
    @patch('api.services.commercial_api_service.AnthropicClient')
    def test_list_available_models(self, mock_anthropic_client):
        """Test listing available models."""
        mock_anthropic_client.return_value = MagicMock()
        
        service = CommercialAPIService(anthropic_api_key="test-key")
        models = service.list_available_models()
        
        # Should include Anthropic models
        anthropic_models = [m for m in models if m['provider'] == 'anthropic']
        assert len(anthropic_models) > 0
        
        # Check model structure
        for model in anthropic_models:
            assert 'id' in model
            assert 'name' in model
            assert 'provider' in model
            assert 'type' in model
            assert model['type'] == 'commercial'
    
    @patch('api.services.commercial_api_service.OpenAIClient')
    @patch('api.services.commercial_api_service.AnthropicClient')
    def test_test_connections(self, mock_anthropic_client, mock_openai_client):
        """Test connection testing functionality."""
        # Mock successful connections
        mock_openai = MagicMock()
        mock_openai.test_connection.return_value = {'success': True, 'message': 'OK'}
        mock_openai_client.return_value = mock_openai
        
        mock_anthropic = MagicMock()
        mock_anthropic.test_connection.return_value = {'success': True, 'message': 'OK'}
        mock_anthropic_client.return_value = mock_anthropic
        
        service = CommercialAPIService(
            openai_api_key="openai-key",
            anthropic_api_key="anthropic-key"
        )
        
        results = service.test_connections()
        
        assert 'tests' in results
        assert 'openai' in results['tests']
        assert 'anthropic' in results['tests']
        assert results['tests']['openai']['success'] is True
        assert results['tests']['anthropic']['success'] is True
        assert results['overall_success'] is True
    
    def test_test_connections_no_clients(self):
        """Test connection testing with no clients initialized."""
        service = CommercialAPIService()
        results = service.test_connections()
        
        assert results['tests']['openai']['success'] is False
        assert results['tests']['anthropic']['success'] is False
        assert results['overall_success'] is False