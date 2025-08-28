"""
Anthropic API client with authentication, rate limiting, and error handling.
"""
import logging
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import threading
from collections import deque

try:
    import anthropic
except ImportError:
    # Create a mock module structure for testing
    class MockAnthropic:
        def __init__(self, api_key):
            pass
    
    class MockAnthropicModule:
        Anthropic = MockAnthropic
    
    anthropic = MockAnthropicModule()

logger = logging.getLogger(__name__)


class AnthropicRateLimiter:
    """Rate limiter for Anthropic API calls."""
    
    def __init__(self, max_requests_per_minute: int = 50):
        self.max_requests = max_requests_per_minute
        self.requests = deque()
        self.lock = threading.Lock()
    
    def wait_if_needed(self) -> None:
        """Wait if rate limit would be exceeded."""
        with self.lock:
            now = time.time()
            
            # Remove requests older than 1 minute
            while self.requests and self.requests[0] < now - 60:
                self.requests.popleft()
            
            # Check if we need to wait
            if len(self.requests) >= self.max_requests:
                sleep_time = 60 - (now - self.requests[0])
                if sleep_time > 0:
                    logger.info(f"Anthropic rate limit reached, sleeping for {sleep_time:.2f} seconds")
                    time.sleep(sleep_time)
                    # Remove old requests after sleeping
                    now = time.time()
                    while self.requests and self.requests[0] < now - 60:
                        self.requests.popleft()
            
            # Record this request
            self.requests.append(now)


class AnthropicClient:
    """Anthropic API client with enhanced error handling and rate limiting."""
    
    def __init__(self, api_key: str, max_requests_per_minute: int = 50):
        if not api_key or api_key.strip() == "":
            raise RuntimeError("Anthropic API key is required")
        
        self.api_key = api_key
        self.rate_limiter = AnthropicRateLimiter(max_requests_per_minute)
        self._client = None
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize the Anthropic client."""
        try:
            if not hasattr(anthropic, 'Anthropic'):
                raise ImportError("Anthropic library not available")
            self._client = anthropic.Anthropic(api_key=self.api_key)
            logger.info("Anthropic client initialized successfully")
        except ImportError as e:
            logger.error(f"Anthropic library not available: {e}")
            raise RuntimeError("Anthropic library not installed")
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic client: {e}")
            raise RuntimeError(f"Anthropic client initialization failed: {e}")
    
    def generate_text(
        self,
        prompt: str,
        model: str = "claude-3-sonnet-20240229",
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop_sequences: Optional[List[str]] = None,
        timeout: int = 30
    ) -> Dict[str, Any]:
        """
        Generate text using Anthropic API.
        
        Args:
            prompt: Input prompt
            model: Model to use
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            stop_sequences: Stop sequences
            timeout: Request timeout in seconds
            
        Returns:
            Dictionary with generated text and metadata
        """
        # Apply rate limiting
        self.rate_limiter.wait_if_needed()
        
        start_time = time.time()
        
        try:
            # Prepare messages
            messages = [{"role": "user", "content": prompt}]
            
            # Make API call
            response = self._client.messages.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop_sequences=stop_sequences,
                timeout=timeout
            )
            
            # Extract response data
            generated_text = response.content[0].text
            stop_reason = response.stop_reason
            
            # Calculate metrics
            end_time = time.time()
            latency_ms = int((end_time - start_time) * 1000)
            
            # Get usage information
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            total_tokens = input_tokens + output_tokens
            
            # Estimate cost
            cost_usd = self._calculate_cost(model, input_tokens, output_tokens)
            
            return {
                'text': generated_text,
                'stop_reason': stop_reason,
                'usage': {
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'total_tokens': total_tokens
                },
                'cost_usd': cost_usd,
                'latency_ms': latency_ms,
                'model': model,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            error_type = type(e).__name__
            error_message = str(e)
            
            logger.error(f"Anthropic API error ({error_type}): {error_message}")
            
            # Handle specific error types
            if "rate_limit" in error_message.lower():
                raise RuntimeError("Anthropic rate limit exceeded. Please try again later.")
            elif "insufficient_quota" in error_message.lower():
                raise RuntimeError("Anthropic quota exceeded. Please check your billing.")
            elif "invalid_api_key" in error_message.lower():
                raise RuntimeError("Invalid Anthropic API key.")
            elif "timeout" in error_message.lower():
                raise RuntimeError("Anthropic API request timed out.")
            elif "overloaded" in error_message.lower():
                raise RuntimeError("Anthropic API is overloaded. Please try again later.")
            else:
                raise RuntimeError(f"Anthropic API error: {error_message}")
    
    def generate_batch(
        self,
        prompts: List[str],
        model: str = "claude-3-sonnet-20240229",
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop_sequences: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate text for multiple prompts.
        
        Args:
            prompts: List of input prompts
            model: Model to use
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            stop_sequences: Stop sequences
            
        Returns:
            List of generation results
        """
        results = []
        
        for i, prompt in enumerate(prompts):
            try:
                result = self.generate_text(
                    prompt=prompt,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop_sequences=stop_sequences
                )
                
                results.append({
                    'index': i,
                    'prompt': prompt,
                    'success': True,
                    'result': result
                })
                
            except Exception as e:
                results.append({
                    'index': i,
                    'prompt': prompt,
                    'success': False,
                    'error': str(e)
                })
        
        return results
    
    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """
        Calculate cost for Anthropic API usage.
        
        Args:
            model: Model used
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            
        Returns:
            Cost in USD
        """
        # Pricing as of 2024 (per 1K tokens)
        pricing = {
            'claude-3-opus-20240229': {'input': 0.015, 'output': 0.075},
            'claude-3-sonnet-20240229': {'input': 0.003, 'output': 0.015},
            'claude-3-haiku-20240307': {'input': 0.00025, 'output': 0.00125},
            'claude-2.1': {'input': 0.008, 'output': 0.024},
            'claude-2.0': {'input': 0.008, 'output': 0.024},
            'claude-instant-1.2': {'input': 0.0008, 'output': 0.0024}
        }
        
        if model not in pricing:
            # Try to match by prefix
            for price_model in pricing:
                if model.startswith(price_model.split('-')[0] + '-' + price_model.split('-')[1]):
                    model = price_model
                    break
            else:
                logger.warning(f"Unknown model for pricing: {model}")
                return 0.0
        
        input_cost = (input_tokens / 1000) * pricing[model]['input']
        output_cost = (output_tokens / 1000) * pricing[model]['output']
        
        return round(input_cost + output_cost, 6)
    
    def test_connection(self) -> Dict[str, Any]:
        """
        Test the Anthropic API connection.
        
        Returns:
            Connection test results
        """
        try:
            # Simple test with minimal usage
            result = self.generate_text(
                prompt="Hello",
                model="claude-3-haiku-20240307",  # Use cheapest model for testing
                max_tokens=5,
                temperature=0
            )
            
            return {
                'success': True,
                'message': 'Anthropic API connection successful',
                'test_response': result['text'][:50],  # First 50 chars
                'latency_ms': result['latency_ms']
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'Anthropic API connection failed: {str(e)}',
                'error': str(e)
            }