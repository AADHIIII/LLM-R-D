"""
OpenAI API client with authentication, rate limiting, and error handling.
"""
import logging
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import threading
from collections import deque

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

logger = logging.getLogger(__name__)


class RateLimiter:
    """Rate limiter for API calls."""
    
    def __init__(self, max_requests_per_minute: int = 60):
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
                    logger.info(f"Rate limit reached, sleeping for {sleep_time:.2f} seconds")
                    time.sleep(sleep_time)
                    # Remove old requests after sleeping
                    now = time.time()
                    while self.requests and self.requests[0] < now - 60:
                        self.requests.popleft()
            
            # Record this request
            self.requests.append(now)


class OpenAIClient:
    """OpenAI API client with enhanced error handling and rate limiting."""
    
    def __init__(self, api_key: str, max_requests_per_minute: int = 60):
        if not api_key or api_key.strip() == "":
            raise RuntimeError("OpenAI API key is required")
        
        self.api_key = api_key
        self.rate_limiter = RateLimiter(max_requests_per_minute)
        self._client = None
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize the OpenAI client."""
        try:
            if OpenAI is None:
                raise ImportError("OpenAI library not available")
            self._client = OpenAI(api_key=self.api_key)
            logger.info("OpenAI client initialized successfully")
        except ImportError as e:
            logger.error(f"OpenAI library not available: {e}")
            raise RuntimeError("OpenAI library not installed")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise RuntimeError(f"OpenAI client initialization failed: {e}")
    
    def generate_text(
        self,
        prompt: str,
        model: str = "gpt-3.5-turbo",
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop: Optional[List[str]] = None,
        timeout: int = 30
    ) -> Dict[str, Any]:
        """
        Generate text using OpenAI API.
        
        Args:
            prompt: Input prompt
            model: Model to use
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            stop: Stop sequences
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
            response = self._client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                timeout=timeout
            )
            
            # Extract response data
            generated_text = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason
            
            # Calculate metrics
            end_time = time.time()
            latency_ms = int((end_time - start_time) * 1000)
            
            # Get usage information
            usage = response.usage
            input_tokens = usage.prompt_tokens
            output_tokens = usage.completion_tokens
            total_tokens = usage.total_tokens
            
            # Estimate cost
            cost_usd = self._calculate_cost(model, input_tokens, output_tokens)
            
            return {
                'text': generated_text,
                'finish_reason': finish_reason,
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
            
            logger.error(f"OpenAI API error ({error_type}): {error_message}")
            
            # Handle specific error types
            if "rate_limit" in error_message.lower():
                raise RuntimeError("OpenAI rate limit exceeded. Please try again later.")
            elif "insufficient_quota" in error_message.lower():
                raise RuntimeError("OpenAI quota exceeded. Please check your billing.")
            elif "invalid_api_key" in error_message.lower():
                raise RuntimeError("Invalid OpenAI API key.")
            elif "timeout" in error_message.lower():
                raise RuntimeError("OpenAI API request timed out.")
            else:
                raise RuntimeError(f"OpenAI API error: {error_message}")
    
    def generate_batch(
        self,
        prompts: List[str],
        model: str = "gpt-3.5-turbo",
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate text for multiple prompts.
        
        Args:
            prompts: List of input prompts
            model: Model to use
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            stop: Stop sequences
            
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
                    stop=stop
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
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        List available OpenAI models.
        
        Returns:
            List of model information
        """
        try:
            # Apply rate limiting
            self.rate_limiter.wait_if_needed()
            
            models_response = self._client.models.list()
            
            # Filter for text generation models
            text_models = []
            for model in models_response.data:
                if any(prefix in model.id for prefix in ['gpt-3.5', 'gpt-4']):
                    text_models.append({
                        'id': model.id,
                        'object': model.object,
                        'created': model.created,
                        'owned_by': model.owned_by
                    })
            
            return text_models
            
        except Exception as e:
            logger.error(f"Error listing OpenAI models: {e}")
            raise RuntimeError(f"Failed to list OpenAI models: {e}")
    
    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """
        Calculate cost for OpenAI API usage.
        
        Args:
            model: Model used
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            
        Returns:
            Cost in USD
        """
        # Pricing as of 2024 (per 1K tokens)
        pricing = {
            'gpt-4': {'input': 0.03, 'output': 0.06},
            'gpt-4-32k': {'input': 0.06, 'output': 0.12},
            'gpt-3.5-turbo': {'input': 0.001, 'output': 0.002},
            'gpt-3.5-turbo-16k': {'input': 0.003, 'output': 0.004},
            'gpt-3.5-turbo-instruct': {'input': 0.0015, 'output': 0.002}
        }
        
        # Use exact model name or find closest match
        if model in pricing:
            base_model = model
        elif model.startswith('gpt-4'):
            base_model = 'gpt-4'
        elif model.startswith('gpt-3.5-turbo'):
            if '16k' in model:
                base_model = 'gpt-3.5-turbo-16k'
            elif 'instruct' in model:
                base_model = 'gpt-3.5-turbo-instruct'
            else:
                base_model = 'gpt-3.5-turbo'
        else:
            base_model = model
        
        if base_model not in pricing:
            logger.warning(f"Unknown model for pricing: {model}")
            return 0.0
        
        input_cost = (input_tokens / 1000) * pricing[base_model]['input']
        output_cost = (output_tokens / 1000) * pricing[base_model]['output']
        
        return round(input_cost + output_cost, 6)
    
    def test_connection(self) -> Dict[str, Any]:
        """
        Test the OpenAI API connection.
        
        Returns:
            Connection test results
        """
        try:
            # Simple test with minimal usage
            result = self.generate_text(
                prompt="Hello",
                model="gpt-3.5-turbo",
                max_tokens=5,
                temperature=0
            )
            
            return {
                'success': True,
                'message': 'OpenAI API connection successful',
                'test_response': result['text'][:50],  # First 50 chars
                'latency_ms': result['latency_ms']
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'OpenAI API connection failed: {str(e)}',
                'error': str(e)
            }