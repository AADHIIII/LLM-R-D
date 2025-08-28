"""
Google Gemini API client for text generation.
"""
import logging
import time
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    genai = None
    GEMINI_AVAILABLE = False
    logger.warning("Google Generative AI library not available. Install with: pip install google-generativeai")


class GeminiClient:
    """Client for Google Gemini API interactions."""
    
    def __init__(self, api_key: str):
        """
        Initialize Gemini client.
        
        Args:
            api_key: Google AI API key
        """
        if not GEMINI_AVAILABLE:
            raise ImportError("Google Generative AI library not installed. Install with: pip install google-generativeai")
        
        self.api_key = api_key
        genai.configure(api_key=api_key)
        
        # Available models
        self.models = {
            'gemini-pro': 'gemini-pro',
            'gemini-pro-vision': 'gemini-pro-vision',
            'gemini-1.5-pro': 'gemini-1.5-pro',
            'gemini-1.5-flash': 'gemini-1.5-flash'
        }
        
        # Cost per 1K tokens (approximate, check current pricing)
        self.pricing = {
            'gemini-pro': {'input': 0.0005, 'output': 0.0015},
            'gemini-pro-vision': {'input': 0.0005, 'output': 0.0015},
            'gemini-1.5-pro': {'input': 0.00125, 'output': 0.00375},
            'gemini-1.5-flash': {'input': 0.000075, 'output': 0.0003}
        }
        
        logger.info("Gemini client initialized")
    
    def generate_text(
        self,
        prompt: str,
        model: str = 'gemini-pro',
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: int = 40,
        stop_sequences: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate text using Gemini model.
        
        Args:
            prompt: Input prompt
            model: Model name
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            stop_sequences: Stop sequences
            
        Returns:
            Dictionary with generated text and metadata
        """
        start_time = time.time()
        
        try:
            # Get the model
            gemini_model = genai.GenerativeModel(self.models.get(model, model))
            
            # Configure generation parameters
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                stop_sequences=stop_sequences or []
            )
            
            # Generate response
            response = gemini_model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Extract text from response
            generated_text = response.text if response.text else ""
            
            # Estimate token usage (Gemini doesn't provide exact counts in free tier)
            input_tokens = self._estimate_tokens(prompt)
            output_tokens = self._estimate_tokens(generated_text)
            total_tokens = input_tokens + output_tokens
            
            # Calculate cost
            cost_usd = self._calculate_cost(model, input_tokens, output_tokens)
            
            # Get finish reason
            finish_reason = self._get_finish_reason(response)
            
            return {
                'text': generated_text,
                'usage': {
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'total_tokens': total_tokens
                },
                'cost_usd': cost_usd,
                'latency_ms': latency_ms,
                'finish_reason': finish_reason,
                'timestamp': datetime.utcnow().isoformat(),
                'model': model,
                'provider': 'google'
            }
            
        except Exception as e:
            logger.error(f"Error generating text with Gemini {model}: {e}")
            raise RuntimeError(f"Gemini generation failed: {e}")
    
    def generate_batch(
        self,
        prompts: List[str],
        model: str = 'gemini-pro',
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: int = 40,
        stop_sequences: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate text for multiple prompts.
        
        Args:
            prompts: List of input prompts
            model: Model name
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
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
                    top_k=top_k,
                    stop_sequences=stop_sequences
                )
                result['batch_index'] = i
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error in batch generation for prompt {i}: {e}")
                results.append({
                    'batch_index': i,
                    'error': str(e),
                    'timestamp': datetime.utcnow().isoformat()
                })
        
        return results
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        List available Gemini models.
        
        Returns:
            List of model information
        """
        try:
            # Get available models from API
            available_models = []
            
            for model_name, model_id in self.models.items():
                try:
                    # Try to get model info
                    model_info = genai.get_model(f'models/{model_id}')
                    available_models.append({
                        'id': model_name,
                        'name': model_info.display_name if hasattr(model_info, 'display_name') else model_name,
                        'description': getattr(model_info, 'description', f'Google {model_name} model'),
                        'input_token_limit': getattr(model_info, 'input_token_limit', 30720),
                        'output_token_limit': getattr(model_info, 'output_token_limit', 2048),
                        'supported_generation_methods': getattr(model_info, 'supported_generation_methods', ['generateContent'])
                    })
                except Exception as e:
                    logger.warning(f"Could not get info for model {model_name}: {e}")
                    # Add basic info
                    available_models.append({
                        'id': model_name,
                        'name': model_name,
                        'description': f'Google {model_name} model',
                        'input_token_limit': 30720,
                        'output_token_limit': 2048,
                        'supported_generation_methods': ['generateContent']
                    })
            
            return available_models
            
        except Exception as e:
            logger.error(f"Error listing Gemini models: {e}")
            # Return default models
            return [
                {
                    'id': 'gemini-pro',
                    'name': 'Gemini Pro',
                    'description': 'Google Gemini Pro model for text generation',
                    'input_token_limit': 30720,
                    'output_token_limit': 2048,
                    'supported_generation_methods': ['generateContent']
                },
                {
                    'id': 'gemini-1.5-flash',
                    'name': 'Gemini 1.5 Flash',
                    'description': 'Fast and efficient Gemini model',
                    'input_token_limit': 1000000,
                    'output_token_limit': 8192,
                    'supported_generation_methods': ['generateContent']
                }
            ]
    
    def test_connection(self) -> Dict[str, Any]:
        """
        Test connection to Gemini API.
        
        Returns:
            Dictionary with test results
        """
        try:
            # Try a simple generation
            result = self.generate_text(
                prompt="Hello, world!",
                model='gemini-pro',
                max_tokens=10
            )
            
            return {
                'success': True,
                'latency_ms': result['latency_ms'],
                'model_tested': 'gemini-pro',
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.
        
        Args:
            text: Input text
            
        Returns:
            Estimated token count
        """
        # Rough estimation: ~4 characters per token for English text
        return max(1, len(text) // 4)
    
    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """
        Calculate cost for API usage.
        
        Args:
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            
        Returns:
            Cost in USD
        """
        if model not in self.pricing:
            model = 'gemini-pro'  # Default pricing
        
        pricing = self.pricing[model]
        
        input_cost = (input_tokens / 1000) * pricing['input']
        output_cost = (output_tokens / 1000) * pricing['output']
        
        return input_cost + output_cost
    
    def _get_finish_reason(self, response) -> str:
        """
        Extract finish reason from response.
        
        Args:
            response: Gemini API response
            
        Returns:
            Finish reason string
        """
        try:
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'finish_reason'):
                    return str(candidate.finish_reason)
            return 'stop'
        except Exception:
            return 'unknown'