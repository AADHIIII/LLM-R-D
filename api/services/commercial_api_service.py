"""
Unified commercial API service for managing OpenAI and Anthropic clients.
"""
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import threading

from .openai_client import OpenAIClient
from .anthropic_client import AnthropicClient
from .gemini_client import GeminiClient

logger = logging.getLogger(__name__)


class CommercialAPIService:
    """Unified service for commercial API interactions."""
    
    def __init__(self, openai_api_key: Optional[str] = None, anthropic_api_key: Optional[str] = None, gemini_api_key: Optional[str] = None):
        self.openai_client = None
        self.anthropic_client = None
        self.gemini_client = None
        self.lock = threading.Lock()
        
        # Initialize clients if API keys are provided
        if openai_api_key:
            try:
                self.openai_client = OpenAIClient(openai_api_key)
                logger.info("OpenAI client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
        
        if anthropic_api_key:
            try:
                self.anthropic_client = AnthropicClient(anthropic_api_key)
                logger.info("Anthropic client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Anthropic client: {e}")
        
        if gemini_api_key:
            try:
                self.gemini_client = GeminiClient(gemini_api_key)
                logger.info("Gemini client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini client: {e}")
    
    def generate_text(
        self,
        prompt: str,
        model_id: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate text using the specified commercial model.
        
        Args:
            prompt: Input prompt
            model_id: Model identifier (e.g., 'gpt-4', 'claude-3-sonnet')
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            stop: Stop sequences
            
        Returns:
            Dictionary with generated text and metadata
        """
        try:
            if self._is_openai_model(model_id):
                return self._generate_openai(
                    prompt, model_id, max_tokens, temperature, top_p, stop
                )
            elif self._is_anthropic_model(model_id):
                return self._generate_anthropic(
                    prompt, model_id, max_tokens, temperature, top_p, stop
                )
            elif self._is_gemini_model(model_id):
                return self._generate_gemini(
                    prompt, model_id, max_tokens, temperature, top_p, stop
                )
            else:
                raise ValueError(f"Unsupported commercial model: {model_id}")
                
        except Exception as e:
            logger.error(f"Error generating text with {model_id}: {e}")
            raise
    
    def generate_batch(
        self,
        prompts: List[str],
        model_id: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate text for multiple prompts using the specified commercial model.
        
        Args:
            prompts: List of input prompts
            model_id: Model identifier
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            stop: Stop sequences
            
        Returns:
            List of generation results
        """
        try:
            if self._is_openai_model(model_id):
                if not self.openai_client:
                    raise RuntimeError("OpenAI client not initialized")
                return self.openai_client.generate_batch(
                    prompts, model_id, max_tokens, temperature, top_p, stop
                )
            elif self._is_anthropic_model(model_id):
                if not self.anthropic_client:
                    raise RuntimeError("Anthropic client not initialized")
                return self.anthropic_client.generate_batch(
                    prompts, model_id, max_tokens, temperature, top_p, stop
                )
            else:
                raise ValueError(f"Unsupported commercial model: {model_id}")
                
        except Exception as e:
            logger.error(f"Error in batch generation with {model_id}: {e}")
            raise
    
    def list_available_models(self) -> List[Dict[str, Any]]:
        """
        List all available commercial models.
        
        Returns:
            List of model information dictionaries
        """
        models = []
        
        # Add OpenAI models
        if self.openai_client:
            try:
                openai_models = self.openai_client.list_models()
                for model in openai_models:
                    models.append({
                        'id': model['id'],
                        'name': model['id'],
                        'provider': 'openai',
                        'type': 'commercial',
                        'status': 'available',
                        'capabilities': ['text-generation', 'chat'],
                        'created': model.get('created'),
                        'owned_by': model.get('owned_by')
                    })
            except Exception as e:
                logger.error(f"Error listing OpenAI models: {e}")
        
        # Add Anthropic models (static list since Anthropic doesn't have a models endpoint)
        if self.anthropic_client:
            anthropic_models = [
                {
                    'id': 'claude-3-opus-20240229',
                    'name': 'Claude 3 Opus',
                    'provider': 'anthropic',
                    'type': 'commercial',
                    'status': 'available',
                    'capabilities': ['text-generation', 'chat'],
                    'context_length': 200000,
                    'description': 'Most powerful Claude 3 model'
                },
                {
                    'id': 'claude-3-sonnet-20240229',
                    'name': 'Claude 3 Sonnet',
                    'provider': 'anthropic',
                    'type': 'commercial',
                    'status': 'available',
                    'capabilities': ['text-generation', 'chat'],
                    'context_length': 200000,
                    'description': 'Balanced Claude 3 model'
                },
                {
                    'id': 'claude-3-haiku-20240307',
                    'name': 'Claude 3 Haiku',
                    'provider': 'anthropic',
                    'type': 'commercial',
                    'status': 'available',
                    'capabilities': ['text-generation', 'chat'],
                    'context_length': 200000,
                    'description': 'Fastest Claude 3 model'
                }
            ]
            models.extend(anthropic_models)
        
        # Add Gemini models
        if self.gemini_client:
            try:
                gemini_models_info = self.gemini_client.list_models()
                for model_info in gemini_models_info:
                    models.append({
                        'id': model_info['id'],
                        'name': model_info['name'],
                        'provider': 'google',
                        'type': 'commercial',
                        'status': 'available',
                        'capabilities': ['text-generation', 'chat'],
                        'context_length': model_info.get('input_token_limit', 30720),
                        'description': model_info.get('description', 'Google Gemini model')
                    })
            except Exception as e:
                logger.error(f"Error listing Gemini models: {e}")
                # Add default Gemini models
                default_gemini_models = [
                    {
                        'id': 'gemini-pro',
                        'name': 'Gemini Pro',
                        'provider': 'google',
                        'type': 'commercial',
                        'status': 'available',
                        'capabilities': ['text-generation', 'chat'],
                        'context_length': 30720,
                        'description': 'Google Gemini Pro model'
                    },
                    {
                        'id': 'gemini-1.5-flash',
                        'name': 'Gemini 1.5 Flash',
                        'provider': 'google',
                        'type': 'commercial',
                        'status': 'available',
                        'capabilities': ['text-generation', 'chat'],
                        'context_length': 1000000,
                        'description': 'Fast and efficient Gemini model'
                    }
                ]
                models.extend(default_gemini_models)
        
        return models
    
    def test_connections(self) -> Dict[str, Any]:
        """
        Test connections to all available commercial APIs.
        
        Returns:
            Dictionary with connection test results
        """
        results = {
            'timestamp': datetime.utcnow().isoformat(),
            'tests': {}
        }
        
        # Test OpenAI connection
        if self.openai_client:
            try:
                openai_result = self.openai_client.test_connection()
                results['tests']['openai'] = openai_result
            except Exception as e:
                results['tests']['openai'] = {
                    'success': False,
                    'error': str(e)
                }
        else:
            results['tests']['openai'] = {
                'success': False,
                'error': 'OpenAI client not initialized (API key not provided)'
            }
        
        # Test Anthropic connection
        if self.anthropic_client:
            try:
                anthropic_result = self.anthropic_client.test_connection()
                results['tests']['anthropic'] = anthropic_result
            except Exception as e:
                results['tests']['anthropic'] = {
                    'success': False,
                    'error': str(e)
                }
        else:
            results['tests']['anthropic'] = {
                'success': False,
                'error': 'Anthropic client not initialized (API key not provided)'
            }
        
        # Overall success
        results['overall_success'] = any(
            test.get('success', False) for test in results['tests'].values()
        )
        
        return results
    
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific commercial model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Model information dictionary or None if not found
        """
        models = self.list_available_models()
        return next((m for m in models if m['id'] == model_id), None)
    
    def _generate_openai(
        self,
        prompt: str,
        model_id: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Generate text using OpenAI client."""
        if not self.openai_client:
            raise RuntimeError("OpenAI client not initialized")
        
        result = self.openai_client.generate_text(
            prompt=prompt,
            model=model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop
        )
        
        # Standardize response format
        return {
            'text': result['text'],
            'metadata': {
                'model_type': 'commercial',
                'provider': 'openai',
                'model_id': model_id,
                'input_tokens': result['usage']['input_tokens'],
                'output_tokens': result['usage']['output_tokens'],
                'total_tokens': result['usage']['total_tokens'],
                'cost_usd': result['cost_usd'],
                'latency_ms': result['latency_ms'],
                'finish_reason': result['finish_reason'],
                'timestamp': result['timestamp']
            }
        }
    
    def _generate_anthropic(
        self,
        prompt: str,
        model_id: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Generate text using Anthropic client."""
        if not self.anthropic_client:
            raise RuntimeError("Anthropic client not initialized")
        
        result = self.anthropic_client.generate_text(
            prompt=prompt,
            model=model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop_sequences=stop
        )
        
        # Standardize response format
        return {
            'text': result['text'],
            'metadata': {
                'model_type': 'commercial',
                'provider': 'anthropic',
                'model_id': model_id,
                'input_tokens': result['usage']['input_tokens'],
                'output_tokens': result['usage']['output_tokens'],
                'total_tokens': result['usage']['total_tokens'],
                'cost_usd': result['cost_usd'],
                'latency_ms': result['latency_ms'],
                'stop_reason': result['stop_reason'],
                'timestamp': result['timestamp']
            }
        }
    
    def _is_openai_model(self, model_id: str) -> bool:
        """Check if model is an OpenAI model."""
        openai_prefixes = ['gpt-3.5', 'gpt-4', 'text-davinci', 'text-curie', 'text-babbage', 'text-ada']
        return any(model_id.startswith(prefix) for prefix in openai_prefixes)
    
    def _is_anthropic_model(self, model_id: str) -> bool:
        """Check if model is an Anthropic model."""
        anthropic_prefixes = ['claude-', 'claude-instant']
        return any(model_id.startswith(prefix) for prefix in anthropic_prefixes)
    
    def _is_gemini_model(self, model_id: str) -> bool:
        """Check if model is a Gemini model."""
        gemini_prefixes = ['gemini-', 'gemini']
        return any(model_id.startswith(prefix) for prefix in gemini_prefixes)
    
    def _generate_gemini(
        self,
        prompt: str,
        model_id: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Generate text using Gemini client."""
        if not self.gemini_client:
            raise RuntimeError("Gemini client not initialized")
        
        result = self.gemini_client.generate_text(
            prompt=prompt,
            model=model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop_sequences=stop
        )
        
        # Standardize response format
        return {
            'text': result['text'],
            'metadata': {
                'model_type': 'commercial',
                'provider': 'google',
                'model_id': model_id,
                'input_tokens': result['usage']['input_tokens'],
                'output_tokens': result['usage']['output_tokens'],
                'total_tokens': result['usage']['total_tokens'],
                'cost_usd': result['cost_usd'],
                'latency_ms': result['latency_ms'],
                'finish_reason': result['finish_reason'],
                'timestamp': result['timestamp']
            }
        }