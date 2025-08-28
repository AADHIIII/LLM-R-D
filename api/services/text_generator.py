"""
Text generation service for both fine-tuned and commercial models.
"""
import logging
from typing import Dict, Any, Optional, List
import time
from datetime import datetime

from .model_loader import ModelLoader
from .commercial_api_service import CommercialAPIService

logger = logging.getLogger(__name__)


class TextGenerator:
    """Service for generating text using various models."""
    
    def __init__(self, openai_api_key: str = None, anthropic_api_key: str = None):
        self.model_loader = ModelLoader()
        self.commercial_api_service = CommercialAPIService(openai_api_key, anthropic_api_key)
    
    def generate(
        self,
        prompt: str,
        model_id: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate text using the specified model.
        
        Args:
            prompt: Input prompt
            model_id: Model identifier
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            stop: Stop sequences
            
        Returns:
            Dictionary with generated text and metadata
        """
        start_time = time.time()
        
        try:
            # Determine model type
            if self._is_commercial_model(model_id):
                result = self.commercial_api_service.generate_text(
                    prompt, model_id, max_tokens, temperature, top_p, stop
                )
            else:
                result = self._generate_fine_tuned(
                    prompt, model_id, max_tokens, temperature, top_p, stop
                )
            
            # Add timing information
            result['metadata']['generation_time_ms'] = int((time.time() - start_time) * 1000)
            result['metadata']['timestamp'] = datetime.utcnow().isoformat()
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating text with model {model_id}: {e}")
            raise
    
    def _is_commercial_model(self, model_id: str) -> bool:
        """Check if model is a commercial API model."""
        commercial_models = [
            'gpt-4', 'gpt-3.5-turbo', 'gpt-3.5-turbo-16k',
            'claude-3-sonnet', 'claude-3-haiku', 'claude-3-opus'
        ]
        return model_id in commercial_models
    
    def _generate_fine_tuned(
        self,
        prompt: str,
        model_id: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Generate text using a fine-tuned model."""
        try:
            # Load model and tokenizer
            model, tokenizer = self.model_loader.load_model(model_id)
            
            # Prepare inputs
            inputs = tokenizer.encode(prompt, return_tensors='pt')
            
            # Set pad token if not set
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Generate text
            with model.eval():
                # Import torch here to avoid import errors if not installed
                import torch
                
                # Generate
                outputs = model.generate(
                    inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=temperature > 0,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    early_stopping=True
                )
                
                # Decode generated text
                generated_text = tokenizer.decode(
                    outputs[0][inputs.shape[1]:], 
                    skip_special_tokens=True
                )
                
                # Apply stop sequences if provided
                if stop:
                    for stop_seq in stop:
                        if stop_seq in generated_text:
                            generated_text = generated_text.split(stop_seq)[0]
                            break
                
                return {
                    'text': generated_text.strip(),
                    'metadata': {
                        'model_type': 'fine-tuned',
                        'model_id': model_id,
                        'input_tokens': inputs.shape[1],
                        'output_tokens': outputs.shape[1] - inputs.shape[1],
                        'parameters': {
                            'max_tokens': max_tokens,
                            'temperature': temperature,
                            'top_p': top_p,
                            'stop': stop
                        }
                    }
                }
                
        except ImportError as e:
            raise RuntimeError(f"PyTorch not available for fine-tuned model generation: {e}")
        except Exception as e:
            raise RuntimeError(f"Error generating with fine-tuned model {model_id}: {e}")
    
    def _estimate_openai_cost(self, model_id: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for OpenAI API call."""
        # OpenAI pricing (per 1K tokens)
        pricing = {
            'gpt-4': {'input': 0.03, 'output': 0.06},
            'gpt-3.5-turbo': {'input': 0.0015, 'output': 0.002},
            'gpt-3.5-turbo-16k': {'input': 0.003, 'output': 0.004},
        }
        
        if model_id not in pricing:
            return 0.0
        
        model_pricing = pricing[model_id]
        input_cost = (input_tokens / 1000) * model_pricing['input']
        output_cost = (output_tokens / 1000) * model_pricing['output']
        
        return input_cost + output_cost
    
    def _estimate_anthropic_cost(self, model_id: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for Anthropic API call."""
        # Anthropic pricing (per 1K tokens)
        pricing = {
            'claude-3-opus': {'input': 0.015, 'output': 0.075},
            'claude-3-sonnet': {'input': 0.003, 'output': 0.015},
            'claude-3-haiku': {'input': 0.00025, 'output': 0.00125},
        }
        
        if model_id not in pricing:
            return 0.0
        
        model_pricing = pricing[model_id]
        input_cost = (input_tokens / 1000) * model_pricing['input']
        output_cost = (output_tokens / 1000) * model_pricing['output']
        
        return input_cost + output_cost
    
