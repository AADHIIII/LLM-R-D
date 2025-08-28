"""
Text generation endpoints.
"""
from flask import Blueprint, jsonify, request, current_app
from typing import Dict, Any, Optional
import time
from datetime import datetime
import os

from ..middleware.validation_middleware import validate_json_schema
from ..middleware.auth_middleware import require_auth, optional_auth
from ..services.model_loader import ModelLoader
from ..services.text_generator import TextGenerator

generate_bp = Blueprint('generate', __name__)

# Global instances - will be initialized with API keys from app config
model_loader = ModelLoader()
text_generator = None


@generate_bp.route('/generate', methods=['POST'])
@require_auth(['model:read'])
@validate_json_schema({
    'prompt': str,
    'model_id': str
})
def generate_text() -> Dict[str, Any]:
    """
    Generate text using specified model.
    
    Expected JSON payload:
    {
        "prompt": "Your prompt here",
        "model_id": "gpt-4",
        "max_tokens": 100,
        "temperature": 0.7,
        "top_p": 1.0,
        "stop": null
    }
    
    Returns:
        JSON response with generated text
    """
    try:
        data = request.get_json()
        
        # Extract parameters
        prompt = data['prompt']
        model_id = data['model_id']
        max_tokens = data.get('max_tokens', 100)
        temperature = data.get('temperature', 0.7)
        top_p = data.get('top_p', 1.0)
        stop = data.get('stop', None)
        
        # Validate parameters
        if not prompt.strip():
            return jsonify({
                'error': 'validation_error',
                'message': 'Prompt cannot be empty',
                'timestamp': datetime.utcnow().isoformat()
            }), 400
        
        if max_tokens <= 0 or max_tokens > 2048:
            return jsonify({
                'error': 'validation_error',
                'message': 'max_tokens must be between 1 and 2048',
                'timestamp': datetime.utcnow().isoformat()
            }), 400
        
        if not 0.0 <= temperature <= 2.0:
            return jsonify({
                'error': 'validation_error',
                'message': 'temperature must be between 0.0 and 2.0',
                'timestamp': datetime.utcnow().isoformat()
            }), 400
        
        # Record start time
        start_time = time.time()
        
        # Initialize text generator if not already done
        global text_generator
        if text_generator is None:
            from ..services.text_generator import TextGenerator
            text_generator = TextGenerator(
                openai_api_key=current_app.config.get('OPENAI_API_KEY'),
                anthropic_api_key=current_app.config.get('ANTHROPIC_API_KEY')
            )
        
        # Generate text
        result = text_generator.generate(
            prompt=prompt,
            model_id=model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop
        )
        
        # Calculate metrics
        end_time = time.time()
        latency_ms = int((end_time - start_time) * 1000)
        
        # Estimate token count (rough approximation)
        input_tokens = len(prompt.split()) * 1.3  # Rough token estimation
        output_tokens = len(result['text'].split()) * 1.3
        
        return jsonify({
            'text': result['text'],
            'model_id': model_id,
            'prompt': prompt,
            'parameters': {
                'max_tokens': max_tokens,
                'temperature': temperature,
                'top_p': top_p,
                'stop': stop
            },
            'metrics': {
                'latency_ms': latency_ms,
                'input_tokens': int(input_tokens),
                'output_tokens': int(output_tokens),
                'total_tokens': int(input_tokens + output_tokens)
            },
            'metadata': result.get('metadata', {}),
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except ValueError as e:
        return jsonify({
            'error': 'validation_error',
            'message': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 400
        
    except Exception as e:
        current_app.logger.error(f"Error generating text: {str(e)}")
        return jsonify({
            'error': 'generation_error',
            'message': 'Failed to generate text',
            'details': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500


@generate_bp.route('/generate/batch', methods=['POST'])
@require_auth(['model:read'])
@validate_json_schema({
    'prompts': list,
    'model_id': str
})
def generate_batch() -> Dict[str, Any]:
    """
    Generate text for multiple prompts using specified model.
    
    Expected JSON payload:
    {
        "prompts": ["prompt1", "prompt2", ...],
        "model_id": "gpt-4",
        "max_tokens": 100,
        "temperature": 0.7
    }
    
    Returns:
        JSON response with generated texts
    """
    try:
        data = request.get_json()
        
        prompts = data['prompts']
        model_id = data['model_id']
        max_tokens = data.get('max_tokens', 100)
        temperature = data.get('temperature', 0.7)
        top_p = data.get('top_p', 1.0)
        
        # Validate batch size
        if len(prompts) > 10:
            return jsonify({
                'error': 'validation_error',
                'message': 'Batch size cannot exceed 10 prompts',
                'timestamp': datetime.utcnow().isoformat()
            }), 400
        
        # Validate prompts
        for i, prompt in enumerate(prompts):
            if not isinstance(prompt, str) or not prompt.strip():
                return jsonify({
                    'error': 'validation_error',
                    'message': f'Prompt at index {i} is invalid',
                    'timestamp': datetime.utcnow().isoformat()
                }), 400
        
        # Generate text for each prompt
        results = []
        start_time = time.time()
        
        # Initialize text generator if not already done
        global text_generator
        if text_generator is None:
            from ..services.text_generator import TextGenerator
            text_generator = TextGenerator(
                openai_api_key=current_app.config.get('OPENAI_API_KEY'),
                anthropic_api_key=current_app.config.get('ANTHROPIC_API_KEY')
            )
        
        for i, prompt in enumerate(prompts):
            try:
                result = text_generator.generate(
                    prompt=prompt,
                    model_id=model_id,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p
                )
                
                results.append({
                    'index': i,
                    'prompt': prompt,
                    'text': result['text'],
                    'success': True,
                    'metadata': result.get('metadata', {})
                })
                
            except Exception as e:
                results.append({
                    'index': i,
                    'prompt': prompt,
                    'text': None,
                    'success': False,
                    'error': str(e)
                })
        
        # Calculate overall metrics
        end_time = time.time()
        total_latency_ms = int((end_time - start_time) * 1000)
        successful_generations = sum(1 for r in results if r['success'])
        
        return jsonify({
            'results': results,
            'model_id': model_id,
            'parameters': {
                'max_tokens': max_tokens,
                'temperature': temperature,
                'top_p': top_p
            },
            'metrics': {
                'total_latency_ms': total_latency_ms,
                'successful_generations': successful_generations,
                'failed_generations': len(prompts) - successful_generations,
                'average_latency_ms': total_latency_ms // len(prompts) if prompts else 0
            },
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        current_app.logger.error(f"Error in batch generation: {str(e)}")
        return jsonify({
            'error': 'batch_generation_error',
            'message': 'Failed to process batch generation',
            'details': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500