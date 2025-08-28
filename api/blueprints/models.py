"""
Model management endpoints.
"""
from flask import Blueprint, jsonify, current_app
from typing import Dict, Any, List
import os
import glob
from datetime import datetime

models_bp = Blueprint('models', __name__)


def get_available_models() -> List[Dict[str, Any]]:
    """
    Get list of available models (both fine-tuned and commercial).
    
    Returns:
        List of model information dictionaries
    """
    models = []
    
    # Add commercial models
    commercial_models = [
        {
            'id': 'gpt-4',
            'name': 'GPT-4',
            'type': 'commercial',
            'provider': 'openai',
            'description': 'OpenAI GPT-4 model',
            'status': 'available' if current_app.config.get('OPENAI_API_KEY') else 'unavailable',
            'capabilities': ['text-generation', 'chat'],
            'context_length': 8192
        },
        {
            'id': 'gpt-3.5-turbo',
            'name': 'GPT-3.5 Turbo',
            'type': 'commercial',
            'provider': 'openai',
            'description': 'OpenAI GPT-3.5 Turbo model',
            'status': 'available' if current_app.config.get('OPENAI_API_KEY') else 'unavailable',
            'capabilities': ['text-generation', 'chat'],
            'context_length': 4096
        },
        {
            'id': 'claude-3-sonnet',
            'name': 'Claude 3 Sonnet',
            'type': 'commercial',
            'provider': 'anthropic',
            'description': 'Anthropic Claude 3 Sonnet model',
            'status': 'available' if current_app.config.get('ANTHROPIC_API_KEY') else 'unavailable',
            'capabilities': ['text-generation', 'chat'],
            'context_length': 200000
        }
    ]
    
    models.extend(commercial_models)
    
    # Add fine-tuned models (scan models directory)
    models_dir = os.path.join(os.getcwd(), 'models')
    if os.path.exists(models_dir):
        for model_path in glob.glob(os.path.join(models_dir, '*')):
            if os.path.isdir(model_path):
                model_name = os.path.basename(model_path)
                
                # Check if it's a valid model directory (has config.json)
                config_path = os.path.join(model_path, 'config.json')
                if os.path.exists(config_path):
                    models.append({
                        'id': model_name,
                        'name': model_name,
                        'type': 'fine-tuned',
                        'provider': 'local',
                        'description': f'Fine-tuned model: {model_name}',
                        'status': 'available',
                        'capabilities': ['text-generation'],
                        'path': model_path,
                        'created_at': datetime.fromtimestamp(
                            os.path.getctime(model_path)
                        ).isoformat()
                    })
    
    return models


@models_bp.route('/models', methods=['GET'])
def list_models() -> Dict[str, Any]:
    """
    List available models.
    
    Returns:
        JSON response with available models
    """
    try:
        models = get_available_models()
        
        return jsonify({
            'models': models,
            'count': len(models),
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        current_app.logger.error(f"Error listing models: {str(e)}")
        return jsonify({
            'error': 'model_listing_error',
            'message': 'Failed to list available models',
            'timestamp': datetime.utcnow().isoformat()
        }), 500


@models_bp.route('/models/<model_id>', methods=['GET'])
def get_model_info(model_id: str) -> Dict[str, Any]:
    """
    Get information about a specific model.
    
    Args:
        model_id: Model identifier
        
    Returns:
        JSON response with model information
    """
    try:
        models = get_available_models()
        
        # Find the requested model
        model = next((m for m in models if m['id'] == model_id), None)
        
        if not model:
            return jsonify({
                'error': 'model_not_found',
                'message': f'Model {model_id} not found',
                'available_models': [m['id'] for m in models],
                'timestamp': datetime.utcnow().isoformat()
            }), 404
        
        # Add additional details for fine-tuned models
        if model['type'] == 'fine-tuned' and 'path' in model:
            model_path = model['path']
            
            # Try to read additional metadata
            metadata_path = os.path.join(model_path, 'training_metadata.json')
            if os.path.exists(metadata_path):
                import json
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    model['training_metadata'] = metadata
                except Exception:
                    pass
            
            # Get model size
            try:
                total_size = sum(
                    os.path.getsize(os.path.join(dirpath, filename))
                    for dirpath, dirnames, filenames in os.walk(model_path)
                    for filename in filenames
                )
                model['size_bytes'] = total_size
                model['size_mb'] = round(total_size / (1024 * 1024), 2)
            except Exception:
                pass
        
        return jsonify({
            'model': model,
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        current_app.logger.error(f"Error getting model info for {model_id}: {str(e)}")
        return jsonify({
            'error': 'model_info_error',
            'message': f'Failed to get information for model {model_id}',
            'timestamp': datetime.utcnow().isoformat()
        }), 500