"""
Commercial API management endpoints.
"""
from flask import Blueprint, jsonify, current_app
from typing import Dict, Any
from datetime import datetime

from ..services.commercial_api_service import CommercialAPIService

commercial_bp = Blueprint('commercial', __name__)


@commercial_bp.route('/commercial/test', methods=['GET'])
def test_commercial_apis() -> Dict[str, Any]:
    """
    Test connections to commercial APIs.
    
    Returns:
        JSON response with connection test results
    """
    try:
        # Initialize commercial API service
        service = CommercialAPIService(
            openai_api_key=current_app.config.get('OPENAI_API_KEY'),
            anthropic_api_key=current_app.config.get('ANTHROPIC_API_KEY')
        )
        
        # Test connections
        results = service.test_connections()
        
        return jsonify(results)
        
    except Exception as e:
        current_app.logger.error(f"Error testing commercial APIs: {str(e)}")
        return jsonify({
            'error': 'test_error',
            'message': 'Failed to test commercial API connections',
            'details': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500


@commercial_bp.route('/commercial/models', methods=['GET'])
def list_commercial_models() -> Dict[str, Any]:
    """
    List available commercial models.
    
    Returns:
        JSON response with commercial models
    """
    try:
        # Initialize commercial API service
        service = CommercialAPIService(
            openai_api_key=current_app.config.get('OPENAI_API_KEY'),
            anthropic_api_key=current_app.config.get('ANTHROPIC_API_KEY')
        )
        
        # Get available models
        models = service.list_available_models()
        
        return jsonify({
            'models': models,
            'count': len(models),
            'providers': list(set(m['provider'] for m in models)),
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        current_app.logger.error(f"Error listing commercial models: {str(e)}")
        return jsonify({
            'error': 'listing_error',
            'message': 'Failed to list commercial models',
            'details': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500


@commercial_bp.route('/commercial/usage', methods=['GET'])
def get_usage_stats() -> Dict[str, Any]:
    """
    Get usage statistics for commercial APIs.
    
    Returns:
        JSON response with usage statistics
    """
    # This would typically connect to a database to get usage stats
    # For now, return a placeholder response
    return jsonify({
        'message': 'Usage statistics endpoint - to be implemented with database integration',
        'timestamp': datetime.utcnow().isoformat()
    })