"""
Flask application factory and main app configuration.
"""
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import os
from typing import Dict, Any

from .blueprints.health import health_bp
from .blueprints.models import models_bp
from .blueprints.generate import generate_bp
from .blueprints.evaluate import evaluate_bp
from .blueprints.commercial import commercial_bp
from .blueprints.cost_tracking import cost_tracking_bp
from .blueprints.feedback import feedback_bp
from .blueprints.monitoring import monitoring_bp
from .blueprints.auth import auth_bp
from .blueprints.upload import upload_bp
from .middleware.logging_middleware import setup_logging_middleware
from .middleware.validation_middleware import setup_validation_middleware
from .middleware.security_middleware import setup_security_middleware
from .middleware.auth_middleware import setup_auth_middleware


def create_app(config_name: str = 'development') -> Flask:
    """
    Application factory pattern for creating Flask app.
    
    Args:
        config_name: Configuration environment name
        
    Returns:
        Configured Flask application
    """
    app = Flask(__name__)
    
    # Load configuration
    app.config.from_object(f'api.config.{config_name.title()}Config')
    
    # Update API keys from environment (in case they changed after import)
    app.config['OPENAI_API_KEY'] = os.environ.get('OPENAI_API_KEY')
    app.config['ANTHROPIC_API_KEY'] = os.environ.get('ANTHROPIC_API_KEY')
    
    # Validate production configuration
    if config_name == 'production':
        if not app.config['OPENAI_API_KEY']:
            raise ValueError("OPENAI_API_KEY environment variable is required in production")
        if not app.config['ANTHROPIC_API_KEY']:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required in production")
    
    # Initialize CORS
    CORS(app, 
         origins=app.config.get('CORS_ORIGINS', ['http://localhost:3000']),
         methods=['GET', 'POST', 'PUT', 'DELETE'],
         allow_headers=['Content-Type', 'Authorization'])
    
    # Setup middleware
    setup_logging_middleware(app)
    setup_validation_middleware(app)
    setup_security_middleware(app)
    setup_auth_middleware(app)
    
    # Register blueprints
    app.register_blueprint(health_bp, url_prefix='/api/v1')
    app.register_blueprint(models_bp, url_prefix='/api/v1')
    app.register_blueprint(generate_bp, url_prefix='/api/v1')
    app.register_blueprint(evaluate_bp, url_prefix='/api/v1')
    app.register_blueprint(commercial_bp, url_prefix='/api/v1')
    app.register_blueprint(cost_tracking_bp)
    app.register_blueprint(feedback_bp)
    app.register_blueprint(monitoring_bp, url_prefix='/api/v1/monitoring')
    app.register_blueprint(auth_bp, url_prefix='/api/v1/auth')
    app.register_blueprint(upload_bp, url_prefix='/api/v1')
    
    # Setup error handlers
    setup_error_handlers(app)
    
    # Setup logging
    setup_logging(app)
    
    return app


def setup_error_handlers(app: Flask) -> None:
    """Setup global error handlers for the Flask app."""
    
    @app.errorhandler(ValueError)
    def handle_validation_error(error):
        return jsonify({
            'success': False,
            'error': 'validation_error',
            'message': str(error),
            'timestamp': datetime.utcnow().isoformat()
        }), 400
    
    # Import ValidationError here to avoid circular imports
    from utils.exceptions import ValidationError
    
    @app.errorhandler(ValidationError)
    def handle_custom_validation_error(error):
        return jsonify({
            'success': False,
            'error': 'validation_error',
            'message': str(error),
            'timestamp': datetime.utcnow().isoformat()
        }), 400
    
    @app.errorhandler(RuntimeError)
    def handle_model_error(error):
        return jsonify({
            'error': 'model_error',
            'message': str(error),
            'timestamp': datetime.utcnow().isoformat()
        }), 500
    
    @app.errorhandler(404)
    def handle_not_found(error):
        return jsonify({
            'error': 'not_found',
            'message': 'The requested resource was not found',
            'timestamp': datetime.utcnow().isoformat()
        }), 404
    
    @app.errorhandler(500)
    def handle_internal_error(error):
        app.logger.error(f'Internal server error: {error}')
        return jsonify({
            'error': 'internal_server_error',
            'message': 'An internal server error occurred',
            'timestamp': datetime.utcnow().isoformat()
        }), 500


def setup_logging(app: Flask) -> None:
    """Setup application logging configuration."""
    if not app.debug and not app.testing:
        # Production logging setup
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(levelname)s %(name)s %(message)s'
        )
        
        # File handler for errors
        if not os.path.exists('logs'):
            os.mkdir('logs')
        
        file_handler = logging.FileHandler('logs/api.log')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
        ))
        app.logger.addHandler(file_handler)
        app.logger.setLevel(logging.INFO)
        app.logger.info('API Gateway startup')


if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5000, debug=True)