"""
Request validation middleware.
"""
from flask import Flask, request, jsonify, g
from functools import wraps
from typing import Any, Callable, Dict
import json
import logging

from utils.security import InputValidator, audit_logger

logger = logging.getLogger(__name__)

# Custom validation error class
class ValidationError(ValueError):
    """Custom validation error."""
    pass


def setup_validation_middleware(app: Flask) -> None:
    """
    Setup request validation middleware.
    
    Args:
        app: Flask application instance
    """
    
    @app.before_request
    def validate_content_type() -> None:
        """Validate content type for POST/PUT requests."""
        if request.method in ['POST', 'PUT']:
            if not request.is_json and request.content_length and request.content_length > 0:
                return jsonify({
                    'error': 'validation_error',
                    'message': 'Content-Type must be application/json for POST/PUT requests'
                }), 400
    
    @app.before_request
    def validate_content_length() -> None:
        """Validate request content length."""
        max_length = app.config.get('MAX_CONTENT_LENGTH', 16 * 1024 * 1024)
        if request.content_length and request.content_length > max_length:
            return jsonify({
                'error': 'validation_error',
                'message': f'Request too large. Maximum size: {max_length} bytes'
            }), 413
    
    @app.before_request
    def security_validation() -> None:
        """Perform security validation on requests."""
        # Skip validation for health checks and static files
        if request.endpoint in ['health.health_check', 'static']:
            return
        
        # Check for SQL injection in query parameters
        for param_name, param_value in request.args.items():
            if InputValidator.check_sql_injection(param_value):
                audit_logger.log_security_event(
                    event_type="SQL_INJECTION_ATTEMPT",
                    user_id=getattr(g, 'current_user', {}).get('user_id'),
                    details=f"Parameter: {param_name}",
                    severity="WARNING"
                )
                return jsonify({
                    'error': 'security_violation',
                    'message': 'Potentially malicious input detected'
                }), 400
        
        # Validate JSON payload if present
        if request.is_json:
            try:
                data = request.get_json()
                if data:
                    for key, value in data.items():
                        if isinstance(value, str) and InputValidator.check_sql_injection(value):
                            audit_logger.log_security_event(
                                event_type="SQL_INJECTION_ATTEMPT",
                                user_id=getattr(g, 'current_user', {}).get('user_id'),
                                details=f"JSON field: {key}",
                                severity="WARNING"
                            )
                            return jsonify({
                                'error': 'security_violation',
                                'message': 'Potentially malicious input detected'
                            }), 400
            except Exception as e:
                logger.warning(f"Error validating JSON payload: {e}")
                return jsonify({
                    'error': 'validation_error',
                    'message': 'Invalid JSON payload'
                }), 400


def validate_json_schema(schema: Dict[str, Any], optional_fields: list = None) -> Callable:
    """
    Decorator to validate JSON request against a schema with security checks.
    
    Args:
        schema: JSON schema to validate against (required fields)
        optional_fields: List of optional field names
        
    Returns:
        Decorator function
    """
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not request.is_json:
                raise ValidationError("Request must be JSON")
            
            data = request.get_json()
            if not data:
                raise ValidationError("Request body cannot be empty")
            
            try:
                # Use secure validation from InputValidator
                required_fields = list(schema.keys())
                validated_data = InputValidator.validate_json_structure(
                    data, required_fields, optional_fields
                )
                
                # Type validation
                for field, field_type in schema.items():
                    if field in validated_data:
                        if not isinstance(validated_data[field], field_type):
                            raise ValidationError(
                                f"Field '{field}' must be of type {field_type.__name__}"
                            )
                
                # Store validated data in request context
                g.validated_data = validated_data
                
            except ValueError as e:
                # Log security violation if it's a security-related error
                if "injection" in str(e).lower():
                    audit_logger.log_security_event(
                        event_type="VALIDATION_SECURITY_VIOLATION",
                        user_id=getattr(g, 'current_user', {}).get('user_id'),
                        details=str(e),
                        severity="WARNING"
                    )
                raise ValidationError(str(e))
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator


def validate_query_params(required_params: list = None, 
                         optional_params: list = None) -> Callable:
    """
    Decorator to validate query parameters.
    
    Args:
        required_params: List of required query parameter names
        optional_params: List of optional query parameter names
        
    Returns:
        Decorator function
    """
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def decorated_function(*args, **kwargs):
            req_params = required_params or []
            opt_params = optional_params or []
            
            # Check required parameters
            for param in req_params:
                if param not in request.args:
                    raise ValidationError(f"Missing required query parameter: {param}")
            
            # Check for unexpected parameters
            allowed_params = set(req_params + opt_params)
            provided_params = set(request.args.keys())
            unexpected_params = provided_params - allowed_params
            
            if unexpected_params:
                raise ValidationError(
                    f"Unexpected query parameters: {', '.join(unexpected_params)}"
                )
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator