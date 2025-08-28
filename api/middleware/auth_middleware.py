"""
Authentication middleware for API endpoints.
"""
from functools import wraps
from flask import request, jsonify, g, current_app
from typing import List, Optional, Callable, Any
import logging

from ..services.jwt_service import JWTService
from ..services.auth_service import AuthService


logger = logging.getLogger(__name__)


def require_auth(permissions: Optional[List[str]] = None):
    """
    Decorator to require authentication for API endpoints.
    
    Args:
        permissions: List of required permissions (optional)
        
    Returns:
        Decorator function
    """
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Skip authentication in testing mode if configured
            if current_app.config.get('TESTING') and current_app.config.get('SKIP_AUTH', False):
                g.current_user = {'user_id': 'test-user', 'role': 'admin'}
                return f(*args, **kwargs)
            
            auth_result = authenticate_request()
            if not auth_result['success']:
                return jsonify({
                    'success': False,
                    'error': 'authentication_required',
                    'message': auth_result['error']
                }), 401
            
            # Set current user in Flask g object
            g.current_user = auth_result['user']
            g.auth_method = auth_result['method']
            
            # Check permissions if specified
            if permissions:
                if not check_permissions(auth_result['user'], permissions):
                    return jsonify({
                        'success': False,
                        'error': 'insufficient_permissions',
                        'message': 'You do not have permission to access this resource'
                    }), 403
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator


def require_api_key(permissions: Optional[List[str]] = None):
    """
    Decorator to require API key authentication for API endpoints.
    
    Args:
        permissions: List of required permissions (optional)
        
    Returns:
        Decorator function
    """
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def decorated_function(*args, **kwargs):
            auth_result = authenticate_api_key()
            if not auth_result['success']:
                return jsonify({
                    'success': False,
                    'error': 'api_key_required',
                    'message': auth_result['error']
                }), 401
            
            # Set current user and API key info
            g.current_user = auth_result['user']
            g.api_key = auth_result['api_key']
            g.auth_method = 'api_key'
            
            # Check API key permissions
            api_key_permissions = auth_result.get('permissions', [])
            if permissions:
                missing_permissions = [p for p in permissions if p not in api_key_permissions]
                if missing_permissions:
                    return jsonify({
                        'success': False,
                        'error': 'insufficient_api_key_permissions',
                        'message': f'API key missing permissions: {", ".join(missing_permissions)}'
                    }), 403
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator


def optional_auth():
    """
    Decorator for endpoints that work with or without authentication.
    Sets g.current_user if authenticated, None otherwise.
    """
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def decorated_function(*args, **kwargs):
            auth_result = authenticate_request()
            if auth_result['success']:
                g.current_user = auth_result['user']
                g.auth_method = auth_result['method']
            else:
                g.current_user = None
                g.auth_method = None
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator


def authenticate_request() -> dict:
    """
    Authenticate request using JWT token or API key.
    
    Returns:
        Dictionary with authentication result
    """
    # Try JWT authentication first
    auth_header = request.headers.get('Authorization')
    if auth_header:
        token = JWTService.extract_token_from_header(auth_header)
        if token:
            user_info = JWTService.get_user_from_token(token)
            if user_info:
                return {
                    'success': True,
                    'user': user_info,
                    'method': 'jwt'
                }
    
    # Try API key authentication
    api_key = request.headers.get('X-API-Key')
    if api_key:
        auth_service = AuthService()
        result = auth_service.authenticate_api_key(api_key)
        if result['success']:
            return {
                'success': True,
                'user': result['user'],
                'api_key': result['api_key'],
                'permissions': result['permissions'],
                'method': 'api_key'
            }
    
    return {
        'success': False,
        'error': 'No valid authentication provided'
    }


def authenticate_api_key() -> dict:
    """
    Authenticate request using API key only.
    
    Returns:
        Dictionary with authentication result
    """
    api_key = request.headers.get('X-API-Key')
    if not api_key:
        return {
            'success': False,
            'error': 'API key required in X-API-Key header'
        }
    
    auth_service = AuthService()
    return auth_service.authenticate_api_key(api_key)


def check_permissions(user: dict, required_permissions: List[str]) -> bool:
    """
    Check if user has required permissions.
    
    Args:
        user: User information dictionary
        required_permissions: List of required permission strings
        
    Returns:
        True if user has all required permissions
    """
    user_role = user.get('role')
    if not user_role:
        return False
    
    # Admin has all permissions
    if user_role == 'admin':
        return True
    
    # Get user permissions based on role
    role_permissions = {
        'researcher': [
            'model:create', 'model:read', 'model:update',
            'experiment:create', 'experiment:read', 'experiment:update', 'experiment:delete',
            'api_key:create', 'api_key:read', 'api_key:update'
        ],
        'developer': [
            'model:read', 'experiment:read', 'experiment:create',
            'api_key:create', 'api_key:read'
        ],
        'viewer': [
            'model:read', 'experiment:read'
        ]
    }
    
    user_permissions = role_permissions.get(user_role, [])
    
    # Check if user has all required permissions
    return all(permission in user_permissions for permission in required_permissions)


def get_current_user() -> Optional[dict]:
    """
    Get current authenticated user from Flask g object.
    
    Returns:
        User information dictionary or None if not authenticated
    """
    return getattr(g, 'current_user', None)


def get_auth_method() -> Optional[str]:
    """
    Get authentication method used for current request.
    
    Returns:
        Authentication method ('jwt' or 'api_key') or None
    """
    return getattr(g, 'auth_method', None)


def setup_auth_middleware(app):
    """
    Setup authentication middleware for Flask app.
    
    Args:
        app: Flask application instance
    """
    
    @app.before_request
    def log_auth_attempts():
        """Log authentication attempts for security monitoring."""
        if request.endpoint and not request.endpoint.startswith('static'):
            auth_header = request.headers.get('Authorization')
            api_key_header = request.headers.get('X-API-Key')
            
            if auth_header or api_key_header:
                logger.info(
                    f"Auth attempt: {request.method} {request.path} "
                    f"from {request.remote_addr} "
                    f"auth_type={'jwt' if auth_header else 'api_key'}"
                )
    
    @app.after_request
    def log_auth_success(response):
        """Log successful authentications."""
        if hasattr(g, 'current_user') and g.current_user:
            logger.info(
                f"Auth success: user_id={g.current_user.get('user_id')} "
                f"method={getattr(g, 'auth_method', 'unknown')} "
                f"endpoint={request.endpoint}"
            )
        return response