"""
Authentication API endpoints.
"""
from flask import Blueprint, request, jsonify, current_app
from datetime import datetime
import logging

from ..services.auth_service import AuthService
from ..models.user import UserRole
from ..middleware.auth_middleware import require_auth, get_current_user
from utils.exceptions import ValidationError


logger = logging.getLogger(__name__)
auth_bp = Blueprint('auth', __name__)


@auth_bp.route('/register', methods=['POST'])
def register():
    """
    Register a new user account.
    
    Expected JSON payload:
    {
        "username": "string",
        "email": "string", 
        "password": "string",
        "role": "viewer|developer|researcher|admin" (optional, default: viewer)
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'validation_error',
                'message': 'JSON payload required'
            }), 400
        
        # Extract and validate required fields
        username = data.get('username', '').strip()
        email = data.get('email', '').strip()
        password = data.get('password', '')
        role_str = data.get('role', 'viewer').lower()
        
        if not all([username, email, password]):
            return jsonify({
                'success': False,
                'error': 'validation_error',
                'message': 'Username, email, and password are required'
            }), 400
        
        # Validate role
        try:
            role = UserRole(role_str)
        except ValueError:
            return jsonify({
                'success': False,
                'error': 'validation_error',
                'message': f'Invalid role. Must be one of: {[r.value for r in UserRole]}'
            }), 400
        
        # Only admins can create admin accounts
        current_user = get_current_user()
        if role == UserRole.ADMIN and (not current_user or current_user.get('role') != 'admin'):
            return jsonify({
                'success': False,
                'error': 'permission_denied',
                'message': 'Only administrators can create admin accounts'
            }), 403
        
        # Register user
        auth_service = AuthService()
        result = auth_service.register_user(username, email, password, role)
        
        if result['success']:
            logger.info(f"User registered: {username} ({email}) with role {role.value}")
            return jsonify(result), 201
        else:
            return jsonify(result), 400
            
    except Exception as e:
        logger.error(f"Registration error: {e}")
        return jsonify({
            'success': False,
            'error': 'registration_failed',
            'message': 'Registration failed due to server error'
        }), 500


@auth_bp.route('/login', methods=['POST'])
def login():
    """
    Authenticate user and return JWT tokens.
    
    Expected JSON payload:
    {
        "username": "string",  // Can be username or email
        "password": "string"
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'validation_error',
                'message': 'JSON payload required'
            }), 400
        
        username = data.get('username', '').strip()
        password = data.get('password', '')
        
        if not username or not password:
            return jsonify({
                'success': False,
                'error': 'validation_error',
                'message': 'Username and password are required'
            }), 400
        
        # Authenticate user
        auth_service = AuthService()
        result = auth_service.authenticate_user(username, password)
        
        if result['success']:
            logger.info(f"User logged in: {username}")
            return jsonify(result), 200
        else:
            logger.warning(f"Failed login attempt: {username} from {request.remote_addr}")
            return jsonify(result), 401
            
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({
            'success': False,
            'error': 'authentication_failed',
            'message': 'Authentication failed due to server error'
        }), 500


@auth_bp.route('/refresh', methods=['POST'])
def refresh_token():
    """
    Refresh access token using refresh token.
    
    Expected JSON payload:
    {
        "refresh_token": "string"
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'validation_error',
                'message': 'JSON payload required'
            }), 400
        
        refresh_token = data.get('refresh_token')
        if not refresh_token:
            return jsonify({
                'success': False,
                'error': 'validation_error',
                'message': 'Refresh token is required'
            }), 400
        
        # Refresh token
        auth_service = AuthService()
        result = auth_service.refresh_token(refresh_token)
        
        if result['success']:
            return jsonify(result), 200
        else:
            return jsonify(result), 401
            
    except Exception as e:
        logger.error(f"Token refresh error: {e}")
        return jsonify({
            'success': False,
            'error': 'token_refresh_failed',
            'message': 'Token refresh failed due to server error'
        }), 500


@auth_bp.route('/logout', methods=['POST'])
@require_auth()
def logout():
    """
    Logout user (invalidate tokens).
    
    Note: In a production system, you would add the token to a blacklist.
    """
    try:
        current_user = get_current_user()
        logger.info(f"User logged out: {current_user.get('username')}")
        
        return jsonify({
            'success': True,
            'message': 'Logged out successfully'
        }), 200
        
    except Exception as e:
        logger.error(f"Logout error: {e}")
        return jsonify({
            'success': False,
            'error': 'logout_failed',
            'message': 'Logout failed due to server error'
        }), 500


@auth_bp.route('/profile', methods=['GET'])
@require_auth()
def get_profile():
    """Get current user profile information."""
    try:
        current_user = get_current_user()
        
        # Get additional user details from database
        auth_service = AuthService()
        permissions = auth_service.get_user_permissions(current_user['user_id'])
        
        profile = {
            **current_user,
            'permissions': permissions,
            'last_accessed': datetime.utcnow().isoformat()
        }
        
        return jsonify({
            'success': True,
            'profile': profile
        }), 200
        
    except Exception as e:
        logger.error(f"Profile retrieval error: {e}")
        return jsonify({
            'success': False,
            'error': 'profile_retrieval_failed',
            'message': 'Failed to retrieve profile'
        }), 500


@auth_bp.route('/api-keys', methods=['GET'])
@require_auth(['api_key:read'])
def list_api_keys():
    """List user's API keys."""
    try:
        current_user = get_current_user()
        auth_service = AuthService()
        
        # Get user's API keys (implementation depends on repository)
        # For now, return empty list
        return jsonify({
            'success': True,
            'api_keys': [],
            'message': 'API key listing not yet implemented'
        }), 200
        
    except Exception as e:
        logger.error(f"API key listing error: {e}")
        return jsonify({
            'success': False,
            'error': 'api_key_listing_failed',
            'message': 'Failed to list API keys'
        }), 500


@auth_bp.route('/api-keys', methods=['POST'])
@require_auth(['api_key:create'])
def create_api_key():
    """
    Create new API key for current user.
    
    Expected JSON payload:
    {
        "name": "string",
        "permissions": ["permission1", "permission2"],
        "expires_days": 30 (optional)
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'validation_error',
                'message': 'JSON payload required'
            }), 400
        
        name = data.get('name', '').strip()
        permissions = data.get('permissions', [])
        expires_days = data.get('expires_days')
        
        if not name:
            return jsonify({
                'success': False,
                'error': 'validation_error',
                'message': 'API key name is required'
            }), 400
        
        if not isinstance(permissions, list):
            return jsonify({
                'success': False,
                'error': 'validation_error',
                'message': 'Permissions must be a list'
            }), 400
        
        current_user = get_current_user()
        
        # Validate that user has the permissions they're trying to grant
        user_permissions = AuthService().get_user_permissions(current_user['user_id'])
        invalid_permissions = [p for p in permissions if p not in user_permissions]
        
        if invalid_permissions:
            return jsonify({
                'success': False,
                'error': 'permission_denied',
                'message': f'You do not have these permissions: {", ".join(invalid_permissions)}'
            }), 403
        
        # Create API key
        auth_service = AuthService()
        result = auth_service.create_api_key(
            current_user['user_id'],
            name,
            permissions,
            expires_days
        )
        
        if result['success']:
            logger.info(f"API key created: {name} for user {current_user['username']}")
            return jsonify(result), 201
        else:
            return jsonify(result), 400
            
    except Exception as e:
        logger.error(f"API key creation error: {e}")
        return jsonify({
            'success': False,
            'error': 'api_key_creation_failed',
            'message': 'Failed to create API key'
        }), 500


@auth_bp.route('/api-keys/<api_key_id>', methods=['DELETE'])
@require_auth(['api_key:delete'])
def revoke_api_key(api_key_id: str):
    """Revoke API key."""
    try:
        current_user = get_current_user()
        auth_service = AuthService()
        
        result = auth_service.revoke_api_key(current_user['user_id'], api_key_id)
        
        if result['success']:
            logger.info(f"API key revoked: {api_key_id} by user {current_user['username']}")
            return jsonify(result), 200
        else:
            return jsonify(result), 400
            
    except Exception as e:
        logger.error(f"API key revocation error: {e}")
        return jsonify({
            'success': False,
            'error': 'api_key_revocation_failed',
            'message': 'Failed to revoke API key'
        }), 500


@auth_bp.route('/validate', methods=['GET'])
@require_auth()
def validate_token():
    """Validate current authentication token."""
    try:
        current_user = get_current_user()
        
        return jsonify({
            'success': True,
            'valid': True,
            'user': current_user,
            'message': 'Token is valid'
        }), 200
        
    except Exception as e:
        logger.error(f"Token validation error: {e}")
        return jsonify({
            'success': False,
            'error': 'token_validation_failed',
            'message': 'Token validation failed'
        }), 500