"""
Authentication service for user management and authentication.
"""
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import re
from flask import current_app

from ..models.user import User, UserRole, UserStatus, APIKey
from .jwt_service import JWTService
from database.repositories import UserRepository, APIKeyRepository


class AuthService:
    """Service for user authentication and management."""
    
    def __init__(self):
        self.user_repo = UserRepository()
        self.api_key_repo = APIKeyRepository()
        self.jwt_service = JWTService()
    
    def register_user(self, username: str, email: str, password: str, 
                     role: UserRole = UserRole.VIEWER) -> Dict[str, Any]:
        """
        Register a new user.
        
        Args:
            username: Unique username
            email: User email address
            password: Plain text password
            role: User role (default: VIEWER)
            
        Returns:
            Dictionary with success status and user data or error message
        """
        # Validate input
        validation_error = self._validate_user_input(username, email, password)
        if validation_error:
            return {'success': False, 'error': validation_error}
        
        # Check if user already exists
        if self.user_repo.get_by_username(username):
            return {'success': False, 'error': 'Username already exists'}
        
        if self.user_repo.get_by_email(email):
            return {'success': False, 'error': 'Email already registered'}
        
        # Create new user
        user = User(
            username=username,
            email=email,
            role=role,
            status=UserStatus.ACTIVE  # Auto-activate for now
        )
        user.set_password(password)
        
        # Save user
        try:
            saved_user = self.user_repo.create(user)
            return {
                'success': True,
                'user': saved_user.to_dict(),
                'message': 'User registered successfully'
            }
        except Exception as e:
            current_app.logger.error(f"Failed to register user: {e}")
            return {'success': False, 'error': 'Registration failed'}
    
    def authenticate_user(self, username: str, password: str) -> Dict[str, Any]:
        """
        Authenticate user with username/password.
        
        Args:
            username: Username or email
            password: Plain text password
            
        Returns:
            Dictionary with authentication result and tokens
        """
        # Get user by username or email
        user = self.user_repo.get_by_username(username)
        if not user:
            user = self.user_repo.get_by_email(username)
        
        if not user:
            return {'success': False, 'error': 'Invalid credentials'}
        
        # Check if account is locked
        if user.locked_until and user.locked_until > datetime.utcnow():
            return {
                'success': False, 
                'error': f'Account locked until {user.locked_until.isoformat()}'
            }
        
        # Verify password
        if not user.verify_password(password):
            # Increment failed attempts
            user.failed_login_attempts += 1
            
            # Lock account after 5 failed attempts
            if user.failed_login_attempts >= 5:
                user.locked_until = datetime.utcnow() + timedelta(minutes=30)
            
            self.user_repo.update(user)
            return {'success': False, 'error': 'Invalid credentials'}
        
        # Check if user is active
        if not user.is_active():
            return {'success': False, 'error': 'Account is not active'}
        
        # Reset failed attempts and update last login
        user.failed_login_attempts = 0
        user.locked_until = None
        user.last_login = datetime.utcnow()
        self.user_repo.update(user)
        
        # Generate tokens
        tokens = self.jwt_service.generate_tokens(user)
        
        return {
            'success': True,
            'user': user.to_dict(),
            'tokens': tokens,
            'message': 'Authentication successful'
        }
    
    def authenticate_api_key(self, api_key: str) -> Dict[str, Any]:
        """
        Authenticate using API key.
        
        Args:
            api_key: API key string
            
        Returns:
            Dictionary with authentication result
        """
        # Get API key by prefix (first 8 characters)
        key_prefix = api_key[:8] if len(api_key) >= 8 else api_key
        api_key_obj = self.api_key_repo.get_by_prefix(key_prefix)
        
        if not api_key_obj or not api_key_obj.verify_key(api_key):
            return {'success': False, 'error': 'Invalid API key'}
        
        if not api_key_obj.is_valid():
            return {'success': False, 'error': 'API key expired or inactive'}
        
        # Get associated user
        user = self.user_repo.get_by_id(api_key_obj.user_id)
        if not user or not user.is_active():
            return {'success': False, 'error': 'Associated user account is not active'}
        
        # Record API key usage
        api_key_obj.record_usage()
        self.api_key_repo.update(api_key_obj)
        
        return {
            'success': True,
            'user': user.to_dict(),
            'api_key': api_key_obj.to_dict(),
            'permissions': api_key_obj.permissions
        }
    
    def create_api_key(self, user_id: str, name: str, permissions: List[str],
                      expires_days: Optional[int] = None) -> Dict[str, Any]:
        """
        Create new API key for user.
        
        Args:
            user_id: User ID
            name: API key name/description
            permissions: List of permissions for the key
            expires_days: Days until expiration (None for no expiration)
            
        Returns:
            Dictionary with API key data
        """
        user = self.user_repo.get_by_id(user_id)
        if not user:
            return {'success': False, 'error': 'User not found'}
        
        # Generate API key
        key_string = APIKey.generate_key()
        
        api_key = APIKey(
            user_id=user_id,
            name=name,
            permissions=permissions,
            expires_at=datetime.utcnow() + timedelta(days=expires_days) if expires_days else None
        )
        api_key.set_key(key_string)
        
        try:
            saved_key = self.api_key_repo.create(api_key)
            return {
                'success': True,
                'api_key': saved_key.to_dict(),
                'key': key_string,  # Only returned once
                'message': 'API key created successfully'
            }
        except Exception as e:
            current_app.logger.error(f"Failed to create API key: {e}")
            return {'success': False, 'error': 'Failed to create API key'}
    
    def revoke_api_key(self, user_id: str, api_key_id: str) -> Dict[str, Any]:
        """
        Revoke API key.
        
        Args:
            user_id: User ID (for authorization)
            api_key_id: API key ID to revoke
            
        Returns:
            Dictionary with operation result
        """
        api_key = self.api_key_repo.get_by_id(api_key_id)
        if not api_key:
            return {'success': False, 'error': 'API key not found'}
        
        # Check ownership or admin permission
        user = self.user_repo.get_by_id(user_id)
        if not user:
            return {'success': False, 'error': 'User not found'}
        
        if api_key.user_id != user_id and not user.has_permission('api_key:delete'):
            return {'success': False, 'error': 'Permission denied'}
        
        # Deactivate API key
        api_key.is_active = False
        self.api_key_repo.update(api_key)
        
        return {'success': True, 'message': 'API key revoked successfully'}
    
    def refresh_token(self, refresh_token: str) -> Dict[str, Any]:
        """
        Refresh access token using refresh token.
        
        Args:
            refresh_token: Valid refresh token
            
        Returns:
            Dictionary with new tokens or error
        """
        payload = self.jwt_service.verify_token(refresh_token, 'refresh')
        if not payload:
            return {'success': False, 'error': 'Invalid refresh token'}
        
        user = self.user_repo.get_by_id(payload.get('user_id'))
        if not user or not user.is_active():
            return {'success': False, 'error': 'User account not active'}
        
        tokens = self.jwt_service.refresh_access_token(refresh_token, user)
        if not tokens:
            return {'success': False, 'error': 'Failed to refresh token'}
        
        return {
            'success': True,
            'tokens': tokens,
            'message': 'Token refreshed successfully'
        }
    
    def get_user_permissions(self, user_id: str) -> List[str]:
        """
        Get all permissions for user.
        
        Args:
            user_id: User ID
            
        Returns:
            List of permission strings
        """
        user = self.user_repo.get_by_id(user_id)
        if not user:
            return []
        
        # Get role-based permissions
        role_permissions = {
            UserRole.ADMIN: [
                'user:create', 'user:read', 'user:update', 'user:delete',
                'model:create', 'model:read', 'model:update', 'model:delete',
                'experiment:create', 'experiment:read', 'experiment:update', 'experiment:delete',
                'api_key:create', 'api_key:read', 'api_key:update', 'api_key:delete',
                'system:monitor', 'system:configure'
            ],
            UserRole.RESEARCHER: [
                'model:create', 'model:read', 'model:update',
                'experiment:create', 'experiment:read', 'experiment:update', 'experiment:delete',
                'api_key:create', 'api_key:read', 'api_key:update'
            ],
            UserRole.DEVELOPER: [
                'model:read', 'experiment:read', 'experiment:create',
                'api_key:create', 'api_key:read'
            ],
            UserRole.VIEWER: [
                'model:read', 'experiment:read'
            ]
        }
        
        return role_permissions.get(user.role, [])
    
    def _validate_user_input(self, username: str, email: str, password: str) -> Optional[str]:
        """
        Validate user registration input.
        
        Args:
            username: Username to validate
            email: Email to validate
            password: Password to validate
            
        Returns:
            Error message if validation fails, None if valid
        """
        # Username validation
        if not username or len(username) < 3:
            return 'Username must be at least 3 characters long'
        
        if not re.match(r'^[a-zA-Z0-9_-]+$', username):
            return 'Username can only contain letters, numbers, underscores, and hyphens'
        
        # Email validation
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not email or not re.match(email_pattern, email):
            return 'Invalid email address'
        
        # Password validation
        if not password or len(password) < 8:
            return 'Password must be at least 8 characters long'
        
        if not re.search(r'[A-Z]', password):
            return 'Password must contain at least one uppercase letter'
        
        if not re.search(r'[a-z]', password):
            return 'Password must contain at least one lowercase letter'
        
        if not re.search(r'\d', password):
            return 'Password must contain at least one number'
        
        return None