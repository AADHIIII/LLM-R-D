"""
JWT token service for authentication.
"""
import jwt
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from flask import current_app
import secrets

from ..models.user import User, UserRole


class JWTService:
    """Service for JWT token management."""
    
    @staticmethod
    def generate_tokens(user: User) -> Dict[str, str]:
        """
        Generate access and refresh tokens for user.
        
        Args:
            user: User instance
            
        Returns:
            Dictionary containing access_token and refresh_token
        """
        now = datetime.utcnow()
        
        # Access token payload (short-lived)
        access_payload = {
            'user_id': user.id,
            'username': user.username,
            'email': user.email,
            'role': user.role.value,
            'iat': now,
            'exp': now + timedelta(hours=1),  # 1 hour expiry
            'type': 'access'
        }
        
        # Refresh token payload (long-lived)
        refresh_payload = {
            'user_id': user.id,
            'iat': now,
            'exp': now + timedelta(days=30),  # 30 days expiry
            'type': 'refresh',
            'jti': secrets.token_hex(16)  # Unique token ID for revocation
        }
        
        secret_key = current_app.config['SECRET_KEY']
        
        access_token = jwt.encode(access_payload, secret_key, algorithm='HS256')
        refresh_token = jwt.encode(refresh_payload, secret_key, algorithm='HS256')
        
        return {
            'access_token': access_token,
            'refresh_token': refresh_token,
            'token_type': 'Bearer',
            'expires_in': 3600  # 1 hour in seconds
        }
    
    @staticmethod
    def verify_token(token: str, token_type: str = 'access') -> Optional[Dict[str, Any]]:
        """
        Verify and decode JWT token.
        
        Args:
            token: JWT token string
            token_type: Expected token type ('access' or 'refresh')
            
        Returns:
            Decoded token payload or None if invalid
        """
        try:
            secret_key = current_app.config['SECRET_KEY']
            payload = jwt.decode(token, secret_key, algorithms=['HS256'])
            
            # Verify token type
            if payload.get('type') != token_type:
                return None
            
            # Check expiration
            if datetime.utcnow() > datetime.fromtimestamp(payload['exp']):
                return None
            
            return payload
            
        except jwt.InvalidTokenError:
            return None
        except Exception:
            return None
    
    @staticmethod
    def refresh_access_token(refresh_token: str, user: User) -> Optional[Dict[str, str]]:
        """
        Generate new access token using refresh token.
        
        Args:
            refresh_token: Valid refresh token
            user: User instance
            
        Returns:
            New token pair or None if refresh token invalid
        """
        payload = JWTService.verify_token(refresh_token, 'refresh')
        if not payload or payload.get('user_id') != user.id:
            return None
        
        # Generate new token pair
        return JWTService.generate_tokens(user)
    
    @staticmethod
    def extract_token_from_header(auth_header: str) -> Optional[str]:
        """
        Extract token from Authorization header.
        
        Args:
            auth_header: Authorization header value
            
        Returns:
            Token string or None if invalid format
        """
        if not auth_header:
            return None
        
        parts = auth_header.split()
        if len(parts) != 2 or parts[0].lower() != 'bearer':
            return None
        
        return parts[1]
    
    @staticmethod
    def get_user_from_token(token: str) -> Optional[Dict[str, Any]]:
        """
        Get user information from access token.
        
        Args:
            token: JWT access token
            
        Returns:
            User information or None if token invalid
        """
        payload = JWTService.verify_token(token, 'access')
        if not payload:
            return None
        
        return {
            'user_id': payload.get('user_id'),
            'username': payload.get('username'),
            'email': payload.get('email'),
            'role': payload.get('role')
        }


class TokenBlacklist:
    """Simple in-memory token blacklist (use Redis in production)."""
    
    _blacklisted_tokens = set()
    
    @classmethod
    def add_token(cls, jti: str) -> None:
        """Add token ID to blacklist."""
        cls._blacklisted_tokens.add(jti)
    
    @classmethod
    def is_blacklisted(cls, jti: str) -> bool:
        """Check if token ID is blacklisted."""
        return jti in cls._blacklisted_tokens
    
    @classmethod
    def clear_expired(cls) -> None:
        """Clear expired tokens from blacklist (implement with TTL in Redis)."""
        # In production, use Redis with TTL for automatic cleanup
        pass