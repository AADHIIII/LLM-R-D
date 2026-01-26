"""
User and authentication models.
"""
import uuid
import hashlib
import secrets
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List


class UserRole(Enum):
    """User role enumeration."""
    ADMIN = "admin"
    RESEARCHER = "researcher"
    DEVELOPER = "developer"
    VIEWER = "viewer"


class UserStatus(Enum):
    """User account status enumeration."""
    PENDING = "pending"
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"


class User:
    """User model for authentication and authorization."""

    ROLE_PERMISSIONS = {
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

    def __init__(
        self,
        username: str = "",
        email: str = "",
        role: UserRole = UserRole.VIEWER,
        status: UserStatus = UserStatus.PENDING
    ):
        self.id = str(uuid.uuid4())
        self.username = username
        self.email = email
        self.role = role
        self.status = status
        self.password_hash: Optional[str] = None
        self.failed_login_attempts: int = 0
        self.locked_until: Optional[datetime] = None
        self.last_login: Optional[datetime] = None
        self.created_at: datetime = datetime.utcnow()

    def set_password(self, password: str) -> None:
        """Hash and store password with salt."""
        salt = secrets.token_hex(16)
        hash_obj = hashlib.sha256((salt + password).encode())
        self.password_hash = f"{salt}:{hash_obj.hexdigest()}"

    def verify_password(self, password: str) -> bool:
        """Verify password against stored hash."""
        if not self.password_hash:
            return False

        try:
            salt, stored_hash = self.password_hash.split(':')
            hash_obj = hashlib.sha256((salt + password).encode())
            return hash_obj.hexdigest() == stored_hash
        except ValueError:
            return False

    def is_active(self) -> bool:
        """Check if user account is active and not locked."""
        if self.status != UserStatus.ACTIVE:
            return False

        if self.locked_until and self.locked_until > datetime.utcnow():
            return False

        return True

    def has_permission(self, permission: str) -> bool:
        """Check if user has the specified permission."""
        permissions = self.ROLE_PERMISSIONS.get(self.role, [])
        return permission in permissions

    def to_dict(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """Convert user to dictionary."""
        data = {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'role': self.role.value,
            'status': self.status.value,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None
        }

        if include_sensitive:
            data['failed_login_attempts'] = self.failed_login_attempts
            data['locked_until'] = self.locked_until.isoformat() if self.locked_until else None

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'User':
        """Create user from dictionary."""
        role = UserRole(data.get('role', 'viewer'))
        status = UserStatus(data.get('status', 'pending'))

        user = cls(
            username=data.get('username', ''),
            email=data.get('email', ''),
            role=role,
            status=status
        )

        if 'id' in data:
            user.id = data['id']

        return user


class APIKey:
    """API Key model for programmatic access."""

    def __init__(
        self,
        user_id: str = "",
        name: str = "",
        permissions: Optional[List[str]] = None,
        expires_at: Optional[datetime] = None,
        is_active: bool = True
    ):
        self.id = str(uuid.uuid4())
        self.user_id = user_id
        self.name = name
        self.permissions = permissions or []
        self.expires_at = expires_at
        self.is_active = is_active
        self.key_prefix: str = ""
        self.key_hash: str = ""
        self.usage_count: int = 0
        self.last_used: Optional[datetime] = None
        self.created_at: datetime = datetime.utcnow()

    @staticmethod
    def generate_key() -> str:
        """Generate a new API key string."""
        return f"llm_opt_{secrets.token_hex(24)}"

    def set_key(self, key_string: str) -> None:
        """Hash and store API key with prefix for lookup."""
        self.key_prefix = key_string[:8]
        hash_obj = hashlib.sha256(key_string.encode())
        self.key_hash = hash_obj.hexdigest()

    def verify_key(self, key_string: str) -> bool:
        """Verify API key against stored hash."""
        if not self.key_hash:
            return False

        hash_obj = hashlib.sha256(key_string.encode())
        return hash_obj.hexdigest() == self.key_hash

    def is_valid(self) -> bool:
        """Check if API key is valid (active and not expired)."""
        if not self.is_active:
            return False

        if self.expires_at and self.expires_at < datetime.utcnow():
            return False

        return True

    def record_usage(self) -> None:
        """Record API key usage."""
        self.usage_count += 1
        self.last_used = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert API key to dictionary (without sensitive data)."""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'name': self.name,
            'key_prefix': self.key_prefix,
            'permissions': self.permissions,
            'is_active': self.is_active,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'usage_count': self.usage_count,
            'last_used': self.last_used.isoformat() if self.last_used else None,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
