"""
Tests for authentication system.
"""
import pytest
import json
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from api.app import create_app
from api.models.user import User, UserRole, UserStatus, APIKey
from api.services.auth_service import AuthService
from api.services.jwt_service import JWTService


@pytest.fixture
def app():
    """Create test Flask app."""
    app = create_app('testing')
    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return app.test_client()


@pytest.fixture
def auth_service():
    """Create auth service instance."""
    return AuthService()


@pytest.fixture
def sample_user():
    """Create sample user for testing."""
    user = User(
        username="testuser",
        email="test@example.com",
        role=UserRole.RESEARCHER,
        status=UserStatus.ACTIVE
    )
    user.set_password("TestPassword123")
    return user


class TestUserModel:
    """Test User model functionality."""
    
    def test_user_creation(self):
        """Test user creation with valid data."""
        user = User(
            username="testuser",
            email="test@example.com",
            role=UserRole.DEVELOPER
        )
        
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.role == UserRole.DEVELOPER
        assert user.status == UserStatus.PENDING
        assert user.failed_login_attempts == 0
    
    def test_password_hashing(self):
        """Test password hashing and verification."""
        user = User()
        password = "TestPassword123"
        
        user.set_password(password)
        
        assert user.password_hash != password
        assert ":" in user.password_hash  # Salt:hash format
        assert user.verify_password(password)
        assert not user.verify_password("wrongpassword")
    
    def test_user_permissions(self):
        """Test role-based permissions."""
        admin = User(role=UserRole.ADMIN)
        researcher = User(role=UserRole.RESEARCHER)
        viewer = User(role=UserRole.VIEWER)
        
        # Admin has all permissions
        assert admin.has_permission('user:delete')
        assert admin.has_permission('system:configure')
        
        # Researcher has model and experiment permissions
        assert researcher.has_permission('model:create')
        assert researcher.has_permission('experiment:delete')
        assert not researcher.has_permission('user:delete')
        
        # Viewer has read-only permissions
        assert viewer.has_permission('model:read')
        assert not viewer.has_permission('model:create')
        assert not viewer.has_permission('experiment:delete')
    
    def test_user_active_status(self):
        """Test user active status checking."""
        user = User(status=UserStatus.ACTIVE)
        assert user.is_active()
        
        user.status = UserStatus.INACTIVE
        assert not user.is_active()
        
        user.status = UserStatus.SUSPENDED
        assert not user.is_active()
        
        # Test locked user
        user.status = UserStatus.ACTIVE
        user.locked_until = datetime.utcnow() + timedelta(minutes=30)
        assert not user.is_active()
        
        # Test unlocked user
        user.locked_until = datetime.utcnow() - timedelta(minutes=30)
        assert user.is_active()
    
    def test_user_serialization(self):
        """Test user to/from dict conversion."""
        user = User(
            username="testuser",
            email="test@example.com",
            role=UserRole.RESEARCHER
        )
        
        # Test to_dict
        user_dict = user.to_dict()
        assert user_dict['username'] == "testuser"
        assert user_dict['email'] == "test@example.com"
        assert user_dict['role'] == "researcher"
        assert 'password_hash' not in user_dict
        
        # Test to_dict with sensitive data
        sensitive_dict = user.to_dict(include_sensitive=True)
        assert 'failed_login_attempts' in sensitive_dict
        
        # Test from_dict
        new_user = User.from_dict(user_dict)
        assert new_user.username == user.username
        assert new_user.email == user.email
        assert new_user.role == user.role


class TestAPIKeyModel:
    """Test APIKey model functionality."""
    
    def test_api_key_generation(self):
        """Test API key generation."""
        key = APIKey.generate_key()
        
        assert key.startswith("llm_opt_")
        assert len(key) > 20  # Should be reasonably long
    
    def test_api_key_hashing(self):
        """Test API key hashing and verification."""
        api_key = APIKey()
        key_string = "test-api-key-12345"
        
        api_key.set_key(key_string)
        
        assert api_key.key_prefix == "test-api"
        assert api_key.key_hash != key_string
        assert api_key.verify_key(key_string)
        assert not api_key.verify_key("wrong-key")
    
    def test_api_key_validity(self):
        """Test API key validity checking."""
        api_key = APIKey(is_active=True)
        assert api_key.is_valid()
        
        api_key.is_active = False
        assert not api_key.is_valid()
        
        # Test expiration
        api_key.is_active = True
        api_key.expires_at = datetime.utcnow() - timedelta(days=1)
        assert not api_key.is_valid()
        
        api_key.expires_at = datetime.utcnow() + timedelta(days=1)
        assert api_key.is_valid()
    
    def test_api_key_usage_tracking(self):
        """Test API key usage tracking."""
        api_key = APIKey()
        initial_count = api_key.usage_count
        
        api_key.record_usage()
        
        assert api_key.usage_count == initial_count + 1
        assert api_key.last_used is not None
        assert isinstance(api_key.last_used, datetime)


class TestJWTService:
    """Test JWT service functionality."""
    
    def test_token_generation(self, app, sample_user):
        """Test JWT token generation."""
        with app.app_context():
            tokens = JWTService.generate_tokens(sample_user)
            
            assert 'access_token' in tokens
            assert 'refresh_token' in tokens
            assert 'token_type' in tokens
            assert 'expires_in' in tokens
            assert tokens['token_type'] == 'Bearer'
    
    def test_token_verification(self, app, sample_user):
        """Test JWT token verification."""
        with app.app_context():
            tokens = JWTService.generate_tokens(sample_user)
            access_token = tokens['access_token']
            refresh_token = tokens['refresh_token']
            
            # Verify access token
            access_payload = JWTService.verify_token(access_token, 'access')
            assert access_payload is not None
            assert access_payload['user_id'] == sample_user.id
            assert access_payload['type'] == 'access'
            
            # Verify refresh token
            refresh_payload = JWTService.verify_token(refresh_token, 'refresh')
            assert refresh_payload is not None
            assert refresh_payload['user_id'] == sample_user.id
            assert refresh_payload['type'] == 'refresh'
    
    def test_invalid_token_verification(self, app):
        """Test verification of invalid tokens."""
        with app.app_context():
            # Test invalid token
            assert JWTService.verify_token("invalid-token") is None
            
            # Test wrong token type
            user = User(username="test")
            tokens = JWTService.generate_tokens(user)
            access_token = tokens['access_token']
            
            # Try to verify access token as refresh token
            assert JWTService.verify_token(access_token, 'refresh') is None
    
    def test_token_extraction(self):
        """Test token extraction from headers."""
        # Valid Bearer token
        header = "Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9"
        token = JWTService.extract_token_from_header(header)
        assert token == "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9"
        
        # Invalid format
        assert JWTService.extract_token_from_header("InvalidHeader") is None
        assert JWTService.extract_token_from_header("Basic token") is None
        assert JWTService.extract_token_from_header("") is None
        assert JWTService.extract_token_from_header(None) is None
    
    def test_user_from_token(self, app, sample_user):
        """Test extracting user info from token."""
        with app.app_context():
            tokens = JWTService.generate_tokens(sample_user)
            access_token = tokens['access_token']
            
            user_info = JWTService.get_user_from_token(access_token)
            
            assert user_info is not None
            assert user_info['user_id'] == sample_user.id
            assert user_info['username'] == sample_user.username
            assert user_info['email'] == sample_user.email
            assert user_info['role'] == sample_user.role.value


class TestAuthService:
    """Test authentication service functionality."""
    
    def test_user_registration(self, auth_service):
        """Test user registration."""
        result = auth_service.register_user(
            username="newuser",
            email="newuser@example.com",
            password="ValidPassword123",
            role=UserRole.DEVELOPER
        )
        
        assert result['success'] is True
        assert 'user' in result
        assert result['user']['username'] == "newuser"
        assert result['user']['email'] == "newuser@example.com"
        assert result['user']['role'] == "developer"
    
    def test_duplicate_user_registration(self, auth_service):
        """Test registration with duplicate username/email."""
        # Register first user
        auth_service.register_user(
            username="testuser",
            email="test@example.com",
            password="ValidPassword123"
        )
        
        # Try to register with same username
        result = auth_service.register_user(
            username="testuser",
            email="different@example.com",
            password="ValidPassword123"
        )
        assert result['success'] is False
        assert "already exists" in result['error']
        
        # Try to register with same email
        result = auth_service.register_user(
            username="differentuser",
            email="test@example.com",
            password="ValidPassword123"
        )
        assert result['success'] is False
        assert "already registered" in result['error']
    
    def test_invalid_registration_data(self, auth_service):
        """Test registration with invalid data."""
        # Short username
        result = auth_service.register_user("ab", "test@example.com", "ValidPassword123")
        assert result['success'] is False
        
        # Invalid email
        result = auth_service.register_user("testuser", "invalid-email", "ValidPassword123")
        assert result['success'] is False
        
        # Weak password
        result = auth_service.register_user("testuser", "test@example.com", "weak")
        assert result['success'] is False
    
    def test_user_authentication(self, auth_service):
        """Test user authentication."""
        # Register user first
        auth_service.register_user(
            username="authuser",
            email="auth@example.com",
            password="ValidPassword123"
        )
        
        # Test successful authentication
        result = auth_service.authenticate_user("authuser", "ValidPassword123")
        assert result['success'] is True
        assert 'user' in result
        assert 'tokens' in result
        
        # Test authentication with email
        result = auth_service.authenticate_user("auth@example.com", "ValidPassword123")
        assert result['success'] is True
    
    def test_failed_authentication(self, auth_service):
        """Test failed authentication scenarios."""
        # Register user first
        auth_service.register_user(
            username="authuser",
            email="auth@example.com",
            password="ValidPassword123"
        )
        
        # Wrong password
        result = auth_service.authenticate_user("authuser", "WrongPassword")
        assert result['success'] is False
        
        # Non-existent user
        result = auth_service.authenticate_user("nonexistent", "ValidPassword123")
        assert result['success'] is False
    
    def test_account_lockout(self, auth_service):
        """Test account lockout after failed attempts."""
        # Register user
        auth_service.register_user(
            username="lockuser",
            email="lock@example.com",
            password="ValidPassword123"
        )
        
        # Make 5 failed login attempts
        for _ in range(5):
            result = auth_service.authenticate_user("lockuser", "WrongPassword")
            assert result['success'] is False
        
        # Account should now be locked
        result = auth_service.authenticate_user("lockuser", "ValidPassword123")
        assert result['success'] is False
        assert "locked" in result['error']
    
    def test_api_key_creation(self, auth_service):
        """Test API key creation."""
        # Register user first
        user_result = auth_service.register_user(
            username="apiuser",
            email="api@example.com",
            password="ValidPassword123",
            role=UserRole.RESEARCHER
        )
        user_id = user_result['user']['id']
        
        # Create API key
        result = auth_service.create_api_key(
            user_id=user_id,
            name="Test API Key",
            permissions=["model:read", "experiment:create"]
        )
        
        assert result['success'] is True
        assert 'api_key' in result
        assert 'key' in result  # Raw key returned only once
        assert result['api_key']['name'] == "Test API Key"
        assert result['api_key']['permissions'] == ["model:read", "experiment:create"]
    
    def test_api_key_authentication(self, auth_service):
        """Test API key authentication."""
        # Register user and create API key
        user_result = auth_service.register_user(
            username="apiuser",
            email="api@example.com",
            password="ValidPassword123"
        )
        user_id = user_result['user']['id']
        
        key_result = auth_service.create_api_key(
            user_id=user_id,
            name="Test Key",
            permissions=["model:read"]
        )
        api_key = key_result['key']
        
        # Test authentication
        result = auth_service.authenticate_api_key(api_key)
        assert result['success'] is True
        assert 'user' in result
        assert 'api_key' in result
        assert result['permissions'] == ["model:read"]
    
    def test_invalid_api_key_authentication(self, auth_service):
        """Test authentication with invalid API key."""
        result = auth_service.authenticate_api_key("invalid-key")
        assert result['success'] is False


class TestAuthenticationEndpoints:
    """Test authentication API endpoints."""
    
    def test_register_endpoint(self, client):
        """Test user registration endpoint."""
        response = client.post('/api/v1/auth/register', 
                             json={
                                 'username': 'testuser',
                                 'email': 'test@example.com',
                                 'password': 'ValidPassword123',
                                 'role': 'developer'
                             })
        
        assert response.status_code == 201
        data = json.loads(response.data)
        assert data['success'] is True
        assert 'user' in data
    
    def test_register_invalid_data(self, client):
        """Test registration with invalid data."""
        response = client.post('/api/v1/auth/register',
                             json={
                                 'username': 'ab',  # Too short
                                 'email': 'invalid-email',
                                 'password': 'weak'
                             })
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['success'] is False
    
    def test_login_endpoint(self, client):
        """Test user login endpoint."""
        # Register user first
        client.post('/api/v1/auth/register',
                   json={
                       'username': 'loginuser',
                       'email': 'login@example.com',
                       'password': 'ValidPassword123'
                   })
        
        # Test login
        response = client.post('/api/v1/auth/login',
                             json={
                                 'username': 'loginuser',
                                 'password': 'ValidPassword123'
                             })
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True
        assert 'tokens' in data
        assert 'access_token' in data['tokens']
        assert 'refresh_token' in data['tokens']
    
    def test_login_invalid_credentials(self, client):
        """Test login with invalid credentials."""
        response = client.post('/api/v1/auth/login',
                             json={
                                 'username': 'nonexistent',
                                 'password': 'WrongPassword'
                             })
        
        assert response.status_code == 401
        data = json.loads(response.data)
        assert data['success'] is False
    
    def test_protected_endpoint_without_auth(self, client):
        """Test accessing protected endpoint without authentication."""
        response = client.post('/api/v1/generate',
                             json={
                                 'prompt': 'Test prompt',
                                 'model_id': 'gpt-4'
                             })
        
        assert response.status_code == 401
        data = json.loads(response.data)
        assert data['error'] == 'authentication_required'
    
    def test_protected_endpoint_with_auth(self, client):
        """Test accessing protected endpoint with authentication."""
        # Register and login user
        client.post('/api/v1/auth/register',
                   json={
                       'username': 'authuser',
                       'email': 'auth@example.com',
                       'password': 'ValidPassword123',
                       'role': 'researcher'
                   })
        
        login_response = client.post('/api/v1/auth/login',
                                   json={
                                       'username': 'authuser',
                                       'password': 'ValidPassword123'
                                   })
        
        tokens = json.loads(login_response.data)['tokens']
        access_token = tokens['access_token']
        
        # Test protected endpoint with token
        with patch('api.services.text_generator.TextGenerator') as mock_generator:
            mock_instance = MagicMock()
            mock_instance.generate.return_value = {'text': 'Generated text'}
            mock_generator.return_value = mock_instance
            
            response = client.post('/api/v1/generate',
                                 json={
                                     'prompt': 'Test prompt',
                                     'model_id': 'gpt-4'
                                 },
                                 headers={
                                     'Authorization': f'Bearer {access_token}'
                                 })
            
            # Should not get 401 (might get other errors due to mocking)
            assert response.status_code != 401
    
    def test_api_key_creation_endpoint(self, client):
        """Test API key creation endpoint."""
        # Register and login user
        client.post('/api/v1/auth/register',
                   json={
                       'username': 'keyuser',
                       'email': 'key@example.com',
                       'password': 'ValidPassword123',
                       'role': 'researcher'
                   })
        
        login_response = client.post('/api/v1/auth/login',
                                   json={
                                       'username': 'keyuser',
                                       'password': 'ValidPassword123'
                                   })
        
        tokens = json.loads(login_response.data)['tokens']
        access_token = tokens['access_token']
        
        # Create API key
        response = client.post('/api/v1/auth/api-keys',
                             json={
                                 'name': 'Test API Key',
                                 'permissions': ['model:read', 'experiment:create']
                             },
                             headers={
                                 'Authorization': f'Bearer {access_token}'
                             })
        
        assert response.status_code == 201
        data = json.loads(response.data)
        assert data['success'] is True
        assert 'key' in data  # Raw key returned
    
    def test_token_refresh_endpoint(self, client):
        """Test token refresh endpoint."""
        # Register and login user
        client.post('/api/v1/auth/register',
                   json={
                       'username': 'refreshuser',
                       'email': 'refresh@example.com',
                       'password': 'ValidPassword123'
                   })
        
        login_response = client.post('/api/v1/auth/login',
                                   json={
                                       'username': 'refreshuser',
                                       'password': 'ValidPassword123'
                                   })
        
        tokens = json.loads(login_response.data)['tokens']
        refresh_token = tokens['refresh_token']
        
        # Refresh token
        response = client.post('/api/v1/auth/refresh',
                             json={
                                 'refresh_token': refresh_token
                             })
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True
        assert 'tokens' in data


class TestSecurityFeatures:
    """Test security features and edge cases."""
    
    def test_password_requirements(self, auth_service):
        """Test password strength requirements."""
        test_cases = [
            ("short", False),  # Too short
            ("nouppercase123", False),  # No uppercase
            ("NOLOWERCASE123", False),  # No lowercase
            ("NoNumbers", False),  # No numbers
            ("ValidPassword123", True),  # Valid password
        ]
        
        for password, should_succeed in test_cases:
            result = auth_service.register_user(
                username=f"user_{password}",
                email=f"{password}@example.com",
                password=password
            )
            assert result['success'] == should_succeed
    
    def test_sql_injection_protection(self, client):
        """Test protection against SQL injection attempts."""
        # Try SQL injection in username
        response = client.post('/api/v1/auth/login',
                             json={
                                 'username': "admin'; DROP TABLE users; --",
                                 'password': 'password'
                             })
        
        # Should not cause server error
        assert response.status_code in [400, 401]
    
    def test_rate_limiting(self, client):
        """Test rate limiting on authentication endpoints."""
        # This would require actual rate limiting implementation
        # For now, just test that multiple requests don't cause errors
        for _ in range(10):
            response = client.post('/api/v1/auth/login',
                                 json={
                                     'username': 'testuser',
                                     'password': 'wrongpassword'
                                 })
            assert response.status_code in [400, 401, 429]  # 429 for rate limit
    
    def test_token_expiration_handling(self, app, sample_user):
        """Test handling of expired tokens."""
        with app.app_context():
            # Create token with past expiration (would need to mock datetime)
            # For now, test with invalid token format
            invalid_token = "invalid.token.format"
            
            user_info = JWTService.get_user_from_token(invalid_token)
            assert user_info is None
    
    def test_permission_escalation_prevention(self, client):
        """Test prevention of permission escalation."""
        # Register viewer user
        client.post('/api/v1/auth/register',
                   json={
                       'username': 'viewer',
                       'email': 'viewer@example.com',
                       'password': 'ValidPassword123',
                       'role': 'viewer'
                   })
        
        login_response = client.post('/api/v1/auth/login',
                                   json={
                                       'username': 'viewer',
                                       'password': 'ValidPassword123'
                                   })
        
        tokens = json.loads(login_response.data)['tokens']
        access_token = tokens['access_token']
        
        # Try to create API key with permissions viewer doesn't have
        response = client.post('/api/v1/auth/api-keys',
                             json={
                                 'name': 'Escalated Key',
                                 'permissions': ['user:delete', 'system:configure']
                             },
                             headers={
                                 'Authorization': f'Bearer {access_token}'
                             })
        
        assert response.status_code == 403
        data = json.loads(response.data)
        assert 'permission' in data['error'].lower()


if __name__ == '__main__':
    pytest.main([__file__])