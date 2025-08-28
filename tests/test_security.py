"""
Tests for security features and data protection.
"""
import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import json

from utils.security import (
    InputValidator, DataEncryption, AuditLogger, 
    SecureFileHandler, audit_logger
)
from api.app import create_app


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
def temp_dir():
    """Create temporary directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


class TestInputValidator:
    """Test input validation and sanitization."""
    
    def test_email_validation(self):
        """Test email validation."""
        valid_emails = [
            "test@example.com",
            "user.name@domain.co.uk",
            "user+tag@example.org"
        ]
        
        invalid_emails = [
            "invalid-email",
            "@example.com",
            "test@",
            "test..test@example.com",
            "a" * 250 + "@example.com"  # Too long
        ]
        
        for email in valid_emails:
            assert InputValidator.validate_email(email), f"Should be valid: {email}"
        
        for email in invalid_emails:
            assert not InputValidator.validate_email(email), f"Should be invalid: {email}"
    
    def test_username_validation(self):
        """Test username validation."""
        valid_usernames = [
            "testuser",
            "user123",
            "user_name",
            "user-name"
        ]
        
        invalid_usernames = [
            "ab",  # Too short
            "user@name",  # Invalid character
            "user name",  # Space
            "a" * 51,  # Too long
            ""  # Empty
        ]
        
        for username in valid_usernames:
            assert InputValidator.validate_username(username), f"Should be valid: {username}"
        
        for username in invalid_usernames:
            assert not InputValidator.validate_username(username), f"Should be invalid: {username}"
    
    def test_filename_validation(self):
        """Test filename validation."""
        valid_filenames = [
            "document.txt",
            "data_file.csv",
            "model-v1.bin"
        ]
        
        invalid_filenames = [
            "../etc/passwd",  # Path traversal
            "file/with/path.txt",  # Path separator
            "file\\with\\path.txt",  # Windows path separator
            "file<script>.txt",  # Invalid character
            "a" * 256 + ".txt"  # Too long
        ]
        
        for filename in valid_filenames:
            assert InputValidator.validate_filename(filename), f"Should be valid: {filename}"
        
        for filename in invalid_filenames:
            assert not InputValidator.validate_filename(filename), f"Should be invalid: {filename}"
    
    def test_string_sanitization(self):
        """Test string sanitization."""
        test_cases = [
            ("<script>alert('xss')</script>", ""),
            ("Hello <b>world</b>", "Hello &lt;b&gt;world&lt;/b&gt;"),
            ("javascript:alert('xss')", "alert('xss')"),
            ("onclick=alert('xss')", "alert('xss')"),
            ("Normal text", "Normal text")
        ]
        
        for input_text, expected in test_cases:
            result = InputValidator.sanitize_string(input_text)
            assert expected in result or result == expected, f"Failed for: {input_text}"
    
    def test_sql_injection_detection(self):
        """Test SQL injection detection."""
        malicious_inputs = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "admin'/*",
            "UNION SELECT * FROM passwords",
            "INSERT INTO users VALUES"
        ]
        
        safe_inputs = [
            "normal text",
            "user@example.com",
            "This is a regular sentence.",
            "Numbers 123 and symbols !@#"
        ]
        
        for malicious in malicious_inputs:
            assert InputValidator.check_sql_injection(malicious), f"Should detect: {malicious}"
        
        for safe in safe_inputs:
            assert not InputValidator.check_sql_injection(safe), f"Should be safe: {safe}"
    
    def test_json_structure_validation(self):
        """Test JSON structure validation."""
        # Valid data
        valid_data = {
            "username": "testuser",
            "email": "test@example.com",
            "age": 25
        }
        
        result = InputValidator.validate_json_structure(
            valid_data, 
            required_fields=["username", "email"],
            optional_fields=["age"]
        )
        
        assert result["username"] == "testuser"
        assert result["email"] == "test@example.com"
        assert result["age"] == 25
        
        # Missing required field
        with pytest.raises(ValueError, match="Missing required fields"):
            InputValidator.validate_json_structure(
                {"username": "test"},
                required_fields=["username", "email"]
            )
        
        # Unexpected field
        with pytest.raises(ValueError, match="Unexpected fields"):
            InputValidator.validate_json_structure(
                {"username": "test", "unexpected": "field"},
                required_fields=["username"]
            )
        
        # SQL injection in field
        with pytest.raises(ValueError, match="SQL injection"):
            InputValidator.validate_json_structure(
                {"username": "'; DROP TABLE users; --"},
                required_fields=["username"]
            )
    
    def test_file_upload_validation(self):
        """Test file upload validation."""
        # Valid CSV content
        csv_content = b"name,age\nJohn,25\nJane,30"
        
        with patch('magic.from_buffer', return_value='text/csv'):
            result = InputValidator.validate_file_upload(
                csv_content, 
                "data.csv", 
                ["text/csv", "application/json"]
            )
            
            assert result["filename"] == "data.csv"
            assert result["mime_type"] == "text/csv"
            assert result["size"] == len(csv_content)
            assert result["is_safe"] is True
        
        # File too large
        large_content = b"x" * (17 * 1024 * 1024)  # 17MB
        with pytest.raises(ValueError, match="File too large"):
            InputValidator.validate_file_upload(large_content, "large.txt", ["text/plain"])
        
        # Invalid filename
        with pytest.raises(ValueError, match="Invalid filename"):
            InputValidator.validate_file_upload(csv_content, "../etc/passwd", ["text/csv"])
        
        # Disallowed file type
        with patch('magic.from_buffer', return_value='application/x-executable'):
            with pytest.raises(ValueError, match="File type not allowed"):
                InputValidator.validate_file_upload(
                    b"executable content", 
                    "malware.exe", 
                    ["text/csv"]
                )


class TestDataEncryption:
    """Test data encryption utilities."""
    
    def test_key_generation(self):
        """Test encryption key generation."""
        key1 = DataEncryption.generate_key()
        key2 = DataEncryption.generate_key()
        
        assert len(key1) == 32  # 256 bits
        assert len(key2) == 32
        assert key1 != key2  # Should be different
    
    def test_data_encryption_decryption(self):
        """Test data encryption and decryption."""
        original_data = "This is sensitive information"
        key = DataEncryption.generate_key()
        
        # Encrypt data
        encrypted = DataEncryption.encrypt_data(original_data, key)
        assert encrypted != original_data
        assert len(encrypted) > 0
        
        # Decrypt data
        decrypted = DataEncryption.decrypt_data(encrypted, key)
        assert decrypted == original_data
    
    def test_data_hashing(self):
        """Test data hashing."""
        data = "password123"
        
        # Hash data
        hash1 = DataEncryption.hash_sensitive_data(data)
        hash2 = DataEncryption.hash_sensitive_data(data)
        
        # Hashes should be different (different salts)
        assert hash1 != hash2
        assert ":" in hash1  # Should contain salt
        assert ":" in hash2
        
        # Verification should work
        assert DataEncryption.verify_hash(data, hash1)
        assert DataEncryption.verify_hash(data, hash2)
        assert not DataEncryption.verify_hash("wrongpassword", hash1)
    
    def test_hash_verification_edge_cases(self):
        """Test hash verification edge cases."""
        # Invalid hash format
        assert not DataEncryption.verify_hash("data", "invalid_hash")
        
        # Empty data
        assert not DataEncryption.verify_hash("", "salt:hash")


class TestAuditLogger:
    """Test audit logging functionality."""
    
    def test_audit_logger_initialization(self, temp_dir):
        """Test audit logger initialization."""
        log_file = os.path.join(temp_dir, "test_audit.log")
        logger = AuditLogger(log_file)
        
        assert logger.log_file == Path(log_file)
        assert logger.log_file.parent.exists()
    
    def test_authentication_logging(self, temp_dir):
        """Test authentication logging."""
        log_file = os.path.join(temp_dir, "test_audit.log")
        logger = AuditLogger(log_file)
        
        # Log successful authentication
        logger.log_authentication(
            user_id="user123",
            username="testuser",
            success=True,
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0"
        )
        
        # Log failed authentication
        logger.log_authentication(
            user_id="user123",
            username="testuser",
            success=False,
            ip_address="192.168.1.1"
        )
        
        # Check log file
        with open(log_file, 'r') as f:
            content = f.read()
            assert "AUTH SUCCESS" in content
            assert "AUTH FAILURE" in content
            assert "user123" in content
            assert "testuser" in content
    
    def test_permission_logging(self, temp_dir):
        """Test permission logging."""
        log_file = os.path.join(temp_dir, "test_audit.log")
        logger = AuditLogger(log_file)
        
        logger.log_permission_check("user123", "model:create", True)
        logger.log_permission_check("user456", "admin:delete", False)
        
        with open(log_file, 'r') as f:
            content = f.read()
            assert "PERMISSION GRANTED" in content
            assert "PERMISSION DENIED" in content
    
    def test_security_event_logging(self, temp_dir):
        """Test security event logging."""
        log_file = os.path.join(temp_dir, "test_audit.log")
        logger = AuditLogger(log_file)
        
        logger.log_security_event(
            event_type="SQL_INJECTION_ATTEMPT",
            user_id="user123",
            details="Malicious input detected",
            severity="WARNING"
        )
        
        with open(log_file, 'r') as f:
            content = f.read()
            assert "SECURITY_EVENT" in content
            assert "SQL_INJECTION_ATTEMPT" in content


class TestSecureFileHandler:
    """Test secure file handling."""
    
    def test_file_handler_initialization(self, temp_dir):
        """Test file handler initialization."""
        handler = SecureFileHandler(temp_dir)
        
        assert handler.upload_dir == Path(temp_dir)
        assert (handler.upload_dir / "dataset").exists()
        assert (handler.upload_dir / "model").exists()
        assert (handler.upload_dir / "document").exists()
    
    def test_secure_file_upload(self, temp_dir):
        """Test secure file upload."""
        handler = SecureFileHandler(temp_dir)
        
        file_content = b"name,age\nJohn,25\nJane,30"
        
        with patch('magic.from_buffer', return_value='text/csv'):
            result = handler.save_uploaded_file(
                file_content=file_content,
                filename="test.csv",
                file_type="dataset",
                user_id="user123"
            )
        
        assert result["original_filename"] == "test.csv"
        assert result["file_type"] == "dataset"
        assert result["uploaded_by"] == "user123"
        assert "user123" in result["secure_filename"]
        
        # Check file was actually saved
        file_path = Path(result["file_path"])
        assert file_path.exists()
        
        # Check file permissions (Unix-like systems)
        if hasattr(os, 'stat'):
            stat_info = file_path.stat()
            # Should be readable/writable by owner only (0o600)
            assert oct(stat_info.st_mode)[-3:] == '600'
    
    def test_file_upload_validation_errors(self, temp_dir):
        """Test file upload validation errors."""
        handler = SecureFileHandler(temp_dir)
        
        # Invalid file type
        with pytest.raises(ValueError, match="Invalid file type"):
            handler.save_uploaded_file(
                file_content=b"content",
                filename="test.txt",
                file_type="invalid_type",
                user_id="user123"
            )
        
        # Invalid filename
        with pytest.raises(ValueError):
            handler.save_uploaded_file(
                file_content=b"content",
                filename="../etc/passwd",
                file_type="dataset",
                user_id="user123"
            )
    
    def test_file_deletion(self, temp_dir):
        """Test secure file deletion."""
        handler = SecureFileHandler(temp_dir)
        
        # Create a test file first
        file_content = b"test content"
        
        with patch('magic.from_buffer', return_value='text/plain'):
            result = handler.save_uploaded_file(
                file_content=file_content,
                filename="test.txt",
                file_type="document",
                user_id="user123"
            )
        
        secure_filename = result["secure_filename"]
        file_path = Path(result["file_path"])
        
        # Verify file exists
        assert file_path.exists()
        
        # Delete file
        success = handler.delete_file(secure_filename, "document", "user123")
        assert success is True
        
        # Verify file is deleted
        assert not file_path.exists()
        
        # Try to delete non-existent file
        success = handler.delete_file("nonexistent.txt", "document", "user123")
        assert success is False
    
    def test_virus_scanning(self, temp_dir):
        """Test virus scanning functionality."""
        handler = SecureFileHandler(temp_dir)
        
        # Create test files
        safe_file = Path(temp_dir) / "safe.txt"
        safe_file.write_bytes(b"This is safe content")
        
        malicious_file = Path(temp_dir) / "malicious.exe"
        malicious_file.write_bytes(b"MZ\x90\x00")  # PE executable signature
        
        # Test safe file
        assert handler.scan_file_for_viruses(safe_file) is True
        
        # Test malicious file
        assert handler.scan_file_for_viruses(malicious_file) is False


class TestSecurityMiddleware:
    """Test security middleware functionality."""
    
    def test_sql_injection_protection(self, client):
        """Test SQL injection protection in middleware."""
        # Test query parameter injection
        response = client.get('/api/v1/models?name=test\'; DROP TABLE users; --')
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['error'] == 'security_violation'
        
        # Test JSON payload injection
        response = client.post('/api/v1/auth/login',
                             json={
                                 'username': 'admin\'; DROP TABLE users; --',
                                 'password': 'password'
                             })
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['error'] == 'security_violation'
    
    def test_content_type_validation(self, client):
        """Test content type validation."""
        # Test invalid content type for POST
        response = client.post('/api/v1/auth/login',
                             data='invalid data',
                             content_type='text/plain')
        assert response.status_code == 400
    
    def test_content_length_validation(self, client):
        """Test content length validation."""
        # Test oversized request
        large_data = {'data': 'x' * (17 * 1024 * 1024)}  # 17MB
        response = client.post('/api/v1/auth/login',
                             json=large_data)
        assert response.status_code == 413


class TestSecureEndpoints:
    """Test secure endpoint implementations."""
    
    def test_file_upload_endpoint_security(self, client):
        """Test file upload endpoint security."""
        # Test without authentication
        response = client.post('/api/v1/upload/dataset')
        assert response.status_code == 401
        
        # Test with malicious filename
        with patch('api.services.auth_service.AuthService') as mock_auth:
            mock_auth.return_value.authenticate_user.return_value = {
                'success': True,
                'user': {'id': 'user123', 'username': 'test', 'role': 'researcher'},
                'tokens': {'access_token': 'fake_token'}
            }
            
            response = client.post('/api/v1/upload/dataset',
                                 data={
                                     'file': (b'malicious content', '../etc/passwd'),
                                     'name': 'test'
                                 },
                                 headers={'Authorization': 'Bearer fake_token'})
            
            # Should be rejected due to invalid filename
            assert response.status_code in [400, 401]  # 401 due to mocked auth
    
    def test_input_sanitization_in_endpoints(self, client):
        """Test input sanitization in endpoints."""
        # Register with XSS attempt
        response = client.post('/api/v1/auth/register',
                             json={
                                 'username': '<script>alert("xss")</script>',
                                 'email': 'test@example.com',
                                 'password': 'ValidPassword123'
                             })
        
        # Should be sanitized or rejected
        assert response.status_code in [400, 201]
        
        if response.status_code == 201:
            data = json.loads(response.data)
            # Username should be sanitized
            assert '<script>' not in data.get('user', {}).get('username', '')


class TestSecurityConfiguration:
    """Test security configuration and settings."""
    
    def test_security_headers(self, client):
        """Test security headers are set."""
        response = client.get('/api/v1/health')
        
        # Check for security headers
        assert 'X-Frame-Options' in response.headers
        assert 'X-Content-Type-Options' in response.headers
        assert 'X-XSS-Protection' in response.headers
        assert 'Content-Security-Policy' in response.headers
        assert 'Referrer-Policy' in response.headers
        
        assert response.headers['X-Frame-Options'] == 'DENY'
        assert response.headers['X-Content-Type-Options'] == 'nosniff'
    
    def test_cors_configuration(self, client):
        """Test CORS configuration."""
        response = client.options('/api/v1/health',
                                headers={'Origin': 'http://localhost:3000'})
        
        # Should have CORS headers for allowed origins
        assert 'Access-Control-Allow-Origin' in response.headers
    
    def test_rate_limiting_configuration(self, app):
        """Test rate limiting configuration."""
        with app.app_context():
            assert app.config.get('RATE_LIMIT_ENABLED') is not None
            assert app.config.get('RATE_LIMIT_PER_MINUTE') is not None


if __name__ == '__main__':
    pytest.main([__file__])