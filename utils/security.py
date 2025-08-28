"""
Security utilities for data protection and validation.
"""
import re
import html
import hashlib
import secrets
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import logging
from pathlib import Path
import mimetypes
import magic

logger = logging.getLogger(__name__)


class InputValidator:
    """Input validation and sanitization utilities."""
    
    # Common regex patterns
    EMAIL_PATTERN = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    USERNAME_PATTERN = re.compile(r'^[a-zA-Z0-9_-]{3,50}$')
    FILENAME_PATTERN = re.compile(r'^[a-zA-Z0-9._-]+$')
    
    # Dangerous patterns to block
    SQL_INJECTION_PATTERNS = [
        re.compile(r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)", re.IGNORECASE),
        re.compile(r"[';\"\\]"),
        re.compile(r"--"),
        re.compile(r"/\*.*\*/"),
    ]
    
    XSS_PATTERNS = [
        re.compile(r"<script[^>]*>.*?</script>", re.IGNORECASE | re.DOTALL),
        re.compile(r"javascript:", re.IGNORECASE),
        re.compile(r"on\w+\s*=", re.IGNORECASE),
        re.compile(r"<iframe[^>]*>.*?</iframe>", re.IGNORECASE | re.DOTALL),
    ]
    
    @classmethod
    def validate_email(cls, email: str) -> bool:
        """
        Validate email address format.
        
        Args:
            email: Email address to validate
            
        Returns:
            True if valid email format
        """
        if not email or len(email) > 254:
            return False
        return bool(cls.EMAIL_PATTERN.match(email))
    
    @classmethod
    def validate_username(cls, username: str) -> bool:
        """
        Validate username format.
        
        Args:
            username: Username to validate
            
        Returns:
            True if valid username format
        """
        if not username:
            return False
        return bool(cls.USERNAME_PATTERN.match(username))
    
    @classmethod
    def validate_filename(cls, filename: str) -> bool:
        """
        Validate filename for security.
        
        Args:
            filename: Filename to validate
            
        Returns:
            True if safe filename
        """
        if not filename or len(filename) > 255:
            return False
        
        # Check for path traversal attempts
        if '..' in filename or '/' in filename or '\\' in filename:
            return False
        
        return bool(cls.FILENAME_PATTERN.match(filename))
    
    @classmethod
    def sanitize_string(cls, text: str, max_length: int = 1000) -> str:
        """
        Sanitize string input to prevent XSS and other attacks.
        
        Args:
            text: Text to sanitize
            max_length: Maximum allowed length
            
        Returns:
            Sanitized text
        """
        if not text:
            return ""
        
        # Truncate if too long
        if len(text) > max_length:
            text = text[:max_length]
        
        # HTML escape
        text = html.escape(text)
        
        # Remove potential XSS patterns
        for pattern in cls.XSS_PATTERNS:
            text = pattern.sub('', text)
        
        return text.strip()
    
    @classmethod
    def check_sql_injection(cls, text: str) -> bool:
        """
        Check if text contains potential SQL injection patterns.
        
        Args:
            text: Text to check
            
        Returns:
            True if potential SQL injection detected
        """
        if not text:
            return False
        
        for pattern in cls.SQL_INJECTION_PATTERNS:
            if pattern.search(text):
                return True
        
        return False
    
    @classmethod
    def validate_json_structure(cls, data: Dict[str, Any], required_fields: List[str], 
                               optional_fields: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Validate JSON structure and sanitize string fields.
        
        Args:
            data: JSON data to validate
            required_fields: List of required field names
            optional_fields: List of optional field names
            
        Returns:
            Validated and sanitized data
            
        Raises:
            ValueError: If validation fails
        """
        if not isinstance(data, dict):
            raise ValueError("Data must be a dictionary")
        
        # Check required fields
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")
        
        # Check for unexpected fields
        allowed_fields = set(required_fields + (optional_fields or []))
        unexpected_fields = set(data.keys()) - allowed_fields
        if unexpected_fields:
            raise ValueError(f"Unexpected fields: {', '.join(unexpected_fields)}")
        
        # Sanitize string fields
        sanitized_data = {}
        for key, value in data.items():
            if isinstance(value, str):
                # Check for SQL injection
                if cls.check_sql_injection(value):
                    raise ValueError(f"Potential SQL injection detected in field: {key}")
                
                # Sanitize string
                sanitized_data[key] = cls.sanitize_string(value)
            else:
                sanitized_data[key] = value
        
        return sanitized_data
    
    @classmethod
    def validate_file_upload(cls, file_content: bytes, filename: str, 
                           allowed_types: List[str]) -> Dict[str, Any]:
        """
        Validate uploaded file for security.
        
        Args:
            file_content: File content bytes
            filename: Original filename
            allowed_types: List of allowed MIME types
            
        Returns:
            Validation result with file info
            
        Raises:
            ValueError: If file validation fails
        """
        if not file_content:
            raise ValueError("File content is empty")
        
        if not cls.validate_filename(filename):
            raise ValueError("Invalid filename")
        
        # Check file size (max 16MB)
        max_size = 16 * 1024 * 1024
        if len(file_content) > max_size:
            raise ValueError(f"File too large. Maximum size: {max_size} bytes")
        
        # Detect MIME type from content
        try:
            mime_type = magic.from_buffer(file_content, mime=True)
        except Exception:
            # Fallback to filename-based detection
            mime_type, _ = mimetypes.guess_type(filename)
        
        if not mime_type or mime_type not in allowed_types:
            raise ValueError(f"File type not allowed. Allowed types: {', '.join(allowed_types)}")
        
        # Check for embedded executables or scripts
        dangerous_signatures = [
            b'MZ',  # PE executable
            b'\x7fELF',  # ELF executable
            b'#!/bin/',  # Shell script
            b'<script',  # JavaScript
            b'<?php',  # PHP script
        ]
        
        for signature in dangerous_signatures:
            if signature in file_content[:1024]:  # Check first 1KB
                raise ValueError("File contains potentially dangerous content")
        
        return {
            'filename': filename,
            'mime_type': mime_type,
            'size': len(file_content),
            'is_safe': True
        }


class DataEncryption:
    """Data encryption utilities for sensitive information."""
    
    @staticmethod
    def generate_key() -> bytes:
        """Generate a new encryption key."""
        return secrets.token_bytes(32)  # 256-bit key
    
    @staticmethod
    def encrypt_data(data: str, key: bytes) -> str:
        """
        Encrypt sensitive data.
        
        Args:
            data: Data to encrypt
            key: Encryption key
            
        Returns:
            Encrypted data as hex string
        """
        try:
            from cryptography.fernet import Fernet
            import base64
            
            # Create Fernet key from our key
            fernet_key = base64.urlsafe_b64encode(key)
            cipher = Fernet(fernet_key)
            
            # Encrypt data
            encrypted_data = cipher.encrypt(data.encode('utf-8'))
            return encrypted_data.hex()
            
        except ImportError:
            logger.warning("Cryptography library not available, using basic encoding")
            # Fallback to base64 (not secure, just for development)
            import base64
            return base64.b64encode(data.encode('utf-8')).decode('utf-8')
    
    @staticmethod
    def decrypt_data(encrypted_data: str, key: bytes) -> str:
        """
        Decrypt sensitive data.
        
        Args:
            encrypted_data: Encrypted data as hex string
            key: Encryption key
            
        Returns:
            Decrypted data
        """
        try:
            from cryptography.fernet import Fernet
            import base64
            
            # Create Fernet key from our key
            fernet_key = base64.urlsafe_b64encode(key)
            cipher = Fernet(fernet_key)
            
            # Decrypt data
            encrypted_bytes = bytes.fromhex(encrypted_data)
            decrypted_data = cipher.decrypt(encrypted_bytes)
            return decrypted_data.decode('utf-8')
            
        except ImportError:
            logger.warning("Cryptography library not available, using basic decoding")
            # Fallback from base64
            import base64
            return base64.b64decode(encrypted_data.encode('utf-8')).decode('utf-8')
    
    @staticmethod
    def hash_sensitive_data(data: str, salt: Optional[str] = None) -> str:
        """
        Hash sensitive data for storage.
        
        Args:
            data: Data to hash
            salt: Optional salt (generated if not provided)
            
        Returns:
            Salted hash as hex string
        """
        if salt is None:
            salt = secrets.token_hex(32)
        
        # Use PBKDF2 for key derivation
        hash_obj = hashlib.pbkdf2_hmac(
            'sha256',
            data.encode('utf-8'),
            salt.encode('utf-8'),
            100000  # 100k iterations
        )
        
        return f"{salt}:{hash_obj.hex()}"
    
    @staticmethod
    def verify_hash(data: str, stored_hash: str) -> bool:
        """
        Verify data against stored hash.
        
        Args:
            data: Data to verify
            stored_hash: Stored hash with salt
            
        Returns:
            True if data matches hash
        """
        try:
            salt, hash_hex = stored_hash.split(':', 1)
            
            # Compute hash with same salt
            computed_hash = hashlib.pbkdf2_hmac(
                'sha256',
                data.encode('utf-8'),
                salt.encode('utf-8'),
                100000
            )
            
            return computed_hash.hex() == hash_hex
            
        except (ValueError, AttributeError):
            return False


class AuditLogger:
    """Audit logging for sensitive operations."""
    
    def __init__(self, log_file: str = "logs/audit.log"):
        """
        Initialize audit logger.
        
        Args:
            log_file: Path to audit log file
        """
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(exist_ok=True)
        
        # Setup dedicated audit logger
        self.logger = logging.getLogger('audit')
        self.logger.setLevel(logging.INFO)
        
        # Create file handler if not exists
        if not self.logger.handlers:
            handler = logging.FileHandler(self.log_file)
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def log_authentication(self, user_id: str, username: str, success: bool, 
                          ip_address: str, user_agent: str = None):
        """
        Log authentication attempt.
        
        Args:
            user_id: User ID
            username: Username
            success: Whether authentication succeeded
            ip_address: Client IP address
            user_agent: Client user agent
        """
        self.logger.info(
            f"AUTH {'SUCCESS' if success else 'FAILURE'} - "
            f"user_id={user_id} username={username} ip={ip_address} "
            f"user_agent={user_agent or 'unknown'}"
        )
    
    def log_permission_check(self, user_id: str, permission: str, granted: bool):
        """
        Log permission check.
        
        Args:
            user_id: User ID
            permission: Permission being checked
            granted: Whether permission was granted
        """
        self.logger.info(
            f"PERMISSION {'GRANTED' if granted else 'DENIED'} - "
            f"user_id={user_id} permission={permission}"
        )
    
    def log_data_access(self, user_id: str, resource_type: str, resource_id: str, 
                       action: str):
        """
        Log data access.
        
        Args:
            user_id: User ID
            resource_type: Type of resource accessed
            resource_id: Resource identifier
            action: Action performed (read, create, update, delete)
        """
        self.logger.info(
            f"DATA_ACCESS - user_id={user_id} resource_type={resource_type} "
            f"resource_id={resource_id} action={action}"
        )
    
    def log_security_event(self, event_type: str, user_id: str = None, 
                          details: str = None, severity: str = "INFO"):
        """
        Log security event.
        
        Args:
            event_type: Type of security event
            user_id: User ID (if applicable)
            details: Additional details
            severity: Event severity (INFO, WARNING, ERROR, CRITICAL)
        """
        log_method = getattr(self.logger, severity.lower(), self.logger.info)
        log_method(
            f"SECURITY_EVENT - type={event_type} user_id={user_id or 'system'} "
            f"details={details or 'none'}"
        )
    
    def log_api_key_usage(self, api_key_id: str, user_id: str, endpoint: str, 
                         success: bool):
        """
        Log API key usage.
        
        Args:
            api_key_id: API key ID
            user_id: User ID
            endpoint: API endpoint accessed
            success: Whether request succeeded
        """
        self.logger.info(
            f"API_KEY_USAGE {'SUCCESS' if success else 'FAILURE'} - "
            f"api_key_id={api_key_id} user_id={user_id} endpoint={endpoint}"
        )


class SecureFileHandler:
    """Secure file upload and handling utilities."""
    
    ALLOWED_UPLOAD_TYPES = {
        'dataset': ['text/csv', 'application/json', 'text/plain'],
        'model': ['application/octet-stream', 'application/zip'],
        'document': ['text/plain', 'text/markdown', 'application/pdf']
    }
    
    def __init__(self, upload_dir: str = "uploads"):
        """
        Initialize secure file handler.
        
        Args:
            upload_dir: Directory for file uploads
        """
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for different file types
        for file_type in self.ALLOWED_UPLOAD_TYPES:
            (self.upload_dir / file_type).mkdir(exist_ok=True)
    
    def save_uploaded_file(self, file_content: bytes, filename: str, 
                          file_type: str, user_id: str) -> Dict[str, Any]:
        """
        Securely save uploaded file.
        
        Args:
            file_content: File content bytes
            filename: Original filename
            file_type: Type of file (dataset, model, document)
            user_id: User ID for audit trail
            
        Returns:
            File information dictionary
            
        Raises:
            ValueError: If file validation fails
        """
        # Validate file type
        if file_type not in self.ALLOWED_UPLOAD_TYPES:
            raise ValueError(f"Invalid file type: {file_type}")
        
        allowed_types = self.ALLOWED_UPLOAD_TYPES[file_type]
        
        # Validate file
        file_info = InputValidator.validate_file_upload(
            file_content, filename, allowed_types
        )
        
        # Generate secure filename
        file_extension = Path(filename).suffix
        secure_filename = f"{user_id}_{secrets.token_hex(16)}{file_extension}"
        
        # Save file
        file_path = self.upload_dir / file_type / secure_filename
        
        try:
            with open(file_path, 'wb') as f:
                f.write(file_content)
            
            # Set restrictive permissions (owner read/write only)
            file_path.chmod(0o600)
            
            # Log file upload
            audit_logger = AuditLogger()
            audit_logger.log_data_access(
                user_id=user_id,
                resource_type="file",
                resource_id=secure_filename,
                action="upload"
            )
            
            return {
                'original_filename': filename,
                'secure_filename': secure_filename,
                'file_path': str(file_path),
                'file_type': file_type,
                'size': len(file_content),
                'mime_type': file_info['mime_type'],
                'uploaded_at': datetime.utcnow().isoformat(),
                'uploaded_by': user_id
            }
            
        except Exception as e:
            logger.error(f"Failed to save file: {e}")
            raise ValueError(f"Failed to save file: {e}")
    
    def delete_file(self, secure_filename: str, file_type: str, user_id: str) -> bool:
        """
        Securely delete file.
        
        Args:
            secure_filename: Secure filename
            file_type: Type of file
            user_id: User ID for audit trail
            
        Returns:
            True if file was deleted
        """
        file_path = self.upload_dir / file_type / secure_filename
        
        try:
            if file_path.exists():
                file_path.unlink()
                
                # Log file deletion
                audit_logger = AuditLogger()
                audit_logger.log_data_access(
                    user_id=user_id,
                    resource_type="file",
                    resource_id=secure_filename,
                    action="delete"
                )
                
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete file: {e}")
            return False
    
    def scan_file_for_viruses(self, file_path: Path) -> bool:
        """
        Scan file for viruses (placeholder for actual antivirus integration).
        
        Args:
            file_path: Path to file to scan
            
        Returns:
            True if file is clean
        """
        # In production, integrate with ClamAV or similar
        # For now, just check for basic malicious patterns
        
        try:
            with open(file_path, 'rb') as f:
                content = f.read(1024)  # Read first 1KB
            
            # Check for executable signatures
            malicious_signatures = [
                b'MZ',  # PE executable
                b'\x7fELF',  # ELF executable
                b'PK\x03\x04',  # ZIP file (could contain executables)
            ]
            
            for signature in malicious_signatures:
                if content.startswith(signature):
                    logger.warning(f"Potentially malicious file detected: {file_path}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error scanning file: {e}")
            return False


# Global instances
audit_logger = AuditLogger()
secure_file_handler = SecureFileHandler()