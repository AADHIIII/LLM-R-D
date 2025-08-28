"""
Flask application configuration classes.
"""
import os
from typing import List


class BaseConfig:
    """Base configuration class with common settings."""
    
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    
    # API Configuration
    API_VERSION = 'v1'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file upload
    
    # Model Configuration
    MODEL_CACHE_SIZE = 3  # Number of models to keep in memory
    MODEL_TIMEOUT = 30  # Seconds before model request timeout
    
    # Commercial API Configuration
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY')
    
    # Rate Limiting
    RATE_LIMIT_ENABLED = True
    RATE_LIMIT_PER_MINUTE = 60
    
    # CORS Configuration
    CORS_ORIGINS = ['http://localhost:3000', 'http://localhost:8501']
    
    # Authentication Configuration
    JWT_ACCESS_TOKEN_EXPIRES = 3600  # 1 hour in seconds
    JWT_REFRESH_TOKEN_EXPIRES = 2592000  # 30 days in seconds
    JWT_ALGORITHM = 'HS256'
    
    # Security Configuration
    PASSWORD_MIN_LENGTH = 8
    MAX_LOGIN_ATTEMPTS = 5
    ACCOUNT_LOCKOUT_DURATION = 1800  # 30 minutes in seconds
    
    # Logging
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s %(levelname)s %(name)s: %(message)s'


class DevelopmentConfig(BaseConfig):
    """Development environment configuration."""
    
    DEBUG = True
    TESTING = False
    
    # More permissive settings for development
    RATE_LIMIT_PER_MINUTE = 120
    LOG_LEVEL = 'DEBUG'
    
    # Allow all origins in development
    CORS_ORIGINS = ['*']


class TestingConfig(BaseConfig):
    """Testing environment configuration."""
    
    DEBUG = False
    TESTING = True
    
    # Disable rate limiting for tests
    RATE_LIMIT_ENABLED = False
    
    # Use in-memory storage for tests
    MODEL_CACHE_SIZE = 1
    
    # Mock API keys for testing
    OPENAI_API_KEY = 'test-openai-key'
    ANTHROPIC_API_KEY = 'test-anthropic-key'
    
    # Skip authentication in tests
    SKIP_AUTH = True
    
    # Use test secret key
    SECRET_KEY = 'test-secret-key-for-jwt'


class ProductionConfig(BaseConfig):
    """Production environment configuration."""
    
    DEBUG = False
    TESTING = False
    
    # Stricter settings for production
    RATE_LIMIT_PER_MINUTE = 30
    MODEL_TIMEOUT = 60
    
    # Production CORS origins should be explicitly set
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', '').split(',') if os.environ.get('CORS_ORIGINS') else []


# Configuration mapping
config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}