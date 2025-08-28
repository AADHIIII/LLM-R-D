"""
Security middleware for API endpoints.
"""
from flask import Flask, request, jsonify
from typing import Any
import time
from collections import defaultdict, deque


# Simple in-memory rate limiting (for production, use Redis)
request_counts = defaultdict(lambda: deque())


def setup_security_middleware(app: Flask) -> None:
    """
    Setup security middleware including rate limiting and security headers.
    
    Args:
        app: Flask application instance
    """
    
    @app.before_request
    def rate_limit() -> Any:
        """Apply rate limiting to requests."""
        if not app.config.get('RATE_LIMIT_ENABLED', True):
            return None
        
        client_ip = request.remote_addr
        current_time = time.time()
        rate_limit_per_minute = app.config.get('RATE_LIMIT_PER_MINUTE', 60)
        
        # Clean old requests (older than 1 minute)
        minute_ago = current_time - 60
        while (request_counts[client_ip] and 
               request_counts[client_ip][0] < minute_ago):
            request_counts[client_ip].popleft()
        
        # Check if rate limit exceeded
        if len(request_counts[client_ip]) >= rate_limit_per_minute:
            return jsonify({
                'error': 'rate_limit_exceeded',
                'message': f'Rate limit exceeded. Maximum {rate_limit_per_minute} requests per minute.',
                'retry_after': 60
            }), 429
        
        # Add current request
        request_counts[client_ip].append(current_time)
        return None
    
    @app.after_request
    def add_security_headers(response: Any) -> Any:
        """Add security headers to all responses."""
        # Prevent clickjacking
        response.headers['X-Frame-Options'] = 'DENY'
        
        # Prevent MIME type sniffing
        response.headers['X-Content-Type-Options'] = 'nosniff'
        
        # Enable XSS protection
        response.headers['X-XSS-Protection'] = '1; mode=block'
        
        # Strict transport security (HTTPS only)
        if request.is_secure:
            response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
        
        # Content Security Policy
        response.headers['Content-Security-Policy'] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self' https:; "
            "connect-src 'self' https:; "
            "frame-ancestors 'none';"
        )
        
        # Referrer policy
        response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
        
        return response