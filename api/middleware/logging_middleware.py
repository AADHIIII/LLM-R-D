"""
Request/response logging middleware with structured logging and metrics collection.
"""
import time
import logging
import json
from flask import Flask, request, g, current_app
from typing import Any, Dict, Optional
import uuid
from datetime import datetime

from utils.logging import get_structured_logger, log_error_with_context
from monitoring.metrics_collector import get_metrics_collector, APIMetrics


class RequestLogger:
    """Enhanced request logger with structured logging capabilities."""
    
    def __init__(self, app: Flask):
        self.app = app
        self.logger = get_structured_logger('api.requests')
        
    def log_request_start(self) -> None:
        """Log incoming request with structured data."""
        request_data = {
            'request_id': g.request_id,
            'method': request.method,
            'path': request.path,
            'url': request.url,
            'remote_addr': request.remote_addr,
            'user_agent': request.headers.get('User-Agent', 'Unknown'),
            'content_type': request.headers.get('Content-Type'),
            'content_length': request.headers.get('Content-Length'),
            'timestamp': datetime.utcnow().isoformat(),
            'query_params': dict(request.args) if request.args else None
        }
        
        # Add request headers (excluding sensitive ones)
        safe_headers = {
            k: v for k, v in request.headers.items()
            if k.lower() not in ['authorization', 'cookie', 'x-api-key']
        }
        request_data['headers'] = safe_headers
        
        self.logger.info("Request started", **request_data)
        
        # Log request body for POST/PUT requests (excluding sensitive data)
        if request.method in ['POST', 'PUT'] and request.is_json:
            try:
                data = request.get_json()
                if data:
                    # Remove sensitive fields before logging
                    safe_data = self._sanitize_request_data(data)
                    self.logger.debug(
                        "Request body",
                        request_id=g.request_id,
                        body=safe_data,
                        body_size=len(str(data))
                    )
            except Exception as e:
                self.logger.warning(
                    "Failed to parse request body",
                    request_id=g.request_id,
                    error=str(e)
                )
    
    def log_request_end(self, response: Any) -> None:
        """Log request completion with response details and record metrics."""
        duration = time.time() - g.start_time
        
        response_data = {
            'request_id': g.request_id,
            'method': request.method,
            'path': request.path,
            'status_code': response.status_code,
            'duration_ms': round(duration * 1000, 2),
            'response_size': response.content_length,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Determine log level based on status code
        if response.status_code >= 500:
            log_level = 'error'
        elif response.status_code >= 400:
            log_level = 'warning'
        else:
            log_level = 'info'
        
        getattr(self.logger, log_level)("Request completed", **response_data)
        
        # Log slow requests
        if duration > 5.0:  # 5 seconds threshold
            self.logger.warning(
                "Slow request detected",
                request_id=g.request_id,
                duration_ms=round(duration * 1000, 2),
                threshold_ms=5000
            )
        
        # Record API metrics
        try:
            metrics_collector = get_metrics_collector()
            api_metrics = APIMetrics(
                timestamp=datetime.utcnow(),
                endpoint=request.path,
                method=request.method,
                status_code=response.status_code,
                response_time_ms=round(duration * 1000, 2),
                request_size_bytes=request.content_length or 0,
                response_size_bytes=response.content_length or 0,
                error_count=1 if response.status_code >= 400 else 0
            )
            metrics_collector.record_api_metrics(api_metrics)
        except Exception as e:
            # Don't let metrics collection errors break the response
            self.logger.warning(
                "Failed to record API metrics",
                request_id=g.request_id,
                error=str(e)
            )
    
    def log_request_error(self, error: Exception) -> None:
        """Log request error with full context."""
        error_data = {
            'request_id': g.request_id,
            'method': request.method,
            'path': request.path,
            'duration_ms': round((time.time() - g.start_time) * 1000, 2),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        log_error_with_context(
            self.logger,
            error,
            context=error_data,
            message="Request processing error"
        )
    
    def _sanitize_request_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Remove sensitive fields from request data."""
        sensitive_fields = {
            'password', 'api_key', 'token', 'secret', 'authorization',
            'credit_card', 'ssn', 'social_security', 'private_key'
        }
        
        if isinstance(data, dict):
            return {
                k: self._sanitize_request_data(v) if isinstance(v, (dict, list))
                else '[REDACTED]' if k.lower() in sensitive_fields
                else v
                for k, v in data.items()
            }
        elif isinstance(data, list):
            return [self._sanitize_request_data(item) for item in data]
        else:
            return data


def setup_logging_middleware(app: Flask) -> None:
    """
    Setup enhanced request/response logging middleware.
    
    Args:
        app: Flask application instance
    """
    request_logger = RequestLogger(app)
    
    @app.before_request
    def before_request() -> None:
        """Set up request context and log incoming request."""
        g.start_time = time.time()
        g.request_id = str(uuid.uuid4())[:8]
        
        # Set request context for structured logging
        request_logger.logger.set_context(
            request_id=g.request_id,
            method=request.method,
            path=request.path
        )
        
        request_logger.log_request_start()
    
    @app.after_request
    def after_request(response: Any) -> Any:
        """Log response details and clean up context."""
        try:
            request_logger.log_request_end(response)
        except Exception as e:
            # Don't let logging errors break the response
            current_app.logger.error(f"Error in response logging: {e}")
        
        # Add request ID to response headers for tracing
        response.headers['X-Request-ID'] = g.request_id
        
        # Clear request context
        request_logger.logger.clear_context()
        
        return response
    
    @app.errorhandler(Exception)
    def log_unhandled_exception(error: Exception) -> None:
        """Log unhandled exceptions with request context."""
        if hasattr(g, 'request_id'):
            request_logger.log_request_error(error)
        else:
            # Fallback logging if request context is not available
            logger = get_structured_logger('api.errors')
            log_error_with_context(
                logger,
                error,
                message="Unhandled exception in Flask app"
            )
        
        # Re-raise the exception to let Flask handle it normally
        raise error