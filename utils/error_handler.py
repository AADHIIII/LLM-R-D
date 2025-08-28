"""
Error handling utilities for LLM optimization platform.
Provides centralized error handling and recovery mechanisms.
"""

import traceback
from typing import Optional, Dict, Any, Callable
from functools import wraps

from .exceptions import LLMOptimizationError
from .logging import get_logger

logger = get_logger(__name__)


def handle_errors(
    default_return=None,
    reraise: bool = True,
    log_error: bool = True,
) -> Callable:
    """
    Decorator for handling errors in functions.
    
    Args:
        default_return: Default value to return on error
        reraise: Whether to reraise the exception
        log_error: Whether to log the error
    
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_error:
                    logger.error(
                        f"Error in {func.__name__}: {str(e)}",
                        extra={
                            "function": func.__name__,
                            "args": str(args)[:200],  # Limit arg length
                            "kwargs": str(kwargs)[:200],
                            "traceback": traceback.format_exc(),
                        }
                    )
                
                if reraise:
                    raise
                
                return default_return
        
        return wrapper
    return decorator


class ErrorHandler:
    """Centralized error handler for the application."""
    
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
    
    def handle_exception(
        self,
        exception: Exception,
        context: Optional[Dict[str, Any]] = None,
        user_message: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Handle an exception and return structured error response.
        
        Args:
            exception: The exception to handle
            context: Additional context information
            user_message: User-friendly error message
        
        Returns:
            Structured error response
        """
        context = context or {}
        
        # Log the error
        self.logger.error(
            f"Exception handled: {str(exception)}",
            extra={
                "exception_type": type(exception).__name__,
                "context": context,
                "traceback": traceback.format_exc(),
            }
        )
        
        # Handle custom exceptions
        if isinstance(exception, LLMOptimizationError):
            return exception.to_dict()
        
        # Handle common exceptions
        error_response = self._handle_common_exceptions(exception, context)
        
        # Add user message if provided
        if user_message:
            error_response["user_message"] = user_message
        
        return error_response
    
    def _handle_common_exceptions(
        self,
        exception: Exception,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Handle common Python exceptions."""
        
        if isinstance(exception, FileNotFoundError):
            return {
                "error_code": "FILE_NOT_FOUND",
                "message": f"File not found: {str(exception)}",
                "details": {"file_path": str(exception)},
                "suggested_actions": [
                    "Check if the file path is correct",
                    "Ensure the file exists",
                    "Verify file permissions",
                ],
            }
        
        elif isinstance(exception, PermissionError):
            return {
                "error_code": "PERMISSION_DENIED",
                "message": f"Permission denied: {str(exception)}",
                "details": {"path": str(exception)},
                "suggested_actions": [
                    "Check file/directory permissions",
                    "Run with appropriate privileges",
                    "Verify ownership of files",
                ],
            }
        
        elif isinstance(exception, ValueError):
            return {
                "error_code": "INVALID_VALUE",
                "message": f"Invalid value: {str(exception)}",
                "details": {"value_error": str(exception)},
                "suggested_actions": [
                    "Check input values",
                    "Verify data types",
                    "Review parameter ranges",
                ],
            }
        
        elif isinstance(exception, KeyError):
            return {
                "error_code": "MISSING_KEY",
                "message": f"Missing required key: {str(exception)}",
                "details": {"missing_key": str(exception)},
                "suggested_actions": [
                    "Check required parameters",
                    "Verify configuration keys",
                    "Review API documentation",
                ],
            }
        
        elif isinstance(exception, ImportError):
            return {
                "error_code": "IMPORT_ERROR",
                "message": f"Import error: {str(exception)}",
                "details": {"import_error": str(exception)},
                "suggested_actions": [
                    "Install missing dependencies",
                    "Check Python environment",
                    "Verify package versions",
                ],
            }
        
        else:
            # Generic error handling
            return {
                "error_code": "INTERNAL_ERROR",
                "message": f"An unexpected error occurred: {str(exception)}",
                "details": {
                    "exception_type": type(exception).__name__,
                    "context": context,
                },
                "suggested_actions": [
                    "Check application logs",
                    "Retry the operation",
                    "Contact support if the issue persists",
                ],
            }


# Global error handler instance
error_handler = ErrorHandler()


def get_error_handler() -> ErrorHandler:
    """Get the global error handler instance."""
    return error_handler


def handle_api_error(exception: Exception, status_code: int = 500) -> tuple:
    """
    Handle API errors and return Flask-compatible response.
    
    Args:
        exception: The exception to handle
        status_code: HTTP status code to return
    
    Returns:
        Tuple of (response_dict, status_code)
    """
    from flask import jsonify
    from datetime import datetime
    
    error_response = error_handler.handle_exception(exception)
    
    # Add timestamp and status code
    error_response['timestamp'] = datetime.utcnow().isoformat()
    error_response['status_code'] = status_code
    
    return jsonify(error_response), status_code