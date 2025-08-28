"""
Logging utilities for LLM optimization platform.
Provides structured logging with proper formatting and handlers.
"""

import json
import logging
import logging.handlers
import os
import sys
import traceback
from datetime import datetime
from typing import Optional, Dict, Any, Union
import uuid

from config.settings import get_settings


class JSONFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'thread': record.thread,
            'process': record.process,
        }
        
        # Add exception information if present
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields from record
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'lineno', 'funcName', 'created',
                          'msecs', 'relativeCreated', 'thread', 'threadName',
                          'processName', 'process', 'getMessage', 'exc_info',
                          'exc_text', 'stack_info']:
                log_entry[key] = value
        
        return json.dumps(log_entry, default=str)


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output."""
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        # Add color to levelname
        if record.levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[record.levelname]}"
                f"{record.levelname}"
                f"{self.COLORS['RESET']}"
            )
        
        return super().format(record)


def setup_logging(
    name: Optional[str] = None,
    level: Optional[str] = None,
    log_file: Optional[str] = None,
    console_output: bool = True,
    file_output: bool = True,
    json_format: bool = False,
) -> logging.Logger:
    """
    Set up logging configuration for the application.
    
    Args:
        name: Logger name (defaults to root logger)
        level: Logging level (defaults to settings.log_level)
        log_file: Log file path (defaults to settings.log_file)
        console_output: Whether to output to console
        file_output: Whether to output to file
        json_format: Whether to use JSON formatting for structured logs
    
    Returns:
        Configured logger instance
    """
    settings = get_settings()
    
    # Use provided values or defaults from settings
    level = level or settings.log_level
    log_file = log_file or settings.log_file
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatters
    if json_format:
        structured_formatter = JSONFormatter()
        console_formatter = JSONFormatter()
    else:
        structured_formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_formatter = ColoredFormatter(
            fmt='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if file_output and log_file:
        # Ensure log directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        # Rotating file handler to prevent large log files
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(structured_formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Logger name
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class StructuredLogger:
    """Enhanced logger with structured logging capabilities."""
    
    def __init__(self, name: str):
        self.logger = get_logger(name)
        self.context = {}
    
    def set_context(self, **kwargs) -> None:
        """Set context fields that will be included in all log messages."""
        self.context.update(kwargs)
    
    def clear_context(self) -> None:
        """Clear all context fields."""
        self.context.clear()
    
    def _log_with_context(self, level: int, message: str, **kwargs) -> None:
        """Log message with context and additional fields."""
        extra = {**self.context, **kwargs}
        self.logger.log(level, message, extra=extra)
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message with context."""
        self._log_with_context(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message with context."""
        self._log_with_context(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message with context."""
        self._log_with_context(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        """Log error message with context."""
        self._log_with_context(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs) -> None:
        """Log critical message with context."""
        self._log_with_context(logging.CRITICAL, message, **kwargs)
    
    def exception(self, message: str, **kwargs) -> None:
        """Log exception with full traceback and context."""
        extra = {**self.context, **kwargs}
        self.logger.exception(message, extra=extra)


class LoggerMixin:
    """Mixin class to add logging capabilities to any class."""
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger instance for this class."""
        return get_logger(self.__class__.__name__)
    
    @property
    def structured_logger(self) -> StructuredLogger:
        """Get structured logger instance for this class."""
        return StructuredLogger(self.__class__.__name__)


def log_function_call(func):
    """Decorator to log function calls with parameters and execution time."""
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = datetime.utcnow()
        
        # Log function entry
        logger.debug(
            f"Calling {func.__name__}",
            extra={
                'function': func.__name__,
                'module': func.__module__,
                'args_count': len(args),
                'kwargs_keys': list(kwargs.keys()),
                'start_time': start_time.isoformat()
            }
        )
        
        try:
            result = func(*args, **kwargs)
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            
            # Log successful completion
            logger.debug(
                f"Completed {func.__name__}",
                extra={
                    'function': func.__name__,
                    'module': func.__module__,
                    'duration_seconds': duration,
                    'success': True
                }
            )
            
            return result
            
        except Exception as e:
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            
            # Log exception
            logger.error(
                f"Error in {func.__name__}: {str(e)}",
                extra={
                    'function': func.__name__,
                    'module': func.__module__,
                    'duration_seconds': duration,
                    'success': False,
                    'error_type': type(e).__name__,
                    'error_message': str(e)
                },
                exc_info=True
            )
            raise
    
    return wrapper


def get_structured_logger(name: str) -> StructuredLogger:
    """
    Get a structured logger instance with the specified name.
    
    Args:
        name: Logger name
    
    Returns:
        StructuredLogger instance
    """
    return StructuredLogger(name)


def log_error_with_context(
    logger: Union[logging.Logger, StructuredLogger],
    error: Exception,
    context: Optional[Dict[str, Any]] = None,
    message: Optional[str] = None
) -> None:
    """
    Log error with full context and traceback.
    
    Args:
        logger: Logger instance
        error: Exception to log
        context: Additional context information
        message: Custom error message
    """
    error_message = message or f"Error occurred: {str(error)}"
    
    error_context = {
        'error_type': type(error).__name__,
        'error_message': str(error),
        'traceback': traceback.format_exc()
    }
    
    if context:
        error_context.update(context)
    
    if isinstance(logger, StructuredLogger):
        logger.error(error_message, **error_context)
    else:
        logger.error(error_message, extra=error_context, exc_info=True)


# Set up root logger on import
setup_logging(json_format=os.getenv('LOG_FORMAT', '').lower() == 'json')