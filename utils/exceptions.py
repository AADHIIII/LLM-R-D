"""
Custom exceptions for LLM optimization platform.
Provides structured error handling with specific exception types.
"""

from typing import Optional, Dict, Any, List


class LLMOptimizationError(Exception):
    """Base exception for LLM optimization platform."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        suggested_actions: Optional[List[str]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        self.suggested_actions = suggested_actions or []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
            "suggested_actions": self.suggested_actions,
        }


class ValidationError(LLMOptimizationError):
    """Raised when validation fails."""
    pass


class DataFormatError(LLMOptimizationError):
    """Raised when data format is invalid."""
    pass


class DatasetValidationError(LLMOptimizationError):
    """Raised when dataset validation fails."""
    
    def __init__(
        self,
        message: str,
        invalid_fields: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.invalid_fields = invalid_fields or []
        if self.invalid_fields:
            self.details["invalid_fields"] = self.invalid_fields


class ModelLoadingError(LLMOptimizationError):
    """Raised when model loading fails."""
    
    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.model_name = model_name
        if self.model_name:
            self.details["model_name"] = self.model_name


class TrainingError(LLMOptimizationError):
    """Raised when training fails."""
    
    def __init__(
        self,
        message: str,
        training_step: Optional[int] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.training_step = training_step
        if self.training_step is not None:
            self.details["training_step"] = self.training_step


class APIError(LLMOptimizationError):
    """Raised when API calls fail."""
    
    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        api_provider: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.status_code = status_code
        self.api_provider = api_provider
        if self.status_code:
            self.details["status_code"] = self.status_code
        if self.api_provider:
            self.details["api_provider"] = self.api_provider


class EvaluationError(LLMOptimizationError):
    """Raised when evaluation fails."""
    
    def __init__(
        self,
        message: str,
        metric_name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.metric_name = metric_name
        if self.metric_name:
            self.details["metric_name"] = self.metric_name


class ConfigurationError(LLMOptimizationError):
    """Raised when configuration is invalid."""
    
    def __init__(
        self,
        message: str,
        config_field: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.config_field = config_field
        if self.config_field:
            self.details["config_field"] = self.config_field


class ResourceError(LLMOptimizationError):
    """Raised when resource constraints are exceeded."""
    
    def __init__(
        self,
        message: str,
        resource_type: Optional[str] = None,
        current_usage: Optional[float] = None,
        limit: Optional[float] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.resource_type = resource_type
        self.current_usage = current_usage
        self.limit = limit
        
        if self.resource_type:
            self.details["resource_type"] = self.resource_type
        if self.current_usage is not None:
            self.details["current_usage"] = self.current_usage
        if self.limit is not None:
            self.details["limit"] = self.limit


class DatabaseError(LLMOptimizationError):
    """Raised when database operations fail."""
    
    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        table_name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.operation = operation
        self.table_name = table_name
        
        if self.operation:
            self.details["operation"] = self.operation
        if self.table_name:
            self.details["table_name"] = self.table_name