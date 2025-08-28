"""
Dataset validation module for LLM fine-tuning platform.

This module provides comprehensive validation for datasets in JSONL and CSV formats,
ensuring they contain proper prompt-response pairs and meet quality standards.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from enum import Enum
import logging

from utils.exceptions import ValidationError, DataFormatError


class DatasetFormat(Enum):
    """Supported dataset formats."""
    JSONL = "jsonl"
    CSV = "csv"


@dataclass
class ValidationResult:
    """Result of dataset validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    sample_count: int
    format: DatasetFormat
    schema_info: Dict[str, Any]


@dataclass
class PromptResponsePair:
    """Represents a prompt-response pair with optional metadata."""
    prompt: str
    response: str
    metadata: Optional[Dict[str, Any]] = None


class DatasetValidator:
    """
    Validates datasets for fine-tuning LLM models.
    
    Supports JSONL and CSV formats with comprehensive validation including:
    - Format validation
    - Schema validation for prompt-response pairs
    - Data quality checks
    - Missing value detection
    - Format consistency checks
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the dataset validator.
        
        Args:
            logger: Optional logger instance for validation messages
        """
        self.logger = logger or logging.getLogger(__name__)
        
        # Required fields for prompt-response pairs
        self.required_fields = {"prompt", "response"}
        
        # Optional fields that can be present
        self.optional_fields = {"metadata", "id", "category", "domain"}
        
        # Maximum lengths for validation
        self.max_prompt_length = 4096
        self.max_response_length = 4096
        
        # Minimum dataset size
        self.min_samples = 1
    
    def validate_dataset(self, dataset_path: Union[str, Path]) -> ValidationResult:
        """
        Validate a dataset file.
        
        Args:
            dataset_path: Path to the dataset file
            
        Returns:
            ValidationResult with validation status and details
            
        Raises:
            ValidationError: If the file cannot be read or format is unsupported
        """
        dataset_path = Path(dataset_path)
        
        if not dataset_path.exists():
            raise ValidationError(f"Dataset file not found: {dataset_path}")
        
        if not dataset_path.is_file():
            raise ValidationError(f"Path is not a file: {dataset_path}")
        
        # Determine format from file extension
        format_type = self._detect_format(dataset_path)
        
        self.logger.info(f"Validating {format_type.value} dataset: {dataset_path}")
        
        try:
            if format_type == DatasetFormat.JSONL:
                return self._validate_jsonl(dataset_path)
            elif format_type == DatasetFormat.CSV:
                return self._validate_csv(dataset_path)
            else:
                raise ValidationError(f"Unsupported format: {format_type}")
                
        except Exception as e:
            self.logger.error(f"Validation failed: {str(e)}")
            raise ValidationError(f"Dataset validation failed: {str(e)}")
    
    def _detect_format(self, dataset_path: Path) -> DatasetFormat:
        """Detect dataset format from file extension."""
        suffix = dataset_path.suffix.lower()
        
        if suffix == '.jsonl':
            return DatasetFormat.JSONL
        elif suffix == '.csv':
            return DatasetFormat.CSV
        else:
            raise ValidationError(f"Unsupported file extension: {suffix}")
    
    def _validate_jsonl(self, dataset_path: Path) -> ValidationResult:
        """Validate JSONL format dataset."""
        errors = []
        warnings = []
        samples = []
        
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        sample_errors = self._validate_sample(data, line_num)
                        errors.extend(sample_errors)
                        
                        if not sample_errors:  # Only add valid samples
                            samples.append(data)
                            
                    except json.JSONDecodeError as e:
                        errors.append(f"Line {line_num}: Invalid JSON - {str(e)}")
                        
        except UnicodeDecodeError:
            errors.append("File encoding error. Please ensure the file is UTF-8 encoded.")
        except Exception as e:
            errors.append(f"Error reading file: {str(e)}")
        
        # Check minimum sample count
        if len(samples) < self.min_samples:
            errors.append(f"Dataset must contain at least {self.min_samples} valid samples")
        
        # Generate schema info
        schema_info = self._analyze_schema(samples) if samples else {}
        
        # Add warnings for data quality issues
        warnings.extend(self._check_data_quality(samples))
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            sample_count=len(samples),
            format=DatasetFormat.JSONL,
            schema_info=schema_info
        )
    
    def _validate_csv(self, dataset_path: Path) -> ValidationResult:
        """Validate CSV format dataset."""
        errors = []
        warnings = []
        samples = []
        schema_info = {}
        
        try:
            # Try reading with different encodings
            df = None
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    df = pd.read_csv(dataset_path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                errors.append("Could not read CSV file with any supported encoding")
                return ValidationResult(
                    is_valid=False,
                    errors=errors,
                    warnings=warnings,
                    sample_count=0,
                    format=DatasetFormat.CSV,
                    schema_info={}
                )
            
            # Validate CSV structure
            if df.empty:
                errors.append("CSV file is empty")
                return ValidationResult(
                    is_valid=False,
                    errors=errors,
                    warnings=warnings,
                    sample_count=0,
                    format=DatasetFormat.CSV,
                    schema_info={}
                )
            
            # Check for required columns
            columns = set(df.columns.str.lower())
            missing_required = self.required_fields - columns
            
            if missing_required:
                errors.append(f"Missing required columns: {missing_required}")
            
            # Validate each row
            for idx, row in df.iterrows():
                row_dict = row.to_dict()
                sample_errors = self._validate_sample(row_dict, idx + 2)  # +2 for header and 0-indexing
                errors.extend(sample_errors)
                
                if not sample_errors:
                    samples.append(row_dict)
            
            # Check minimum sample count
            if len(samples) < self.min_samples:
                errors.append(f"Dataset must contain at least {self.min_samples} valid samples")
            
            # Generate schema info
            schema_info = self._analyze_schema(samples) if samples else {}
            
            # Add warnings for data quality issues
            warnings.extend(self._check_data_quality(samples))
            
        except Exception as e:
            errors.append(f"Error reading CSV file: {str(e)}")
            
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            sample_count=len(samples),
            format=DatasetFormat.CSV,
            schema_info=schema_info
        )   
 
    def _validate_sample(self, sample: Dict[str, Any], line_num: int) -> List[str]:
        """
        Validate a single sample (prompt-response pair).
        
        Args:
            sample: Dictionary containing sample data
            line_num: Line number for error reporting
            
        Returns:
            List of validation errors for this sample
        """
        errors = []
        
        # Check if sample is a dictionary
        if not isinstance(sample, dict):
            errors.append(f"Line {line_num}: Sample must be a dictionary/object")
            return errors
        
        # Normalize keys to lowercase for case-insensitive matching
        normalized_sample = {k.lower(): v for k, v in sample.items()}
        sample_keys = set(normalized_sample.keys())
        
        # Check for required fields
        missing_required = self.required_fields - sample_keys
        if missing_required:
            errors.append(f"Line {line_num}: Missing required fields: {missing_required}")
        
        # Validate prompt field
        if 'prompt' in normalized_sample:
            prompt = normalized_sample['prompt']
            if not isinstance(prompt, str):
                errors.append(f"Line {line_num}: 'prompt' must be a string")
            elif not prompt.strip():
                errors.append(f"Line {line_num}: 'prompt' cannot be empty")
            elif len(prompt) > self.max_prompt_length:
                errors.append(f"Line {line_num}: 'prompt' exceeds maximum length of {self.max_prompt_length}")
        
        # Validate response field
        if 'response' in normalized_sample:
            response = normalized_sample['response']
            if not isinstance(response, str):
                errors.append(f"Line {line_num}: 'response' must be a string")
            elif not response.strip():
                errors.append(f"Line {line_num}: 'response' cannot be empty")
            elif len(response) > self.max_response_length:
                errors.append(f"Line {line_num}: 'response' exceeds maximum length of {self.max_response_length}")
        
        # Validate metadata if present
        if 'metadata' in normalized_sample:
            metadata = normalized_sample['metadata']
            if metadata is not None and not isinstance(metadata, dict):
                errors.append(f"Line {line_num}: 'metadata' must be a dictionary or null")
        
        # Check for unexpected fields (warning level)
        all_allowed_fields = self.required_fields | self.optional_fields
        unexpected_fields = sample_keys - all_allowed_fields
        if unexpected_fields:
            # This is handled as a warning in _check_data_quality
            pass
        
        return errors
    
    def _analyze_schema(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze the schema of valid samples.
        
        Args:
            samples: List of valid sample dictionaries
            
        Returns:
            Dictionary containing schema information
        """
        if not samples:
            return {}
        
        schema_info = {
            'total_samples': len(samples),
            'fields': {},
            'prompt_stats': {},
            'response_stats': {}
        }
        
        # Analyze fields
        all_fields = set()
        field_counts = {}
        
        for sample in samples:
            normalized_sample = {k.lower(): v for k, v in sample.items()}
            sample_fields = set(normalized_sample.keys())
            all_fields.update(sample_fields)
            
            for field in sample_fields:
                field_counts[field] = field_counts.get(field, 0) + 1
        
        # Calculate field statistics
        for field in all_fields:
            count = field_counts.get(field, 0)
            schema_info['fields'][field] = {
                'present_in_samples': count,
                'coverage_percentage': (count / len(samples)) * 100,
                'is_required': field in self.required_fields
            }
        
        # Analyze prompt statistics
        prompts = [sample.get('prompt', sample.get('Prompt', '')) for sample in samples]
        prompts = [p for p in prompts if p]  # Filter out empty prompts
        
        if prompts:
            prompt_lengths = [len(p) for p in prompts]
            schema_info['prompt_stats'] = {
                'count': len(prompts),
                'avg_length': sum(prompt_lengths) / len(prompt_lengths),
                'min_length': min(prompt_lengths),
                'max_length': max(prompt_lengths),
                'empty_count': len(samples) - len(prompts)
            }
        
        # Analyze response statistics
        responses = [sample.get('response', sample.get('Response', '')) for sample in samples]
        responses = [r for r in responses if r]  # Filter out empty responses
        
        if responses:
            response_lengths = [len(r) for r in responses]
            schema_info['response_stats'] = {
                'count': len(responses),
                'avg_length': sum(response_lengths) / len(response_lengths),
                'min_length': min(response_lengths),
                'max_length': max(response_lengths),
                'empty_count': len(samples) - len(responses)
            }
        
        return schema_info
    
    def _check_data_quality(self, samples: List[Dict[str, Any]]) -> List[str]:
        """
        Check for data quality issues and return warnings.
        
        Args:
            samples: List of valid sample dictionaries
            
        Returns:
            List of warning messages
        """
        warnings = []
        
        if not samples:
            return warnings
        
        # Check for duplicate prompts
        prompts = []
        for sample in samples:
            prompt = sample.get('prompt', sample.get('Prompt', ''))
            if prompt:
                prompts.append(prompt.strip().lower())
        
        unique_prompts = set(prompts)
        if len(prompts) != len(unique_prompts):
            duplicate_count = len(prompts) - len(unique_prompts)
            warnings.append(f"Found {duplicate_count} duplicate prompts in dataset")
        
        # Check for very short prompts/responses
        short_prompts = 0
        short_responses = 0
        
        for sample in samples:
            normalized_sample = {k.lower(): v for k, v in sample.items()}
            
            prompt = normalized_sample.get('prompt', '')
            if prompt and len(prompt.strip()) < 10:
                short_prompts += 1
            
            response = normalized_sample.get('response', '')
            if response and len(response.strip()) < 10:
                short_responses += 1
        
        if short_prompts > 0:
            warnings.append(f"{short_prompts} prompts are very short (< 10 characters)")
        
        if short_responses > 0:
            warnings.append(f"{short_responses} responses are very short (< 10 characters)")
        
        # Check for unexpected fields
        all_fields = set()
        for sample in samples:
            normalized_sample = {k.lower(): v for k, v in sample.items()}
            all_fields.update(normalized_sample.keys())
        
        all_allowed_fields = self.required_fields | self.optional_fields
        unexpected_fields = all_fields - all_allowed_fields
        
        if unexpected_fields:
            warnings.append(f"Found unexpected fields: {unexpected_fields}")
        
        # Check for missing optional metadata
        samples_with_metadata = sum(1 for sample in samples 
                                  if sample.get('metadata') or sample.get('Metadata'))
        
        if samples_with_metadata == 0:
            warnings.append("No samples contain metadata - consider adding metadata for better tracking")
        
        return warnings
    
    def load_validated_dataset(self, dataset_path: Union[str, Path]) -> List[PromptResponsePair]:
        """
        Load and validate a dataset, returning a list of PromptResponsePair objects.
        
        Args:
            dataset_path: Path to the dataset file
            
        Returns:
            List of PromptResponsePair objects
            
        Raises:
            ValidationError: If validation fails
        """
        validation_result = self.validate_dataset(dataset_path)
        
        if not validation_result.is_valid:
            error_msg = "Dataset validation failed:\n" + "\n".join(validation_result.errors)
            raise ValidationError(error_msg)
        
        # Log warnings if any
        if validation_result.warnings:
            for warning in validation_result.warnings:
                self.logger.warning(warning)
        
        # Load the validated data
        dataset_path = Path(dataset_path)
        pairs = []
        
        if validation_result.format == DatasetFormat.JSONL:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        normalized_data = {k.lower(): v for k, v in data.items()}
                        
                        pair = PromptResponsePair(
                            prompt=normalized_data['prompt'],
                            response=normalized_data['response'],
                            metadata=normalized_data.get('metadata')
                        )
                        pairs.append(pair)
                    except (json.JSONDecodeError, KeyError):
                        # Skip invalid lines (already caught in validation)
                        continue
        
        elif validation_result.format == DatasetFormat.CSV:
            df = pd.read_csv(dataset_path)
            for _, row in df.iterrows():
                row_dict = {k.lower(): v for k, v in row.to_dict().items()}
                
                # Skip rows with missing required fields
                if pd.isna(row_dict.get('prompt')) or pd.isna(row_dict.get('response')):
                    continue
                
                metadata = {}
                for key, value in row_dict.items():
                    if key not in self.required_fields and not pd.isna(value):
                        metadata[key] = value
                
                pair = PromptResponsePair(
                    prompt=str(row_dict['prompt']),
                    response=str(row_dict['response']),
                    metadata=metadata if metadata else None
                )
                pairs.append(pair)
        
        self.logger.info(f"Successfully loaded {len(pairs)} prompt-response pairs")
        return pairs