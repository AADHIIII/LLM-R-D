"""
Unit tests for dataset validation module.

Tests various dataset formats, edge cases, and validation scenarios.
"""

import json
import pytest
import tempfile
import pandas as pd
from pathlib import Path
from unittest.mock import Mock

from fine_tuning.dataset_validator import (
    DatasetValidator,
    DatasetFormat,
    ValidationResult,
    PromptResponsePair
)
from utils.exceptions import ValidationError


class TestDatasetValidator:
    """Test cases for DatasetValidator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = DatasetValidator()
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_temp_file(self, filename: str, content: str) -> Path:
        """Create a temporary file with given content."""
        file_path = self.temp_path / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return file_path
    
    def test_detect_format_jsonl(self):
        """Test format detection for JSONL files."""
        file_path = self.create_temp_file("test.jsonl", "")
        format_type = self.validator._detect_format(file_path)
        assert format_type == DatasetFormat.JSONL
    
    def test_detect_format_csv(self):
        """Test format detection for CSV files."""
        file_path = self.create_temp_file("test.csv", "")
        format_type = self.validator._detect_format(file_path)
        assert format_type == DatasetFormat.CSV
    
    def test_detect_format_unsupported(self):
        """Test format detection for unsupported files."""
        file_path = self.create_temp_file("test.txt", "")
        with pytest.raises(ValidationError, match="Unsupported file extension"):
            self.validator._detect_format(file_path)
    
    def test_validate_nonexistent_file(self):
        """Test validation of non-existent file."""
        with pytest.raises(ValidationError, match="Dataset file not found"):
            self.validator.validate_dataset("nonexistent.jsonl")
    
    def test_validate_valid_jsonl(self):
        """Test validation of valid JSONL dataset."""
        content = '''{"prompt": "What is AI?", "response": "AI is artificial intelligence."}
{"prompt": "Define machine learning", "response": "ML is a subset of AI."}
{"prompt": "Explain deep learning", "response": "Deep learning uses neural networks."}'''
        
        file_path = self.create_temp_file("valid.jsonl", content)
        result = self.validator.validate_dataset(file_path)
        
        assert result.is_valid
        assert result.sample_count == 3
        assert result.format == DatasetFormat.JSONL
        assert len(result.errors) == 0
        assert 'prompt_stats' in result.schema_info
        assert 'response_stats' in result.schema_info
    
    def test_validate_valid_csv(self):
        """Test validation of valid CSV dataset."""
        df = pd.DataFrame({
            'prompt': ['What is AI?', 'Define ML', 'Explain DL'],
            'response': ['AI is artificial intelligence.', 'ML is a subset of AI.', 'DL uses neural networks.']
        })
        
        file_path = self.temp_path / "valid.csv"
        df.to_csv(file_path, index=False)
        
        result = self.validator.validate_dataset(file_path)
        
        assert result.is_valid
        assert result.sample_count == 3
        assert result.format == DatasetFormat.CSV
        assert len(result.errors) == 0
    
    def test_validate_jsonl_missing_required_fields(self):
        """Test validation of JSONL with missing required fields."""
        content = '''{"prompt": "What is AI?"}
{"response": "AI is artificial intelligence."}
{"prompt": "Define ML", "response": "ML is a subset of AI."}'''
        
        file_path = self.create_temp_file("invalid.jsonl", content)
        result = self.validator.validate_dataset(file_path)
        
        assert not result.is_valid
        assert len(result.errors) >= 2  # Two lines missing required fields
        assert any("Missing required fields" in error for error in result.errors)
    
    def test_validate_csv_missing_required_columns(self):
        """Test validation of CSV with missing required columns."""
        df = pd.DataFrame({
            'prompt': ['What is AI?', 'Define ML'],
            'answer': ['AI is artificial intelligence.', 'ML is a subset of AI.']  # Wrong column name
        })
        
        file_path = self.temp_path / "invalid.csv"
        df.to_csv(file_path, index=False)
        
        result = self.validator.validate_dataset(file_path)
        
        assert not result.is_valid
        assert any("Missing required columns" in error for error in result.errors)
    
    def test_validate_empty_jsonl(self):
        """Test validation of empty JSONL file."""
        file_path = self.create_temp_file("empty.jsonl", "")
        result = self.validator.validate_dataset(file_path)
        
        assert not result.is_valid
        assert result.sample_count == 0
        assert any("must contain at least" in error for error in result.errors)
    
    def test_validate_empty_csv(self):
        """Test validation of empty CSV file."""
        file_path = self.create_temp_file("empty.csv", "")
        result = self.validator.validate_dataset(file_path)
        
        assert not result.is_valid
        assert any("Error reading CSV file" in error for error in result.errors)
    
    def test_validate_invalid_json(self):
        """Test validation of JSONL with invalid JSON."""
        content = '''{"prompt": "What is AI?", "response": "AI is artificial intelligence."}
{invalid json}
{"prompt": "Define ML", "response": "ML is a subset of AI."}'''
        
        file_path = self.create_temp_file("invalid_json.jsonl", content)
        result = self.validator.validate_dataset(file_path)
        
        assert not result.is_valid
        assert any("Invalid JSON" in error for error in result.errors)
        assert result.sample_count == 2  # Only valid samples counted
    
    def test_validate_empty_strings(self):
        """Test validation with empty prompt/response strings."""
        content = '''{"prompt": "", "response": "Valid response"}
{"prompt": "Valid prompt", "response": ""}
{"prompt": "   ", "response": "   "}'''
        
        file_path = self.create_temp_file("empty_strings.jsonl", content)
        result = self.validator.validate_dataset(file_path)
        
        assert not result.is_valid
        assert len([e for e in result.errors if "cannot be empty" in e]) >= 3
    
    def test_validate_wrong_data_types(self):
        """Test validation with wrong data types."""
        content = '''{"prompt": 123, "response": "Valid response"}
{"prompt": "Valid prompt", "response": ["list", "not", "string"]}
{"prompt": null, "response": "Valid response"}'''
        
        file_path = self.create_temp_file("wrong_types.jsonl", content)
        result = self.validator.validate_dataset(file_path)
        
        assert not result.is_valid
        assert any("must be a string" in error for error in result.errors)
    
    def test_validate_too_long_content(self):
        """Test validation with content exceeding maximum length."""
        long_prompt = "A" * 5000  # Exceeds max_prompt_length
        long_response = "B" * 5000  # Exceeds max_response_length
        
        content = f'{{"prompt": "{long_prompt}", "response": "Valid response"}}' + '\n'
        content += f'{{"prompt": "Valid prompt", "response": "{long_response}"}}'
        
        file_path = self.create_temp_file("too_long.jsonl", content)
        result = self.validator.validate_dataset(file_path)
        
        assert not result.is_valid
        assert any("exceeds maximum length" in error for error in result.errors)
    
    def test_validate_with_metadata(self):
        """Test validation with optional metadata."""
        content = '''{"prompt": "What is AI?", "response": "AI is artificial intelligence.", "metadata": {"category": "AI", "difficulty": "easy"}}
{"prompt": "Define ML", "response": "ML is a subset of AI.", "metadata": null}
{"prompt": "Explain DL", "response": "DL uses neural networks."}'''
        
        file_path = self.create_temp_file("with_metadata.jsonl", content)
        result = self.validator.validate_dataset(file_path)
        
        assert result.is_valid
        assert result.sample_count == 3
        assert len(result.errors) == 0
    
    def test_validate_invalid_metadata(self):
        """Test validation with invalid metadata."""
        content = '''{"prompt": "What is AI?", "response": "AI is artificial intelligence.", "metadata": "should be dict"}'''
        
        file_path = self.create_temp_file("invalid_metadata.jsonl", content)
        result = self.validator.validate_dataset(file_path)
        
        assert not result.is_valid
        assert any("must be a dictionary" in error for error in result.errors)
    
    def test_data_quality_warnings(self):
        """Test data quality warnings generation."""
        # Create dataset with quality issues
        content = '''{"prompt": "What is AI?", "response": "AI is artificial intelligence."}
{"prompt": "What is AI?", "response": "Different response to same prompt"}
{"prompt": "Short", "response": "Short"}
{"prompt": "Valid prompt", "response": "Valid response", "unexpected_field": "value"}'''
        
        file_path = self.create_temp_file("quality_issues.jsonl", content)
        result = self.validator.validate_dataset(file_path)
        
        assert result.is_valid  # Valid but with warnings
        assert len(result.warnings) > 0
        
        warning_text = " ".join(result.warnings)
        assert "duplicate prompts" in warning_text
        assert "very short" in warning_text
        assert "unexpected fields" in warning_text
    
    def test_schema_analysis(self):
        """Test schema analysis functionality."""
        content = '''{"prompt": "Long prompt with more than ten characters", "response": "Long response with more than ten characters", "metadata": {"category": "test"}}
{"prompt": "Another long prompt", "response": "Another long response"}
{"prompt": "Third prompt", "response": "Third response", "id": "123"}'''
        
        file_path = self.create_temp_file("schema_test.jsonl", content)
        result = self.validator.validate_dataset(file_path)
        
        assert result.is_valid
        schema = result.schema_info
        
        assert schema['total_samples'] == 3
        assert 'prompt' in schema['fields']
        assert 'response' in schema['fields']
        assert schema['fields']['prompt']['coverage_percentage'] == 100.0
        assert schema['fields']['metadata']['coverage_percentage'] == 33.33333333333333
        
        assert 'prompt_stats' in schema
        assert schema['prompt_stats']['count'] == 3
        assert schema['prompt_stats']['avg_length'] > 10
        
        assert 'response_stats' in schema
        assert schema['response_stats']['count'] == 3
    
    def test_load_validated_dataset_success(self):
        """Test successful loading of validated dataset."""
        content = '''{"prompt": "What is AI?", "response": "AI is artificial intelligence.", "metadata": {"category": "AI"}}
{"prompt": "Define ML", "response": "ML is a subset of AI."}'''
        
        file_path = self.create_temp_file("load_test.jsonl", content)
        pairs = self.validator.load_validated_dataset(file_path)
        
        assert len(pairs) == 2
        assert all(isinstance(pair, PromptResponsePair) for pair in pairs)
        assert pairs[0].prompt == "What is AI?"
        assert pairs[0].response == "AI is artificial intelligence."
        assert pairs[0].metadata == {"category": "AI"}
        assert pairs[1].metadata is None
    
    def test_load_validated_dataset_failure(self):
        """Test loading of invalid dataset."""
        content = '''{"prompt": "What is AI?"}'''  # Missing response
        
        file_path = self.create_temp_file("load_fail.jsonl", content)
        
        with pytest.raises(ValidationError, match="Dataset validation failed"):
            self.validator.load_validated_dataset(file_path)
    
    def test_csv_encoding_handling(self):
        """Test CSV files with different encodings."""
        # Create CSV with special characters
        df = pd.DataFrame({
            'prompt': ['What is café?', 'Explain résumé'],
            'response': ['Café is coffee.', 'Résumé is a summary.']
        })
        
        file_path = self.temp_path / "encoding_test.csv"
        df.to_csv(file_path, index=False, encoding='latin-1')
        
        result = self.validator.validate_dataset(file_path)
        assert result.is_valid
        assert result.sample_count == 2
    
    def test_case_insensitive_column_matching(self):
        """Test case-insensitive column matching for CSV."""
        df = pd.DataFrame({
            'Prompt': ['What is AI?', 'Define ML'],
            'Response': ['AI is artificial intelligence.', 'ML is a subset of AI.']
        })
        
        file_path = self.temp_path / "case_test.csv"
        df.to_csv(file_path, index=False)
        
        result = self.validator.validate_dataset(file_path)
        assert result.is_valid
        assert result.sample_count == 2
    
    def test_csv_with_nan_values(self):
        """Test CSV handling with NaN values."""
        df = pd.DataFrame({
            'prompt': ['What is AI?', None, 'Define ML'],
            'response': ['AI is artificial intelligence.', 'Valid response', None]
        })
        
        file_path = self.temp_path / "nan_test.csv"
        df.to_csv(file_path, index=False)
        
        result = self.validator.validate_dataset(file_path)
        # Should have errors for missing values
        assert not result.is_valid or result.sample_count < 3
    
    def test_custom_validation_parameters(self):
        """Test validator with custom parameters."""
        validator = DatasetValidator()
        validator.max_prompt_length = 50
        validator.max_response_length = 50
        validator.min_samples = 5
        
        content = '''{"prompt": "Short prompt", "response": "Short response"}
{"prompt": "Another short prompt", "response": "Another short response"}'''
        
        file_path = self.create_temp_file("custom_params.jsonl", content)
        result = validator.validate_dataset(file_path)
        
        assert not result.is_valid
        assert any("must contain at least 5" in error for error in result.errors)
    
    def test_logger_integration(self):
        """Test integration with logger."""
        mock_logger = Mock()
        validator = DatasetValidator(logger=mock_logger)
        
        content = '''{"prompt": "What is AI?", "response": "AI is artificial intelligence."}'''
        file_path = self.create_temp_file("logger_test.jsonl", content)
        
        result = validator.validate_dataset(file_path)
        
        assert result.is_valid
        mock_logger.info.assert_called()
    
    def test_validation_result_dataclass(self):
        """Test ValidationResult dataclass properties."""
        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=["Test warning"],
            sample_count=5,
            format=DatasetFormat.JSONL,
            schema_info={"test": "info"}
        )
        
        assert result.is_valid
        assert result.warnings == ["Test warning"]
        assert result.sample_count == 5
        assert result.format == DatasetFormat.JSONL
        assert result.schema_info == {"test": "info"}
    
    def test_prompt_response_pair_dataclass(self):
        """Test PromptResponsePair dataclass properties."""
        pair = PromptResponsePair(
            prompt="Test prompt",
            response="Test response",
            metadata={"key": "value"}
        )
        
        assert pair.prompt == "Test prompt"
        assert pair.response == "Test response"
        assert pair.metadata == {"key": "value"}
        
        # Test without metadata
        pair_no_meta = PromptResponsePair(
            prompt="Test prompt",
            response="Test response"
        )
        assert pair_no_meta.metadata is None