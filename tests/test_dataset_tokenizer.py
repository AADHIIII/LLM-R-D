"""
Unit tests for dataset tokenization module.

Tests tokenization functionality, memory efficiency, and accuracy.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from fine_tuning.dataset_tokenizer import (
    DatasetTokenizer,
    TokenizationConfig,
    TokenizedSample
)
from fine_tuning.dataset_validator import PromptResponsePair
from utils.exceptions import ConfigurationError


class TestTokenizationConfig:
    """Test cases for TokenizationConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = TokenizationConfig(model_name="gpt2")
        
        assert config.model_name == "gpt2"
        assert config.max_length == 512
        assert config.padding == "max_length"
        assert config.truncation is True
        assert config.add_special_tokens is True
        assert config.return_attention_mask is True
        assert config.return_token_type_ids is False
        assert config.prompt_template is None
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = TokenizationConfig(
            model_name="distilgpt2",
            max_length=256,
            padding="longest",
            truncation=False,
            prompt_template="Prompt: {prompt}\nResponse: {response}"
        )
        
        assert config.model_name == "distilgpt2"
        assert config.max_length == 256
        assert config.padding == "longest"
        assert config.truncation is False
        assert config.prompt_template == "Prompt: {prompt}\nResponse: {response}"


class TestTokenizedSample:
    """Test cases for TokenizedSample dataclass."""
    
    def test_tokenized_sample_creation(self):
        """Test TokenizedSample creation."""
        sample = TokenizedSample(
            input_ids=[1, 2, 3, 4],
            attention_mask=[1, 1, 1, 1],
            labels=[1, 2, 3, 4],
            original_prompt="Test prompt",
            original_response="Test response",
            metadata={"category": "test"}
        )
        
        assert sample.input_ids == [1, 2, 3, 4]
        assert sample.attention_mask == [1, 1, 1, 1]
        assert sample.labels == [1, 2, 3, 4]
        assert sample.original_prompt == "Test prompt"
        assert sample.original_response == "Test response"
        assert sample.metadata == {"category": "test"}
    
    def test_tokenized_sample_without_metadata(self):
        """Test TokenizedSample without metadata."""
        sample = TokenizedSample(
            input_ids=[1, 2, 3],
            attention_mask=[1, 1, 1],
            labels=[1, 2, 3],
            original_prompt="Test",
            original_response="Response"
        )
        
        assert sample.metadata is None


class TestDatasetTokenizer:
    """Test cases for DatasetTokenizer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = TokenizationConfig(model_name="gpt2", max_length=128)
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def create_mock_tokenizer(self, **kwargs):
        """Create a properly mocked tokenizer."""
        mock_tokenizer = Mock()
        mock_tokenizer.eos_token = kwargs.get('eos_token', "<|endoftext|>")
        mock_tokenizer.eos_token_id = kwargs.get('eos_token_id', 50256)
        mock_tokenizer.pad_token = kwargs.get('pad_token', None)
        mock_tokenizer.pad_token_id = kwargs.get('pad_token_id', 50256)
        mock_tokenizer.bos_token = kwargs.get('bos_token', None)
        mock_tokenizer.unk_token = kwargs.get('unk_token', "<|unk|>")
        mock_tokenizer.__len__ = Mock(return_value=kwargs.get('vocab_size', 50257))
        mock_tokenizer.decode = Mock(return_value=kwargs.get('decode_return', "Decoded text"))
        
        # Default tokenizer call behavior - return the actual dictionary, not a Mock
        default_return = {
            "input_ids": kwargs.get('input_ids', [1, 2, 3, 4]),
            "attention_mask": kwargs.get('attention_mask', [1, 1, 1, 1])
        }
        
        # Configure the mock to return the dictionary when called
        mock_tokenizer.return_value = default_return
        mock_tokenizer.side_effect = None
        
        return mock_tokenizer
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_supported_models(self):
        """Test that supported models are correctly defined."""
        expected_models = {"gpt2", "gpt2-medium", "gpt2-large", "distilgpt2"}
        actual_models = set(DatasetTokenizer.SUPPORTED_MODELS.keys())
        
        assert actual_models == expected_models
        
        # Check that each model has required configuration
        for model_name, model_config in DatasetTokenizer.SUPPORTED_MODELS.items():
            assert "model_name" in model_config
            assert "pad_token" in model_config
            assert "eos_token" in model_config
            assert "default_max_length" in model_config
    
    def test_unsupported_model_error(self):
        """Test error handling for unsupported models."""
        config = TokenizationConfig(model_name="unsupported-model")
        
        with pytest.raises(ConfigurationError, match="Unsupported model"):
            DatasetTokenizer(config)
    
    @patch('fine_tuning.dataset_tokenizer.AutoTokenizer')
    def test_tokenizer_initialization(self, mock_auto_tokenizer):
        """Test tokenizer initialization."""
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<|endoftext|>"
        mock_tokenizer.__len__ = Mock(return_value=50257)
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        
        tokenizer = DatasetTokenizer(self.config)
        
        assert tokenizer.config == self.config
        assert tokenizer.tokenizer == mock_tokenizer
        mock_auto_tokenizer.from_pretrained.assert_called_once_with("gpt2")
        
        # Check that pad token was set
        assert mock_tokenizer.pad_token == "<|endoftext|>"
    
    @patch('fine_tuning.dataset_tokenizer.AutoTokenizer')
    def test_tokenizer_loading_error(self, mock_auto_tokenizer):
        """Test error handling during tokenizer loading."""
        mock_auto_tokenizer.from_pretrained.side_effect = Exception("Loading failed")
        
        with pytest.raises(ConfigurationError, match="Failed to load tokenizer"):
            DatasetTokenizer(self.config)
    
    @patch('fine_tuning.dataset_tokenizer.AutoTokenizer')
    def test_tokenize_single_pair_default_format(self, mock_auto_tokenizer):
        """Test tokenizing a single prompt-response pair with default format."""
        mock_tokenizer = self.create_mock_tokenizer(
            input_ids=[1, 2, 3, 4, 5],
            attention_mask=[1, 1, 1, 1, 1]
        )
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        
        tokenizer = DatasetTokenizer(self.config)
        
        pair = PromptResponsePair(
            prompt="What is AI?",
            response="AI is artificial intelligence."
        )
        
        result = tokenizer.tokenize_single_pair(pair)
        
        assert isinstance(result, TokenizedSample)
        assert result.input_ids == [1, 2, 3, 4, 5]
        assert result.attention_mask == [1, 1, 1, 1, 1]
        # Labels should be masked for prompt tokens (set to -100)
        assert all(label == -100 for label in result.labels)
        assert result.original_prompt == "What is AI?"
        assert result.original_response == "AI is artificial intelligence."
    
    @patch('fine_tuning.dataset_tokenizer.AutoTokenizer')
    def test_tokenize_single_pair_custom_template(self, mock_auto_tokenizer):
        """Test tokenizing with custom prompt template."""
        mock_tokenizer = self.create_mock_tokenizer(
            input_ids=[1, 2, 3, 4],
            attention_mask=[1, 1, 1, 1]
        )
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        
        config = TokenizationConfig(
            model_name="gpt2",
            prompt_template="Q: {prompt}\nA: {response}"
        )
        tokenizer = DatasetTokenizer(config)
        
        pair = PromptResponsePair(
            prompt="What is AI?",
            response="AI is artificial intelligence."
        )
        
        result = tokenizer.tokenize_single_pair(pair)
        
        # Check that the template was used
        expected_text = "Q: What is AI?\nA: AI is artificial intelligence."
        mock_tokenizer.assert_called_with(
            expected_text,
            max_length=512,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors=None
        )
    
    @patch('fine_tuning.dataset_tokenizer.AutoTokenizer')
    @patch('fine_tuning.dataset_tokenizer.DatasetValidator')
    def test_tokenize_prompt_response_pairs(self, mock_validator_class, mock_auto_tokenizer):
        """Test tokenizing multiple prompt-response pairs."""
        mock_tokenizer = self.create_mock_tokenizer(
            input_ids=[1, 2, 3],
            attention_mask=[1, 1, 1]
        )
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        
        tokenizer = DatasetTokenizer(self.config)
        
        pairs = [
            PromptResponsePair("Prompt 1", "Response 1"),
            PromptResponsePair("Prompt 2", "Response 2")
        ]
        
        dataset = tokenizer.tokenize_prompt_response_pairs(pairs)
        
        assert len(dataset) == 2
        assert "input_ids" in dataset.column_names
        assert "attention_mask" in dataset.column_names
        assert "labels" in dataset.column_names
        assert "original_prompt" in dataset.column_names
        assert "original_response" in dataset.column_names
    
    @patch('fine_tuning.dataset_tokenizer.AutoTokenizer')
    def test_get_tokenizer_info(self, mock_auto_tokenizer):
        """Test getting tokenizer information."""
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = "<|endoftext|>"
        mock_tokenizer.eos_token = "<|endoftext|>"
        mock_tokenizer.bos_token = None
        mock_tokenizer.unk_token = "<|unk|>"
        mock_tokenizer.__len__ = Mock(return_value=50257)
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        
        tokenizer = DatasetTokenizer(self.config)
        info = tokenizer.get_tokenizer_info()
        
        assert info["model_name"] == "gpt2"
        assert info["vocab_size"] == 50257
        assert info["pad_token"] == "<|endoftext|>"
        assert info["eos_token"] == "<|endoftext|>"
        assert info["bos_token"] is None
        assert info["unk_token"] == "<|unk|>"
        assert info["max_length"] == 128
        assert info["padding"] == "max_length"
        assert info["truncation"] is True
    
    @patch('fine_tuning.dataset_tokenizer.AutoTokenizer')
    def test_estimate_memory_usage(self, mock_auto_tokenizer):
        """Test memory usage estimation."""
        mock_tokenizer = self.create_mock_tokenizer()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        
        tokenizer = DatasetTokenizer(self.config)
        estimates = tokenizer.estimate_memory_usage(1000)
        
        assert "input_ids_mb" in estimates
        assert "attention_mask_mb" in estimates
        assert "labels_mb" in estimates
        assert "total_mb" in estimates
        assert "total_gb" in estimates
        
        # Check that estimates are reasonable
        assert estimates["total_mb"] > 0
        assert estimates["total_gb"] == estimates["total_mb"] / 1024
    
    @patch('fine_tuning.dataset_tokenizer.AutoTokenizer')
    def test_validate_tokenization(self, mock_auto_tokenizer):
        """Test tokenization validation."""
        mock_tokenizer = self.create_mock_tokenizer()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        
        tokenizer = DatasetTokenizer(self.config)
        
        # Test valid sample
        valid_sample = TokenizedSample(
            input_ids=[1, 2, 3, 4],
            attention_mask=[1, 1, 1, 1],
            labels=[1, 2, 3, 4],
            original_prompt="Test",
            original_response="Response"
        )
        
        issues = tokenizer.validate_tokenization(valid_sample)
        assert len(issues) == 0
        
        # Test sample with mismatched lengths
        invalid_sample = TokenizedSample(
            input_ids=[1, 2, 3],
            attention_mask=[1, 1],  # Wrong length
            labels=[1, 2, 3],
            original_prompt="Test",
            original_response="Response"
        )
        
        issues = tokenizer.validate_tokenization(invalid_sample)
        assert len(issues) > 0
        assert any("Mismatch" in issue for issue in issues)
    
    @patch('fine_tuning.dataset_tokenizer.AutoTokenizer')
    def test_decode_sample(self, mock_auto_tokenizer):
        """Test decoding tokenized samples."""
        mock_tokenizer = self.create_mock_tokenizer(decode_return="Decoded text")
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        
        tokenizer = DatasetTokenizer(self.config)
        
        sample = TokenizedSample(
            input_ids=[1, 2, 3, 4],
            attention_mask=[1, 1, 1, 1],
            labels=[1, 2, 3, 4],
            original_prompt="Test",
            original_response="Response"
        )
        
        decoded = tokenizer.decode_sample(sample)
        assert decoded == "Decoded text"
        mock_tokenizer.decode.assert_called_once_with([1, 2, 3, 4], skip_special_tokens=False)
    
    @patch('fine_tuning.dataset_tokenizer.AutoTokenizer')
    @patch('fine_tuning.dataset_tokenizer.Dataset')
    def test_save_tokenized_dataset(self, mock_dataset_class, mock_auto_tokenizer):
        """Test saving tokenized dataset."""
        mock_tokenizer = self.create_mock_tokenizer(
            pad_token="<|endoftext|>",
            eos_token="<|endoftext|>",
            bos_token=None,  # Ensure this is JSON serializable
            unk_token="<|unk|>"
        )
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        
        # Mock dataset
        mock_dataset = Mock()
        mock_dataset.save_to_disk = Mock()
        
        tokenizer = DatasetTokenizer(self.config)
        output_path = self.temp_path / "test_dataset"
        
        tokenizer.save_tokenized_dataset(mock_dataset, output_path)
        
        mock_dataset.save_to_disk.assert_called_once_with(str(output_path))
        
        # Check that config file was created
        config_path = output_path.parent / f"{output_path.name}_tokenizer_config.json"
        assert config_path.exists()
        
        with open(config_path) as f:
            saved_config = json.load(f)
        
        assert saved_config["model_name"] == "gpt2"
        assert saved_config["vocab_size"] == 50257
    
    @patch('fine_tuning.dataset_tokenizer.Dataset')
    def test_load_tokenized_dataset(self, mock_dataset_class):
        """Test loading tokenized dataset."""
        mock_dataset = Mock()
        mock_dataset_class.load_from_disk.return_value = mock_dataset
        
        dataset_path = self.temp_path / "test_dataset"
        loaded_dataset = DatasetTokenizer.load_tokenized_dataset(dataset_path)
        
        assert loaded_dataset == mock_dataset
        mock_dataset_class.load_from_disk.assert_called_once_with(str(dataset_path))
    
    @patch('fine_tuning.dataset_tokenizer.AutoTokenizer')
    def test_tokenization_with_different_models(self, mock_auto_tokenizer):
        """Test tokenization with different supported models."""
        mock_tokenizer = self.create_mock_tokenizer()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        
        for model_name in DatasetTokenizer.SUPPORTED_MODELS.keys():
            config = TokenizationConfig(model_name=model_name)
            tokenizer = DatasetTokenizer(config)
            
            assert tokenizer.config.model_name == model_name
            assert tokenizer.model_config == DatasetTokenizer.SUPPORTED_MODELS[model_name]
    
    @patch('fine_tuning.dataset_tokenizer.AutoTokenizer')
    def test_tokenization_error_handling(self, mock_auto_tokenizer):
        """Test error handling during tokenization."""
        mock_tokenizer = self.create_mock_tokenizer()
        # Override the return_value to raise an exception
        mock_tokenizer.side_effect = Exception("Tokenization failed")
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        
        tokenizer = DatasetTokenizer(self.config)
        
        pairs = [
            PromptResponsePair("Valid prompt", "Valid response"),
            PromptResponsePair("Invalid prompt", "Invalid response")  # This will fail
        ]
        
        # Should handle errors gracefully and continue with valid samples
        dataset = tokenizer.tokenize_prompt_response_pairs(pairs)
        
        # Should have 0 samples since all failed
        assert len(dataset) == 0
    
    @patch('fine_tuning.dataset_tokenizer.AutoTokenizer')
    def test_prompt_masking_in_labels(self, mock_auto_tokenizer):
        """Test that prompt tokens are masked in labels for loss calculation."""
        mock_tokenizer = self.create_mock_tokenizer()
        
        # Mock tokenizer calls
        def mock_tokenizer_call(text, **kwargs):
            if text == "What is AI?<|endoftext|>":
                # This is the prompt + EOS token call (for masking)
                return {
                    "input_ids": [1, 2, 3, 4],
                    "attention_mask": [1, 1, 1, 1]
                }
            else:
                # This is the full text call (prompt + EOS + response + EOS)
                return {
                    "input_ids": [1, 2, 3, 4, 5, 6, 7],
                    "attention_mask": [1, 1, 1, 1, 1, 1, 1]
                }
        
        # Override the side_effect to use our custom function
        mock_tokenizer.side_effect = mock_tokenizer_call
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        
        tokenizer = DatasetTokenizer(self.config)
        
        pair = PromptResponsePair(
            prompt="What is AI?",
            response="AI is artificial intelligence."
        )
        
        result = tokenizer.tokenize_single_pair(pair)
        
        # Check that first 4 tokens (prompt + EOS) are masked in labels
        # The actual result should have 7 tokens total with first 4 masked
        assert len(result.input_ids) == 7
        assert len(result.labels) == 7
        expected_labels = [-100, -100, -100, -100, 5, 6, 7]
        assert result.labels == expected_labels
    
    @patch('fine_tuning.dataset_tokenizer.AutoTokenizer')
    def test_excessive_padding_warning(self, mock_auto_tokenizer):
        """Test warning for excessive padding."""
        mock_tokenizer = self.create_mock_tokenizer()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        
        tokenizer = DatasetTokenizer(self.config)
        
        # Create sample with excessive padding (more than 50% padding tokens)
        sample = TokenizedSample(
            input_ids=[1, 2, 50256, 50256, 50256, 50256],  # 4 out of 6 are padding
            attention_mask=[1, 1, 0, 0, 0, 0],
            labels=[1, 2, 50256, 50256, 50256, 50256],
            original_prompt="Test",
            original_response="Response"
        )
        
        issues = tokenizer.validate_tokenization(sample)
        assert any("Excessive padding" in issue for issue in issues)