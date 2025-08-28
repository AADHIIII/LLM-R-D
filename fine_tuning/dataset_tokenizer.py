"""
Dataset tokenization module for LLM fine-tuning platform.

This module provides tokenization functionality for datasets using Hugging Face tokenizers,
supporting different base models with proper padding and truncation.
"""

import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import json

try:
    from transformers import AutoTokenizer, PreTrainedTokenizer
    from datasets import Dataset
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    # For testing purposes, create mock classes
    TRANSFORMERS_AVAILABLE = False
    
    class MockAutoTokenizer:
        @classmethod
        def from_pretrained(cls, model_name):
            return MockTokenizer()
    
    class MockTokenizer:
        def __init__(self):
            self.pad_token = None
            self.eos_token = "<|endoftext|>"
            self.eos_token_id = 50256
            self.pad_token_id = 50256
            self.bos_token = None
            self.unk_token = "<|unk|>"
        
        def __len__(self):
            return 50257
        
        def __call__(self, text, **kwargs):
            # Simple mock tokenization
            tokens = text.split()[:kwargs.get('max_length', 512)]
            input_ids = list(range(1, len(tokens) + 1))
            attention_mask = [1] * len(input_ids)
            
            if kwargs.get('padding') == 'max_length':
                max_len = kwargs.get('max_length', 512)
                while len(input_ids) < max_len:
                    input_ids.append(self.pad_token_id)
                    attention_mask.append(0)
            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }
        
        def decode(self, token_ids, skip_special_tokens=False):
            return f"Decoded: {' '.join(map(str, token_ids))}"
    
    class MockDataset:
        def __init__(self, data_dict):
            self.data = data_dict
            self.column_names = list(data_dict.keys())
        
        def __len__(self):
            return len(next(iter(self.data.values())))
        
        def save_to_disk(self, path):
            pass
        
        @classmethod
        def from_dict(cls, data_dict):
            return cls(data_dict)
        
        @classmethod
        def load_from_disk(cls, path):
            return cls({"input_ids": [], "attention_mask": [], "labels": []})
    
    AutoTokenizer = MockAutoTokenizer
    Dataset = MockDataset
    
    if not TRANSFORMERS_AVAILABLE:
        import warnings
        warnings.warn("Transformers library not available. Using mock implementations for testing.")

from .dataset_validator import PromptResponsePair, DatasetValidator
from utils.exceptions import ValidationError, ConfigurationError


@dataclass
class TokenizationConfig:
    """Configuration for dataset tokenization."""
    model_name: str
    max_length: int = 512
    padding: str = "max_length"  # "max_length", "longest", or False
    truncation: bool = True
    add_special_tokens: bool = True
    return_attention_mask: bool = True
    return_token_type_ids: bool = False
    prompt_template: Optional[str] = None  # Template for formatting prompt-response pairs


@dataclass
class TokenizedSample:
    """Represents a tokenized sample with input and target tokens."""
    input_ids: List[int]
    attention_mask: List[int]
    labels: List[int]
    original_prompt: str
    original_response: str
    metadata: Optional[Dict[str, Any]] = None


class DatasetTokenizer:
    """
    Tokenizes datasets for fine-tuning LLM models.
    
    Supports various base models from Hugging Face with configurable tokenization
    parameters including padding, truncation, and special token handling.
    """
    
    # Supported base models with their configurations
    SUPPORTED_MODELS = {
        "gpt2": {
            "model_name": "gpt2",
            "pad_token": "<|endoftext|>",
            "eos_token": "<|endoftext|>",
            "default_max_length": 512
        },
        "gpt2-medium": {
            "model_name": "gpt2-medium",
            "pad_token": "<|endoftext|>",
            "eos_token": "<|endoftext|>",
            "default_max_length": 512
        },
        "gpt2-large": {
            "model_name": "gpt2-large",
            "pad_token": "<|endoftext|>",
            "eos_token": "<|endoftext|>",
            "default_max_length": 512
        },
        "distilgpt2": {
            "model_name": "distilgpt2",
            "pad_token": "<|endoftext|>",
            "eos_token": "<|endoftext|>",
            "default_max_length": 512
        }
    }
    
    def __init__(self, config: TokenizationConfig, logger: Optional[logging.Logger] = None):
        """
        Initialize the dataset tokenizer.
        
        Args:
            config: Tokenization configuration
            logger: Optional logger instance
            
        Raises:
            ConfigurationError: If the model is not supported or configuration is invalid
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Validate model support
        if config.model_name not in self.SUPPORTED_MODELS:
            supported_list = ", ".join(self.SUPPORTED_MODELS.keys())
            raise ConfigurationError(
                f"Unsupported model: {config.model_name}. Supported models: {supported_list}"
            )
        
        self.model_config = self.SUPPORTED_MODELS[config.model_name]
        self.tokenizer = None
        self._load_tokenizer()
    
    def _load_tokenizer(self):
        """Load the tokenizer for the specified model."""
        try:
            self.logger.info(f"Loading tokenizer for model: {self.config.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_config["model_name"])
            
            # Set pad token if not already set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.model_config["pad_token"]
                self.logger.info(f"Set pad token to: {self.tokenizer.pad_token}")
            
            # Ensure EOS token is set
            if self.tokenizer.eos_token is None:
                self.tokenizer.eos_token = self.model_config["eos_token"]
                self.logger.info(f"Set EOS token to: {self.tokenizer.eos_token}")
            
            self.logger.info(f"Successfully loaded tokenizer. Vocab size: {len(self.tokenizer)}")
            
        except Exception as e:
            raise ConfigurationError(f"Failed to load tokenizer for {self.config.model_name}: {str(e)}")
    
    def tokenize_dataset(self, dataset_path: Union[str, Path]) -> Dataset:
        """
        Tokenize a complete dataset from file.
        
        Args:
            dataset_path: Path to the dataset file
            
        Returns:
            Hugging Face Dataset with tokenized samples
            
        Raises:
            ValidationError: If dataset validation fails
        """
        # First validate the dataset
        validator = DatasetValidator(logger=self.logger)
        prompt_response_pairs = validator.load_validated_dataset(dataset_path)
        
        return self.tokenize_prompt_response_pairs(prompt_response_pairs)
    
    def tokenize_prompt_response_pairs(self, pairs: List[PromptResponsePair]) -> Dataset:
        """
        Tokenize a list of prompt-response pairs.
        
        Args:
            pairs: List of PromptResponsePair objects
            
        Returns:
            Hugging Face Dataset with tokenized samples
        """
        self.logger.info(f"Tokenizing {len(pairs)} prompt-response pairs")
        
        tokenized_samples = []
        for i, pair in enumerate(pairs):
            try:
                tokenized_sample = self.tokenize_single_pair(pair)
                tokenized_samples.append(tokenized_sample)
            except Exception as e:
                self.logger.warning(f"Failed to tokenize sample {i}: {str(e)}")
                continue
        
        self.logger.info(f"Successfully tokenized {len(tokenized_samples)} samples")
        
        # Convert to Hugging Face Dataset format
        dataset_dict = {
            "input_ids": [sample.input_ids for sample in tokenized_samples],
            "attention_mask": [sample.attention_mask for sample in tokenized_samples],
            "labels": [sample.labels for sample in tokenized_samples],
            "original_prompt": [sample.original_prompt for sample in tokenized_samples],
            "original_response": [sample.original_response for sample in tokenized_samples]
        }
        
        return Dataset.from_dict(dataset_dict)
    
    def tokenize_single_pair(self, pair: PromptResponsePair) -> TokenizedSample:
        """
        Tokenize a single prompt-response pair.
        
        Args:
            pair: PromptResponsePair to tokenize
            
        Returns:
            TokenizedSample with tokenized data
        """
        # Format the input text
        if self.config.prompt_template:
            formatted_text = self.config.prompt_template.format(
                prompt=pair.prompt,
                response=pair.response
            )
        else:
            # Default format: prompt + response with EOS token
            formatted_text = f"{pair.prompt}{self.tokenizer.eos_token}{pair.response}{self.tokenizer.eos_token}"
        
        # Tokenize the formatted text
        encoding = self.tokenizer(
            formatted_text,
            max_length=self.config.max_length,
            padding=self.config.padding,
            truncation=self.config.truncation,
            add_special_tokens=self.config.add_special_tokens,
            return_attention_mask=self.config.return_attention_mask,
            return_token_type_ids=self.config.return_token_type_ids,
            return_tensors=None  # Return lists, not tensors
        )
        
        # For causal language modeling, labels are the same as input_ids
        # but we typically mask the prompt tokens during loss calculation
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        
        # Create labels (same as input_ids for causal LM)
        labels = input_ids.copy()
        
        # Optionally mask prompt tokens in labels (set to -100 to ignore in loss)
        if self.config.prompt_template is None:
            # Find where the response starts (after first EOS token)
            prompt_tokens = self.tokenizer(
                pair.prompt + self.tokenizer.eos_token,
                add_special_tokens=self.config.add_special_tokens,
                return_tensors=None
            )["input_ids"]
            
            # Mask prompt tokens in labels
            for i in range(min(len(prompt_tokens), len(labels))):
                labels[i] = -100
        
        return TokenizedSample(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            original_prompt=pair.prompt,
            original_response=pair.response,
            metadata=pair.metadata
        )
    
    def get_tokenizer_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded tokenizer.
        
        Returns:
            Dictionary with tokenizer information
        """
        if not self.tokenizer:
            return {}
        
        return {
            "model_name": self.config.model_name,
            "vocab_size": len(self.tokenizer),
            "pad_token": self.tokenizer.pad_token,
            "eos_token": self.tokenizer.eos_token,
            "bos_token": getattr(self.tokenizer, 'bos_token', None),
            "unk_token": getattr(self.tokenizer, 'unk_token', None),
            "max_length": self.config.max_length,
            "padding": self.config.padding,
            "truncation": self.config.truncation
        }
    
    def estimate_memory_usage(self, num_samples: int) -> Dict[str, float]:
        """
        Estimate memory usage for tokenizing a given number of samples.
        
        Args:
            num_samples: Number of samples to estimate for
            
        Returns:
            Dictionary with memory estimates in MB
        """
        # Rough estimates based on typical token counts and data types
        avg_tokens_per_sample = self.config.max_length
        bytes_per_token = 4  # int32
        
        input_ids_mb = (num_samples * avg_tokens_per_sample * bytes_per_token) / (1024 * 1024)
        attention_mask_mb = input_ids_mb  # Same size as input_ids
        labels_mb = input_ids_mb  # Same size as input_ids
        
        total_mb = input_ids_mb + attention_mask_mb + labels_mb
        
        return {
            "input_ids_mb": input_ids_mb,
            "attention_mask_mb": attention_mask_mb,
            "labels_mb": labels_mb,
            "total_mb": total_mb,
            "total_gb": total_mb / 1024
        }
    
    def validate_tokenization(self, tokenized_sample: TokenizedSample) -> List[str]:
        """
        Validate a tokenized sample for common issues.
        
        Args:
            tokenized_sample: TokenizedSample to validate
            
        Returns:
            List of validation warnings/errors
        """
        issues = []
        
        # Check for empty sequences
        if not tokenized_sample.input_ids:
            issues.append("Empty input_ids sequence")
        
        if not tokenized_sample.attention_mask:
            issues.append("Empty attention_mask sequence")
        
        # Check sequence lengths match
        if len(tokenized_sample.input_ids) != len(tokenized_sample.attention_mask):
            issues.append("Mismatch between input_ids and attention_mask lengths")
        
        if len(tokenized_sample.input_ids) != len(tokenized_sample.labels):
            issues.append("Mismatch between input_ids and labels lengths")
        
        # Check for excessive padding
        if self.config.padding == "max_length":
            pad_token_id = self.tokenizer.pad_token_id
            pad_count = tokenized_sample.input_ids.count(pad_token_id)
            pad_ratio = pad_count / len(tokenized_sample.input_ids)
            
            if pad_ratio > 0.5:
                issues.append(f"Excessive padding: {pad_ratio:.1%} of tokens are padding")
        
        # Check for truncation
        if len(tokenized_sample.input_ids) == self.config.max_length:
            if tokenized_sample.input_ids[-1] != self.tokenizer.eos_token_id:
                issues.append("Sequence may be truncated (no EOS token at end)")
        
        return issues
    
    def decode_sample(self, tokenized_sample: TokenizedSample) -> str:
        """
        Decode a tokenized sample back to text for inspection.
        
        Args:
            tokenized_sample: TokenizedSample to decode
            
        Returns:
            Decoded text string
        """
        return self.tokenizer.decode(tokenized_sample.input_ids, skip_special_tokens=False)
    
    def save_tokenized_dataset(self, dataset: Dataset, output_path: Union[str, Path]):
        """
        Save a tokenized dataset to disk.
        
        Args:
            dataset: Tokenized dataset to save
            output_path: Path to save the dataset
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Saving tokenized dataset to: {output_path}")
        dataset.save_to_disk(str(output_path))
        
        # Save tokenizer configuration alongside
        config_path = output_path.parent / f"{output_path.name}_tokenizer_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.get_tokenizer_info(), f, indent=2)
        
        self.logger.info(f"Saved tokenized dataset and configuration")
    
    @classmethod
    def load_tokenized_dataset(cls, dataset_path: Union[str, Path]) -> Dataset:
        """
        Load a previously saved tokenized dataset.
        
        Args:
            dataset_path: Path to the saved dataset
            
        Returns:
            Loaded Dataset
        """
        return Dataset.load_from_disk(str(dataset_path))