"""
Unit tests for training configuration management.

Tests the TrainingConfig dataclass, validation logic, and configuration loading utilities.
"""

import pytest
import tempfile
import json
import yaml
from pathlib import Path
from unittest.mock import patch

from fine_tuning.training_config import (
    TrainingConfig, 
    ConfigurationLoader, 
    validate_config_compatibility
)
from utils.exceptions import ValidationError


class TestTrainingConfig:
    """Test cases for TrainingConfig dataclass."""
    
    def test_default_config_creation(self):
        """Test creating config with default values."""
        config = TrainingConfig()
        
        assert config.base_model == "gpt2"
        assert config.epochs == 3
        assert config.batch_size == 4
        assert config.learning_rate == 5e-5
        assert config.use_lora is True
        assert config.lora_rank == 16
    
    def test_custom_config_creation(self):
        """Test creating config with custom values."""
        config = TrainingConfig(
            base_model="gpt2-medium",
            epochs=5,
            batch_size=8,
            learning_rate=3e-5,
            use_lora=False
        )
        
        assert config.base_model == "gpt2-medium"
        assert config.epochs == 5
        assert config.batch_size == 8
        assert config.learning_rate == 3e-5
        assert config.use_lora is False
    
    def test_validation_success(self):
        """Test successful validation of valid config."""
        config = TrainingConfig(
            base_model="gpt2",
            epochs=3,
            batch_size=4,
            learning_rate=5e-5
        )
        
        # Should not raise any exception
        config.validate()
    
    def test_validation_invalid_base_model(self):
        """Test validation failure for invalid base model."""
        with pytest.raises(ValidationError) as exc_info:
            TrainingConfig(base_model="invalid-model")
        
        assert "Unsupported base model" in str(exc_info.value)
    
    def test_validation_invalid_epochs(self):
        """Test validation failure for invalid epochs."""
        with pytest.raises(ValidationError) as exc_info:
            TrainingConfig(epochs=0)
        
        assert "epochs must be positive" in str(exc_info.value)
    
    def test_validation_invalid_batch_size(self):
        """Test validation failure for invalid batch size."""
        with pytest.raises(ValidationError) as exc_info:
            TrainingConfig(batch_size=-1)
        
        assert "batch_size must be positive" in str(exc_info.value)
    
    def test_validation_invalid_learning_rate(self):
        """Test validation failure for invalid learning rate."""
        with pytest.raises(ValidationError) as exc_info:
            TrainingConfig(learning_rate=1.5)
        
        assert "learning_rate must be between 0 and 1" in str(exc_info.value)
    
    def test_validation_invalid_lora_params(self):
        """Test validation failure for invalid LoRA parameters."""
        with pytest.raises(ValidationError) as exc_info:
            TrainingConfig(use_lora=True, lora_rank=0)
        
        assert "lora_rank must be positive when using LoRA" in str(exc_info.value)
    
    def test_validation_conflicting_quantization(self):
        """Test validation failure for conflicting quantization settings."""
        with pytest.raises(ValidationError) as exc_info:
            TrainingConfig(use_8bit=True, use_4bit=True)
        
        assert "Cannot use both 8-bit and 4-bit quantization" in str(exc_info.value)
    
    def test_validation_conflicting_precision(self):
        """Test validation failure for conflicting precision settings."""
        with pytest.raises(ValidationError) as exc_info:
            TrainingConfig(fp16=True, bf16=True)
        
        assert "Cannot use both fp16 and bf16" in str(exc_info.value)
    
    def test_validation_multiple_errors(self):
        """Test validation with multiple errors."""
        with pytest.raises(ValidationError) as exc_info:
            TrainingConfig(
                epochs=0,
                batch_size=-1,
                learning_rate=2.0
            )
        
        error_msg = str(exc_info.value)
        assert "epochs must be positive" in error_msg
        assert "batch_size must be positive" in error_msg
        assert "learning_rate must be between 0 and 1" in error_msg
    
    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = TrainingConfig(
            base_model="gpt2-medium",
            epochs=5,
            batch_size=8
        )
        
        config_dict = config.to_dict()
        
        assert config_dict["base_model"] == "gpt2-medium"
        assert config_dict["epochs"] == 5
        assert config_dict["batch_size"] == 8
        assert "SUPPORTED_MODELS" not in config_dict
    
    def test_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "base_model": "gpt2-medium",
            "epochs": 5,
            "batch_size": 8,
            "learning_rate": 3e-5,
            "unknown_param": "should_be_ignored"
        }
        
        config = TrainingConfig.from_dict(config_dict)
        
        assert config.base_model == "gpt2-medium"
        assert config.epochs == 5
        assert config.batch_size == 8
        assert config.learning_rate == 3e-5
        # Unknown parameter should be ignored
        assert not hasattr(config, "unknown_param")
    
    def test_to_training_args_dict(self):
        """Test converting to TrainingArguments format."""
        config = TrainingConfig(
            epochs=5,
            batch_size=8,
            learning_rate=3e-5
        )
        
        training_args = config.to_training_args_dict()
        
        assert training_args["num_train_epochs"] == 5
        assert training_args["per_device_train_batch_size"] == 8
        assert training_args["learning_rate"] == 3e-5
        assert "output_dir" in training_args
        assert "save_steps" in training_args
    
    def test_save_load_json(self):
        """Test saving and loading config as JSON."""
        config = TrainingConfig(
            base_model="gpt2-medium",
            epochs=5,
            batch_size=8
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            config.save(temp_path)
            loaded_config = TrainingConfig.load(temp_path)
            
            assert loaded_config.base_model == config.base_model
            assert loaded_config.epochs == config.epochs
            assert loaded_config.batch_size == config.batch_size
        finally:
            Path(temp_path).unlink()
    
    def test_save_load_yaml(self):
        """Test saving and loading config as YAML."""
        config = TrainingConfig(
            base_model="gpt2-medium",
            epochs=5,
            batch_size=8
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name
        
        try:
            config.save(temp_path)
            loaded_config = TrainingConfig.load(temp_path)
            
            assert loaded_config.base_model == config.base_model
            assert loaded_config.epochs == config.epochs
            assert loaded_config.batch_size == config.batch_size
        finally:
            Path(temp_path).unlink()
    
    def test_save_unsupported_format(self):
        """Test saving with unsupported file format."""
        config = TrainingConfig()
        
        with pytest.raises(ValueError) as exc_info:
            config.save("config.txt")
        
        assert "Unsupported file format" in str(exc_info.value)
    
    def test_load_nonexistent_file(self):
        """Test loading from nonexistent file."""
        with pytest.raises(FileNotFoundError):
            TrainingConfig.load("nonexistent.json")
    
    def test_load_unsupported_format(self):
        """Test loading from unsupported file format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            temp_path = f.name
            f.write("some content")
        
        try:
            with pytest.raises(ValueError) as exc_info:
                TrainingConfig.load(temp_path)
            
            assert "Unsupported file format" in str(exc_info.value)
        finally:
            Path(temp_path).unlink()


class TestConfigurationLoader:
    """Test cases for ConfigurationLoader utility class."""
    
    def test_get_default_config(self):
        """Test getting default configuration."""
        config = ConfigurationLoader.get_default_config()
        
        assert isinstance(config, TrainingConfig)
        assert config.base_model == "gpt2"
        assert config.epochs == 3
        assert config.batch_size == 4
    
    def test_get_quick_test_config(self):
        """Test getting quick test configuration."""
        config = ConfigurationLoader.get_quick_test_config()
        
        assert isinstance(config, TrainingConfig)
        assert config.epochs == 1
        assert config.batch_size == 2
        assert config.save_steps == 50
        assert config.use_lora is True
        assert config.lora_rank == 8
    
    def test_get_production_config(self):
        """Test getting production configuration."""
        config = ConfigurationLoader.get_production_config()
        
        assert isinstance(config, TrainingConfig)
        assert config.epochs == 5
        assert config.batch_size == 8
        assert config.gradient_accumulation_steps == 2
        assert config.fp16 is True
        assert config.gradient_checkpointing is True
    
    def test_get_low_resource_config(self):
        """Test getting low resource configuration."""
        config = ConfigurationLoader.get_low_resource_config()
        
        assert isinstance(config, TrainingConfig)
        assert config.batch_size == 1
        assert config.gradient_accumulation_steps == 8
        assert config.use_8bit is True
        assert config.gradient_checkpointing is True
    
    def test_get_config_for_small_model(self):
        """Test getting configuration for small model."""
        config = ConfigurationLoader.get_config_for_model("gpt2")
        
        assert config.base_model == "gpt2"
        assert config.batch_size == 8
        assert config.gradient_accumulation_steps == 1
    
    def test_get_config_for_medium_model(self):
        """Test getting configuration for medium model."""
        config = ConfigurationLoader.get_config_for_model("gpt2-medium")
        
        assert config.base_model == "gpt2-medium"
        assert config.batch_size == 4
        assert config.gradient_accumulation_steps == 2
        assert config.use_lora is True
    
    def test_get_config_for_large_model(self):
        """Test getting configuration for large model."""
        config = ConfigurationLoader.get_config_for_model("gpt2-large")
        
        assert config.base_model == "gpt2-large"
        assert config.batch_size == 2
        assert config.gradient_accumulation_steps == 4
        assert config.use_lora is True
        assert config.use_8bit is True
        assert config.gradient_checkpointing is True


class TestConfigCompatibility:
    """Test cases for configuration compatibility validation."""
    
    def test_no_warnings_for_good_config(self):
        """Test no warnings for well-configured setup."""
        config = TrainingConfig()
        warnings = validate_config_compatibility(config)
        
        assert len(warnings) == 0
    
    def test_warning_for_8bit_with_fp16(self):
        """Test warning for 8-bit quantization with fp16."""
        config = TrainingConfig(use_8bit=True, fp16=True)
        warnings = validate_config_compatibility(config)
        
        assert len(warnings) == 1
        assert "8-bit quantization with fp16" in warnings[0]
    
    def test_warning_for_large_effective_batch_size(self):
        """Test warning for very large effective batch size."""
        config = TrainingConfig(batch_size=16, gradient_accumulation_steps=4)
        warnings = validate_config_compatibility(config)
        
        assert len(warnings) == 1
        assert "Effective batch size is very large" in warnings[0]
    
    def test_warning_for_high_lr_without_lora(self):
        """Test warning for high learning rate without LoRA."""
        config = TrainingConfig(learning_rate=2e-4, use_lora=False)
        warnings = validate_config_compatibility(config)
        
        assert len(warnings) == 1
        assert "High learning rate without LoRA" in warnings[0]
    
    def test_warning_for_gradient_checkpointing_with_workers(self):
        """Test warning for gradient checkpointing with multiple workers."""
        config = TrainingConfig(
            gradient_checkpointing=True, 
            dataloader_num_workers=4
        )
        warnings = validate_config_compatibility(config)
        
        assert len(warnings) == 1
        assert "Gradient checkpointing with multiple workers" in warnings[0]
    
    def test_multiple_warnings(self):
        """Test multiple warnings for problematic config."""
        config = TrainingConfig(
            use_8bit=True,
            fp16=True,
            batch_size=20,
            gradient_accumulation_steps=2,
            learning_rate=2e-4,
            use_lora=False
        )
        warnings = validate_config_compatibility(config)
        
        assert len(warnings) >= 2  # Should have multiple warnings


if __name__ == "__main__":
    pytest.main([__file__])