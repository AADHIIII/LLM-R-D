"""
Training configuration management for fine-tuning operations.

This module provides the TrainingConfig dataclass and related utilities for
managing training parameters, validation, and configuration loading.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import json
import yaml
from utils.exceptions import ValidationError


@dataclass
class TrainingConfig:
    """
    Configuration class for fine-tuning parameters with validation.
    
    This class encapsulates all training parameters needed for fine-tuning
    GPT-style models, with built-in validation and default values.
    """
    
    # Model configuration
    base_model: str = "gpt2"
    model_max_length: int = 512
    
    # Training parameters
    epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    warmup_ratio: float = 0.1
    
    # Optimization settings
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    use_8bit: bool = False
    use_4bit: bool = False
    
    # Training behavior
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 100
    max_grad_norm: float = 1.0
    dataloader_num_workers: int = 0
    
    # Output and checkpointing
    output_dir: str = "./models/fine_tuned"
    run_name: Optional[str] = None
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    
    # Evaluation
    evaluation_strategy: str = "steps"
    eval_accumulation_steps: Optional[int] = None
    
    # Advanced settings
    fp16: bool = False
    bf16: bool = False
    gradient_checkpointing: bool = True
    remove_unused_columns: bool = False
    
    # Custom parameters
    custom_params: Dict[str, Any] = field(default_factory=dict)
    
    # Supported base models
    SUPPORTED_MODELS: List[str] = field(default_factory=lambda: [
        "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl",
        "distilgpt2", "microsoft/DialoGPT-small", "microsoft/DialoGPT-medium"
    ])
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """
        Validate all configuration parameters.
        
        Raises:
            ValidationError: If any parameter is invalid.
        """
        errors = []
        
        # Validate base model
        if self.base_model not in self.SUPPORTED_MODELS:
            errors.append(f"Unsupported base model: {self.base_model}. "
                         f"Supported models: {', '.join(self.SUPPORTED_MODELS)}")
        
        # Validate numeric parameters
        if self.epochs <= 0:
            errors.append("epochs must be positive")
        
        if self.batch_size <= 0:
            errors.append("batch_size must be positive")
        
        if self.gradient_accumulation_steps <= 0:
            errors.append("gradient_accumulation_steps must be positive")
        
        if not (0 < self.learning_rate < 1):
            errors.append("learning_rate must be between 0 and 1")
        
        if not (0 <= self.weight_decay <= 1):
            errors.append("weight_decay must be between 0 and 1")
        
        if self.warmup_steps < 0:
            errors.append("warmup_steps must be non-negative")
        
        if not (0 <= self.warmup_ratio <= 1):
            errors.append("warmup_ratio must be between 0 and 1")
        
        if self.model_max_length <= 0:
            errors.append("model_max_length must be positive")
        
        # Validate LoRA parameters
        if self.use_lora:
            if self.lora_rank <= 0:
                errors.append("lora_rank must be positive when using LoRA")
            
            if self.lora_alpha <= 0:
                errors.append("lora_alpha must be positive when using LoRA")
            
            if not (0 <= self.lora_dropout <= 1):
                errors.append("lora_dropout must be between 0 and 1")
        
        # Validate quantization settings
        if self.use_8bit and self.use_4bit:
            errors.append("Cannot use both 8-bit and 4-bit quantization")
        
        # Validate checkpoint settings
        if self.save_steps <= 0:
            errors.append("save_steps must be positive")
        
        if self.eval_steps <= 0:
            errors.append("eval_steps must be positive")
        
        if self.logging_steps <= 0:
            errors.append("logging_steps must be positive")
        
        if self.save_total_limit <= 0:
            errors.append("save_total_limit must be positive")
        
        # Validate evaluation strategy
        valid_strategies = ["no", "steps", "epoch"]
        if self.evaluation_strategy not in valid_strategies:
            errors.append(f"evaluation_strategy must be one of: {', '.join(valid_strategies)}")
        
        # Validate metric for best model
        valid_metrics = ["eval_loss", "eval_accuracy", "eval_f1", "eval_bleu"]
        if self.metric_for_best_model not in valid_metrics:
            errors.append(f"metric_for_best_model must be one of: {', '.join(valid_metrics)}")
        
        # Validate precision settings
        if self.fp16 and self.bf16:
            errors.append("Cannot use both fp16 and bf16")
        
        if errors:
            raise ValidationError(f"Configuration validation failed: {'; '.join(errors)}")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary format.
        
        Returns:
            Dict containing all configuration parameters.
        """
        config_dict = {}
        for key, value in self.__dict__.items():
            if not key.startswith('_') and key != 'SUPPORTED_MODELS':
                config_dict[key] = value
        return config_dict
    
    def to_training_args_dict(self) -> Dict[str, Any]:
        """
        Convert to format suitable for Hugging Face TrainingArguments.
        
        Returns:
            Dict formatted for TrainingArguments initialization.
        """
        return {
            'output_dir': self.output_dir,
            'num_train_epochs': self.epochs,
            'per_device_train_batch_size': self.batch_size,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'warmup_steps': self.warmup_steps,
            'warmup_ratio': self.warmup_ratio,
            'save_steps': self.save_steps,
            'eval_steps': self.eval_steps,
            'logging_steps': self.logging_steps,
            'max_grad_norm': self.max_grad_norm,
            'dataloader_num_workers': self.dataloader_num_workers,
            'save_total_limit': self.save_total_limit,
            'load_best_model_at_end': self.load_best_model_at_end,
            'metric_for_best_model': self.metric_for_best_model,
            'greater_is_better': self.greater_is_better,
            'evaluation_strategy': self.evaluation_strategy,
            'eval_accumulation_steps': self.eval_accumulation_steps,
            'fp16': self.fp16,
            'bf16': self.bf16,
            'gradient_checkpointing': self.gradient_checkpointing,
            'remove_unused_columns': self.remove_unused_columns,
            'run_name': self.run_name,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """
        Create TrainingConfig from dictionary.
        
        Args:
            config_dict: Dictionary containing configuration parameters.
            
        Returns:
            TrainingConfig instance.
        """
        # Filter out unknown parameters
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_fields}
        
        return cls(**filtered_dict)
    
    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save configuration to file.
        
        Args:
            filepath: Path to save configuration file.
        """
        filepath = Path(filepath)
        config_dict = self.to_dict()
        
        if filepath.suffix.lower() == '.json':
            with open(filepath, 'w') as f:
                json.dump(config_dict, f, indent=2)
        elif filepath.suffix.lower() in ['.yml', '.yaml']:
            with open(filepath, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'TrainingConfig':
        """
        Load configuration from file.
        
        Args:
            filepath: Path to configuration file.
            
        Returns:
            TrainingConfig instance.
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        if filepath.suffix.lower() == '.json':
            with open(filepath, 'r') as f:
                config_dict = json.load(f)
        elif filepath.suffix.lower() in ['.yml', '.yaml']:
            with open(filepath, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        return cls.from_dict(config_dict)


class ConfigurationLoader:
    """
    Utility class for loading training configurations from various sources.
    """
    
    @staticmethod
    def get_default_config() -> TrainingConfig:
        """
        Get default training configuration.
        
        Returns:
            TrainingConfig with default parameters.
        """
        return TrainingConfig()
    
    @staticmethod
    def get_quick_test_config() -> TrainingConfig:
        """
        Get configuration optimized for quick testing.
        
        Returns:
            TrainingConfig with parameters for fast testing.
        """
        return TrainingConfig(
            epochs=1,
            batch_size=2,
            save_steps=50,
            eval_steps=50,
            logging_steps=10,
            warmup_steps=10,
            use_lora=True,
            lora_rank=8,
            gradient_checkpointing=False,
        )
    
    @staticmethod
    def get_production_config() -> TrainingConfig:
        """
        Get configuration optimized for production training.
        
        Returns:
            TrainingConfig with parameters for production use.
        """
        return TrainingConfig(
            epochs=5,
            batch_size=8,
            gradient_accumulation_steps=2,
            learning_rate=3e-5,
            warmup_ratio=0.05,
            save_steps=1000,
            eval_steps=1000,
            logging_steps=50,
            use_lora=True,
            lora_rank=32,
            lora_alpha=64,
            gradient_checkpointing=True,
            fp16=True,
        )
    
    @staticmethod
    def get_low_resource_config() -> TrainingConfig:
        """
        Get configuration optimized for low-resource environments.
        
        Returns:
            TrainingConfig with parameters for limited resources.
        """
        return TrainingConfig(
            epochs=3,
            batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=5e-5,
            use_lora=True,
            lora_rank=16,
            use_8bit=True,
            gradient_checkpointing=True,
            dataloader_num_workers=0,
        )
    
    @staticmethod
    def get_config_for_model(model_name: str) -> TrainingConfig:
        """
        Get configuration optimized for specific model.
        
        Args:
            model_name: Name of the base model.
            
        Returns:
            TrainingConfig optimized for the specified model.
        """
        base_config = TrainingConfig(base_model=model_name)
        
        # Adjust parameters based on model size
        if "large" in model_name or "xl" in model_name:
            # Larger models need smaller batch sizes and more aggressive optimization
            base_config.batch_size = 2
            base_config.gradient_accumulation_steps = 4
            base_config.use_lora = True
            base_config.use_8bit = True
            base_config.gradient_checkpointing = True
        elif "medium" in model_name:
            base_config.batch_size = 4
            base_config.gradient_accumulation_steps = 2
            base_config.use_lora = True
        else:
            # Small models can handle larger batch sizes
            base_config.batch_size = 8
            base_config.gradient_accumulation_steps = 1
        
        return base_config


def validate_config_compatibility(config: TrainingConfig) -> List[str]:
    """
    Check configuration compatibility and provide warnings.
    
    Args:
        config: TrainingConfig to validate.
        
    Returns:
        List of warning messages.
    """
    warnings = []
    
    # Check for potentially problematic combinations
    if config.use_8bit and config.fp16:
        warnings.append("Using 8-bit quantization with fp16 may cause instability")
    
    if config.batch_size * config.gradient_accumulation_steps > 32:
        warnings.append("Effective batch size is very large, consider reducing")
    
    if config.learning_rate > 1e-4 and not config.use_lora:
        warnings.append("High learning rate without LoRA may cause instability")
    
    if config.gradient_checkpointing and config.dataloader_num_workers > 0:
        warnings.append("Gradient checkpointing with multiple workers may cause issues")
    
    return warnings