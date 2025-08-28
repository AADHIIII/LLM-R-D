"""
Fine-tuning service for training GPT-style models.

This module provides the FineTuningService class that handles the complete
fine-tuning pipeline using Hugging Face Transformers with LoRA optimization.
"""

import os
import json
import logging
import time
from typing import Dict, Any, Optional, Callable, List, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
import uuid

try:
    import torch
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, TrainingArguments, 
        Trainer, DataCollatorForLanguageModeling, EarlyStoppingCallback
    )
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
    import bitsandbytes as bnb
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    import warnings
    warnings.warn("Transformers library not available. Using mock implementations for testing.")
    # Mock Dataset for type hints
    Dataset = Any

from fine_tuning.training_config import TrainingConfig
from fine_tuning.dataset_validator import DatasetValidator
from fine_tuning.dataset_tokenizer import DatasetTokenizer
from utils.exceptions import TrainingError, ValidationError
import logging


@dataclass
class TrainingJob:
    """Represents a training job with metadata and status."""
    
    job_id: str
    config: TrainingConfig
    dataset_path: str
    status: str = "pending"  # pending, running, completed, failed, cancelled
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    current_epoch: int = 0
    total_epochs: int = 0
    current_step: int = 0
    total_steps: int = 0
    train_loss: float = 0.0
    eval_loss: Optional[float] = None
    best_metric: Optional[float] = None
    model_path: Optional[str] = None
    error_message: Optional[str] = None
    logs: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.logs is None:
            self.logs = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        # Convert datetime objects to strings
        if self.start_time:
            data['start_time'] = self.start_time.isoformat()
        if self.end_time:
            data['end_time'] = self.end_time.isoformat()
        # Convert config to dict
        data['config'] = self.config.to_dict()
        return data


class TrainingProgressCallback:
    """Callback for tracking training progress."""
    
    def __init__(self, job: TrainingJob, logger: logging.Logger):
        self.job = job
        self.logger = logger
        self.start_time = time.time()
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Called at the beginning of training."""
        self.job.status = "running"
        self.job.start_time = datetime.now()
        self.job.total_epochs = int(args.num_train_epochs)
        self.job.total_steps = state.max_steps
        self.logger.info(f"Training started for job {self.job.job_id}")
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        """Called at the beginning of each epoch."""
        self.job.current_epoch = int(state.epoch)
        self.logger.info(f"Starting epoch {self.job.current_epoch}/{self.job.total_epochs}")
    
    def on_step_end(self, args, state, control, **kwargs):
        """Called at the end of each training step."""
        self.job.current_step = state.global_step
        
        # Update loss if available
        if len(state.log_history) > 0:
            latest_log = state.log_history[-1]
            if 'train_loss' in latest_log:
                self.job.train_loss = latest_log['train_loss']
            if 'eval_loss' in latest_log:
                self.job.eval_loss = latest_log['eval_loss']
    
    def on_evaluate(self, args, state, control, **kwargs):
        """Called after evaluation."""
        if len(state.log_history) > 0:
            latest_log = state.log_history[-1]
            if 'eval_loss' in latest_log:
                self.job.eval_loss = latest_log['eval_loss']
                
                # Update best metric
                if self.job.best_metric is None or latest_log['eval_loss'] < self.job.best_metric:
                    self.job.best_metric = latest_log['eval_loss']
            
            # Log progress
            self.job.logs.append({
                'timestamp': datetime.now().isoformat(),
                'step': state.global_step,
                'epoch': state.epoch,
                **latest_log
            })
    
    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training."""
        self.job.end_time = datetime.now()
        self.job.status = "completed"
        
        duration = time.time() - self.start_time
        self.logger.info(f"Training completed for job {self.job.job_id} in {duration:.2f} seconds")


class FineTuningService:
    """
    Service for fine-tuning GPT-style models with LoRA optimization.
    
    This class handles the complete fine-tuning pipeline including dataset
    validation, model loading, training configuration, and progress tracking.
    """
    
    def __init__(self, base_output_dir: str = "./models"):
        """
        Initialize the fine-tuning service.
        
        Args:
            base_output_dir: Base directory for saving models and outputs.
        """
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.active_jobs: Dict[str, TrainingJob] = {}
        
        # Initialize components
        self.dataset_validator = DatasetValidator()
        # DatasetTokenizer will be initialized per job with specific config
        
        if not TRANSFORMERS_AVAILABLE:
            self.logger.warning("Transformers not available. Service will use mock implementations.")
    
    def create_training_job(
        self, 
        config: TrainingConfig, 
        dataset_path: str,
        job_id: Optional[str] = None
    ) -> TrainingJob:
        """
        Create a new training job.
        
        Args:
            config: Training configuration.
            dataset_path: Path to the training dataset.
            job_id: Optional job ID. If not provided, a UUID will be generated.
            
        Returns:
            TrainingJob instance.
        """
        if job_id is None:
            job_id = str(uuid.uuid4())
        
        job = TrainingJob(
            job_id=job_id,
            config=config,
            dataset_path=dataset_path
        )
        
        self.active_jobs[job_id] = job
        self.logger.info(f"Created training job {job_id}")
        
        return job
    
    def validate_training_setup(self, job: TrainingJob) -> None:
        """
        Validate the training setup before starting.
        
        Args:
            job: Training job to validate.
            
        Raises:
            ValidationError: If validation fails.
        """
        # Validate dataset
        if not Path(job.dataset_path).exists():
            raise ValidationError(f"Dataset file not found: {job.dataset_path}")
        
        validation_result = self.dataset_validator.validate_dataset(job.dataset_path)
        if not validation_result.is_valid:
            raise ValidationError(f"Dataset validation failed: {validation_result.error_message}")
        
        # Validate configuration
        job.config.validate()
        
        # Check GPU availability if using quantization
        if (job.config.use_8bit or job.config.use_4bit) and not torch.cuda.is_available():
            self.logger.warning("Quantization requested but CUDA not available. Disabling quantization.")
            job.config.use_8bit = False
            job.config.use_4bit = False
        
        self.logger.info(f"Training setup validation passed for job {job.job_id}")
    
    def prepare_model_and_tokenizer(self, config: TrainingConfig) -> Tuple[Any, Any]:
        """
        Load and prepare the model and tokenizer.
        
        Args:
            config: Training configuration.
            
        Returns:
            Tuple of (model, tokenizer).
        """
        if not TRANSFORMERS_AVAILABLE:
            # Return mock objects for testing
            return None, None
        
        self.logger.info(f"Loading model and tokenizer: {config.base_model}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Configure quantization
        quantization_config = None
        if config.use_8bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        elif config.use_4bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            quantization_config=quantization_config,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=torch.float16 if config.fp16 else None,
        )
        
        # Configure LoRA if enabled
        if config.use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
        
        return model, tokenizer
    
    def prepare_dataset(self, job: TrainingJob, tokenizer: Any) -> Dataset:
        """
        Prepare the dataset for training.
        
        Args:
            job: Training job containing dataset path.
            tokenizer: Tokenizer to use for processing.
            
        Returns:
            Processed dataset.
        """
        self.logger.info(f"Preparing dataset: {job.dataset_path}")
        
        if not TRANSFORMERS_AVAILABLE:
            # Return mock dataset for testing
            return {"train": None}
        
        # Create tokenization config
        from fine_tuning.dataset_tokenizer import TokenizationConfig, DatasetTokenizer
        tokenization_config = TokenizationConfig(
            model_name=job.config.base_model,
            max_length=job.config.model_max_length
        )
        
        # Create dataset tokenizer
        dataset_tokenizer = DatasetTokenizer(tokenization_config)
        
        # Load and tokenize dataset
        dataset = dataset_tokenizer.tokenize_dataset(
            job.dataset_path,
            tokenizer,
            max_length=job.config.model_max_length
        )
        
        # Split dataset if needed
        if job.config.evaluation_strategy != "no":
            dataset = dataset.train_test_split(test_size=0.1, seed=42)
            return dataset
        
        return {"train": dataset}
    
    def start_training(self, job_id: str) -> TrainingJob:
        """
        Start the training process for a job.
        
        Args:
            job_id: ID of the job to start.
            
        Returns:
            Updated TrainingJob.
            
        Raises:
            TrainingError: If training fails to start or encounters errors.
        """
        if job_id not in self.active_jobs:
            raise TrainingError(f"Job {job_id} not found")
        
        job = self.active_jobs[job_id]
        
        try:
            # Validate setup
            self.validate_training_setup(job)
            
            if not TRANSFORMERS_AVAILABLE:
                # Mock training for testing
                job.status = "running"
                job.start_time = datetime.now()
                time.sleep(1)  # Simulate training time
                job.status = "completed"
                job.end_time = datetime.now()
                job.model_path = str(self.base_output_dir / f"model_{job_id}")
                self.logger.info(f"Mock training completed for job {job_id}")
                return job
            
            # Prepare model and tokenizer
            model, tokenizer = self.prepare_model_and_tokenizer(job.config)
            
            # Prepare dataset
            dataset = self.prepare_dataset(job, tokenizer)
            
            # Set up output directory
            output_dir = self.base_output_dir / f"job_{job_id}"
            output_dir.mkdir(parents=True, exist_ok=True)
            job.config.output_dir = str(output_dir)
            
            # Create training arguments
            training_args_dict = job.config.to_training_args_dict()
            training_args = TrainingArguments(**training_args_dict)
            
            # Create data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,
            )
            
            # Create progress callback
            progress_callback = TrainingProgressCallback(job, self.logger)
            
            # Set up callbacks
            callbacks = [progress_callback]
            if job.config.load_best_model_at_end:
                callbacks.append(EarlyStoppingCallback(early_stopping_patience=3))
            
            # Create trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset["train"],
                eval_dataset=dataset.get("test"),
                data_collator=data_collator,
                callbacks=callbacks,
            )
            
            # Start training
            self.logger.info(f"Starting training for job {job_id}")
            trainer.train()
            
            # Save the model
            model_save_path = output_dir / "final_model"
            trainer.save_model(str(model_save_path))
            tokenizer.save_pretrained(str(model_save_path))
            
            # Save training configuration
            config_path = output_dir / "training_config.json"
            with open(config_path, 'w') as f:
                json.dump(job.config.to_dict(), f, indent=2)
            
            job.model_path = str(model_save_path)
            job.status = "completed"
            
            self.logger.info(f"Training completed successfully for job {job_id}")
            
        except Exception as e:
            job.status = "failed"
            job.error_message = str(e)
            job.end_time = datetime.now()
            self.logger.error(f"Training failed for job {job_id}: {e}")
            raise TrainingError(f"Training failed: {e}") from e
        
        return job
    
    def get_job_status(self, job_id: str) -> Optional[TrainingJob]:
        """
        Get the status of a training job.
        
        Args:
            job_id: ID of the job.
            
        Returns:
            TrainingJob if found, None otherwise.
        """
        return self.active_jobs.get(job_id)
    
    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a running training job.
        
        Args:
            job_id: ID of the job to cancel.
            
        Returns:
            True if job was cancelled, False if not found or not running.
        """
        if job_id not in self.active_jobs:
            return False
        
        job = self.active_jobs[job_id]
        if job.status == "running":
            job.status = "cancelled"
            job.end_time = datetime.now()
            self.logger.info(f"Job {job_id} cancelled")
            return True
        
        return False
    
    def cleanup_job(self, job_id: str) -> bool:
        """
        Clean up resources for a completed job.
        
        Args:
            job_id: ID of the job to clean up.
            
        Returns:
            True if cleanup was successful.
        """
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
            if job.status in ["completed", "failed", "cancelled"]:
                del self.active_jobs[job_id]
                self.logger.info(f"Cleaned up job {job_id}")
                return True
        
        return False
    
    def list_active_jobs(self) -> List[TrainingJob]:
        """
        List all active training jobs.
        
        Returns:
            List of active TrainingJob instances.
        """
        return list(self.active_jobs.values())
    
    def save_job_state(self, job_id: str, filepath: str) -> None:
        """
        Save job state to file for persistence.
        
        Args:
            job_id: ID of the job to save.
            filepath: Path to save the job state.
        """
        if job_id not in self.active_jobs:
            raise ValueError(f"Job {job_id} not found")
        
        job = self.active_jobs[job_id]
        with open(filepath, 'w') as f:
            json.dump(job.to_dict(), f, indent=2)
        
        self.logger.info(f"Job state saved for {job_id}")
    
    def load_job_state(self, filepath: str) -> TrainingJob:
        """
        Load job state from file.
        
        Args:
            filepath: Path to the job state file.
            
        Returns:
            Loaded TrainingJob.
        """
        with open(filepath, 'r') as f:
            job_data = json.load(f)
        
        # Reconstruct TrainingJob
        config = TrainingConfig.from_dict(job_data['config'])
        job = TrainingJob(
            job_id=job_data['job_id'],
            config=config,
            dataset_path=job_data['dataset_path'],
            status=job_data['status'],
            current_epoch=job_data.get('current_epoch', 0),
            total_epochs=job_data.get('total_epochs', 0),
            current_step=job_data.get('current_step', 0),
            total_steps=job_data.get('total_steps', 0),
            train_loss=job_data.get('train_loss', 0.0),
            eval_loss=job_data.get('eval_loss'),
            best_metric=job_data.get('best_metric'),
            model_path=job_data.get('model_path'),
            error_message=job_data.get('error_message'),
            logs=job_data.get('logs', [])
        )
        
        # Parse datetime strings
        if job_data.get('start_time'):
            job.start_time = datetime.fromisoformat(job_data['start_time'])
        if job_data.get('end_time'):
            job.end_time = datetime.fromisoformat(job_data['end_time'])
        
        self.active_jobs[job.job_id] = job
        self.logger.info(f"Job state loaded for {job.job_id}")
        
        return job