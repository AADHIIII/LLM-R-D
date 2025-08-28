"""
Integration tests for the fine-tuning service.

Tests the complete training pipeline including job management, progress tracking,
and model training with LoRA optimization.
"""

import pytest
import tempfile
import json
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

from fine_tuning.fine_tuning_service import FineTuningService, TrainingJob
from fine_tuning.training_config import TrainingConfig
from utils.exceptions import TrainingError, ValidationError


class TestTrainingJob:
    """Test cases for TrainingJob dataclass."""
    
    def test_training_job_creation(self):
        """Test creating a training job."""
        config = TrainingConfig()
        job = TrainingJob(
            job_id="test-job-1",
            config=config,
            dataset_path="/path/to/dataset.jsonl"
        )
        
        assert job.job_id == "test-job-1"
        assert job.config == config
        assert job.dataset_path == "/path/to/dataset.jsonl"
        assert job.status == "pending"
        assert job.logs == []
    
    def test_training_job_to_dict(self):
        """Test converting training job to dictionary."""
        config = TrainingConfig(epochs=5, batch_size=8)
        job = TrainingJob(
            job_id="test-job-1",
            config=config,
            dataset_path="/path/to/dataset.jsonl",
            status="running",
            current_epoch=2,
            total_epochs=5
        )
        
        job_dict = job.to_dict()
        
        assert job_dict["job_id"] == "test-job-1"
        assert job_dict["status"] == "running"
        assert job_dict["current_epoch"] == 2
        assert job_dict["total_epochs"] == 5
        assert isinstance(job_dict["config"], dict)
        assert job_dict["config"]["epochs"] == 5


class TestFineTuningService:
    """Test cases for FineTuningService class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.service = FineTuningService(base_output_dir=self.temp_dir)
        
        # Create a mock dataset file
        self.dataset_path = Path(self.temp_dir) / "test_dataset.jsonl"
        with open(self.dataset_path, 'w') as f:
            f.write('{"prompt": "Hello", "response": "Hi there!"}\n')
            f.write('{"prompt": "How are you?", "response": "I am fine, thank you!"}\n')
    
    def test_service_initialization(self):
        """Test service initialization."""
        assert isinstance(self.service, FineTuningService)
        assert Path(self.service.base_output_dir).exists()
        assert len(self.service.active_jobs) == 0
    
    def test_create_training_job(self):
        """Test creating a training job."""
        config = TrainingConfig(epochs=3, batch_size=4)
        
        job = self.service.create_training_job(
            config=config,
            dataset_path=str(self.dataset_path),
            job_id="test-job-1"
        )
        
        assert job.job_id == "test-job-1"
        assert job.config == config
        assert job.dataset_path == str(self.dataset_path)
        assert job.status == "pending"
        assert "test-job-1" in self.service.active_jobs
    
    def test_create_training_job_auto_id(self):
        """Test creating a training job with auto-generated ID."""
        config = TrainingConfig()
        
        job = self.service.create_training_job(
            config=config,
            dataset_path=str(self.dataset_path)
        )
        
        assert job.job_id is not None
        assert len(job.job_id) > 0
        assert job.job_id in self.service.active_jobs
    
    def test_validate_training_setup_success(self):
        """Test successful training setup validation."""
        config = TrainingConfig()
        job = TrainingJob(
            job_id="test-job-1",
            config=config,
            dataset_path=str(self.dataset_path)
        )
        
        # Should not raise any exception
        self.service.validate_training_setup(job)
    
    def test_validate_training_setup_missing_dataset(self):
        """Test validation failure for missing dataset."""
        config = TrainingConfig()
        job = TrainingJob(
            job_id="test-job-1",
            config=config,
            dataset_path="/nonexistent/dataset.jsonl"
        )
        
        with pytest.raises(ValidationError) as exc_info:
            self.service.validate_training_setup(job)
        
        assert "Dataset file not found" in str(exc_info.value)
    
    def test_validate_training_setup_invalid_config(self):
        """Test validation failure for invalid configuration."""
        # Test that invalid config raises error during creation
        with pytest.raises(ValidationError):
            config = TrainingConfig(epochs=0)  # Invalid - should raise error here
    
    def test_start_training_mock(self):
        """Test starting training with mock implementation."""
        config = TrainingConfig(epochs=1, batch_size=2)
        job = self.service.create_training_job(
            config=config,
            dataset_path=str(self.dataset_path),
            job_id="test-job-1"
        )
        
        # Start training (should use mock implementation)
        completed_job = self.service.start_training("test-job-1")
        
        assert completed_job.status == "completed"
        assert completed_job.start_time is not None
        assert completed_job.end_time is not None
        assert completed_job.model_path is not None
    
    def test_start_training_nonexistent_job(self):
        """Test starting training for nonexistent job."""
        with pytest.raises(TrainingError) as exc_info:
            self.service.start_training("nonexistent-job")
        
        assert "Job nonexistent-job not found" in str(exc_info.value)
    
    def test_get_job_status(self):
        """Test getting job status."""
        config = TrainingConfig()
        job = self.service.create_training_job(
            config=config,
            dataset_path=str(self.dataset_path),
            job_id="test-job-1"
        )
        
        retrieved_job = self.service.get_job_status("test-job-1")
        assert retrieved_job is not None
        assert retrieved_job.job_id == "test-job-1"
        
        # Test nonexistent job
        assert self.service.get_job_status("nonexistent") is None
    
    def test_cancel_job(self):
        """Test cancelling a job."""
        config = TrainingConfig()
        job = self.service.create_training_job(
            config=config,
            dataset_path=str(self.dataset_path),
            job_id="test-job-1"
        )
        
        # Set job to running status
        job.status = "running"
        
        # Cancel the job
        result = self.service.cancel_job("test-job-1")
        assert result is True
        assert job.status == "cancelled"
        assert job.end_time is not None
        
        # Try to cancel nonexistent job
        assert self.service.cancel_job("nonexistent") is False
    
    def test_cleanup_job(self):
        """Test cleaning up a completed job."""
        config = TrainingConfig()
        job = self.service.create_training_job(
            config=config,
            dataset_path=str(self.dataset_path),
            job_id="test-job-1"
        )
        
        # Set job to completed status
        job.status = "completed"
        
        # Cleanup the job
        result = self.service.cleanup_job("test-job-1")
        assert result is True
        assert "test-job-1" not in self.service.active_jobs
        
        # Try to cleanup nonexistent job
        assert self.service.cleanup_job("nonexistent") is False
    
    def test_list_active_jobs(self):
        """Test listing active jobs."""
        config = TrainingConfig()
        
        # Initially no jobs
        assert len(self.service.list_active_jobs()) == 0
        
        # Create some jobs
        job1 = self.service.create_training_job(
            config=config,
            dataset_path=str(self.dataset_path),
            job_id="job-1"
        )
        job2 = self.service.create_training_job(
            config=config,
            dataset_path=str(self.dataset_path),
            job_id="job-2"
        )
        
        active_jobs = self.service.list_active_jobs()
        assert len(active_jobs) == 2
        job_ids = [job.job_id for job in active_jobs]
        assert "job-1" in job_ids
        assert "job-2" in job_ids
    
    def test_save_load_job_state(self):
        """Test saving and loading job state."""
        config = TrainingConfig(epochs=5, batch_size=8)
        job = self.service.create_training_job(
            config=config,
            dataset_path=str(self.dataset_path),
            job_id="test-job-1"
        )
        
        # Update job state
        job.status = "running"
        job.current_epoch = 2
        job.train_loss = 0.5
        
        # Save job state
        state_file = Path(self.temp_dir) / "job_state.json"
        self.service.save_job_state("test-job-1", str(state_file))
        
        assert state_file.exists()
        
        # Clear active jobs and load state
        self.service.active_jobs.clear()
        loaded_job = self.service.load_job_state(str(state_file))
        
        assert loaded_job.job_id == "test-job-1"
        assert loaded_job.status == "running"
        assert loaded_job.current_epoch == 2
        assert loaded_job.train_loss == 0.5
        assert loaded_job.config.epochs == 5
        assert "test-job-1" in self.service.active_jobs
    
    def test_save_job_state_nonexistent(self):
        """Test saving state for nonexistent job."""
        state_file = Path(self.temp_dir) / "job_state.json"
        
        with pytest.raises(ValueError) as exc_info:
            self.service.save_job_state("nonexistent", str(state_file))
        
        assert "Job nonexistent not found" in str(exc_info.value)


class TestTrainingProgressCallback:
    """Test cases for TrainingProgressCallback."""
    
    def test_callback_initialization(self):
        """Test callback initialization."""
        from fine_tuning.fine_tuning_service import TrainingProgressCallback
        import logging
        
        config = TrainingConfig()
        job = TrainingJob(
            job_id="test-job-1",
            config=config,
            dataset_path="/path/to/dataset.jsonl"
        )
        logger = logging.getLogger("test")
        
        callback = TrainingProgressCallback(job, logger)
        
        assert callback.job == job
        assert callback.logger == logger
        assert callback.start_time > 0
    
    def test_callback_training_begin(self):
        """Test callback on training begin."""
        from fine_tuning.fine_tuning_service import TrainingProgressCallback
        import logging
        
        config = TrainingConfig()
        job = TrainingJob(
            job_id="test-job-1",
            config=config,
            dataset_path="/path/to/dataset.jsonl"
        )
        logger = logging.getLogger("test")
        callback = TrainingProgressCallback(job, logger)
        
        # Mock training arguments and state
        args = MagicMock()
        args.num_train_epochs = 5
        state = MagicMock()
        state.max_steps = 1000
        
        callback.on_train_begin(args, state, None)
        
        assert job.status == "running"
        assert job.start_time is not None
        assert job.total_epochs == 5
        assert job.total_steps == 1000
    
    def test_callback_epoch_begin(self):
        """Test callback on epoch begin."""
        from fine_tuning.fine_tuning_service import TrainingProgressCallback
        import logging
        
        config = TrainingConfig()
        job = TrainingJob(
            job_id="test-job-1",
            config=config,
            dataset_path="/path/to/dataset.jsonl"
        )
        logger = logging.getLogger("test")
        callback = TrainingProgressCallback(job, logger)
        
        # Mock state
        state = MagicMock()
        state.epoch = 2.5
        
        callback.on_epoch_begin(None, state, None)
        
        assert job.current_epoch == 2
    
    def test_callback_step_end(self):
        """Test callback on step end."""
        from fine_tuning.fine_tuning_service import TrainingProgressCallback
        import logging
        
        config = TrainingConfig()
        job = TrainingJob(
            job_id="test-job-1",
            config=config,
            dataset_path="/path/to/dataset.jsonl"
        )
        logger = logging.getLogger("test")
        callback = TrainingProgressCallback(job, logger)
        
        # Mock state
        state = MagicMock()
        state.global_step = 100
        state.log_history = [{"train_loss": 0.5}]
        
        callback.on_step_end(None, state, None)
        
        assert job.current_step == 100
        assert job.train_loss == 0.5


if __name__ == "__main__":
    pytest.main([__file__])