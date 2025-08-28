"""
Tests for the model management system.

Tests model saving, loading, versioning, registry management,
and search/filtering capabilities.
"""

import pytest
import tempfile
import json
import shutil
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock

from fine_tuning.model_manager import (
    ModelManager, ModelMetadata, ModelRegistryEntry
)
from fine_tuning.training_config import TrainingConfig
from utils.exceptions import ModelLoadingError


class TestModelMetadata:
    """Test cases for ModelMetadata dataclass."""
    
    def test_metadata_creation(self):
        """Test creating model metadata."""
        now = datetime.now()
        metadata = ModelMetadata(
            model_id="test-model-1",
            name="test_model",
            version="v1",
            base_model="gpt2",
            training_config={"epochs": 3},
            dataset_info={"samples": 100},
            performance_metrics={"loss": 0.5},
            created_at=now,
            updated_at=now,
            model_size_mb=50.0
        )
        
        assert metadata.model_id == "test-model-1"
        assert metadata.name == "test_model"
        assert metadata.version == "v1"
        assert metadata.base_model == "gpt2"
        assert metadata.model_size_mb == 50.0
        assert metadata.tags == []
        assert metadata.custom_metadata == {}
    
    def test_metadata_to_dict(self):
        """Test converting metadata to dictionary."""
        now = datetime.now()
        metadata = ModelMetadata(
            model_id="test-model-1",
            name="test_model",
            version="v1",
            base_model="gpt2",
            training_config={"epochs": 3},
            dataset_info={"samples": 100},
            performance_metrics={"loss": 0.5},
            created_at=now,
            updated_at=now,
            model_size_mb=50.0,
            tags=["test", "gpt2"]
        )
        
        metadata_dict = metadata.to_dict()
        
        assert metadata_dict["model_id"] == "test-model-1"
        assert metadata_dict["name"] == "test_model"
        assert metadata_dict["tags"] == ["test", "gpt2"]
        assert isinstance(metadata_dict["created_at"], str)
        assert isinstance(metadata_dict["updated_at"], str)
    
    def test_metadata_from_dict(self):
        """Test creating metadata from dictionary."""
        now = datetime.now()
        metadata_dict = {
            "model_id": "test-model-1",
            "name": "test_model",
            "version": "v1",
            "base_model": "gpt2",
            "training_config": {"epochs": 3},
            "dataset_info": {"samples": 100},
            "performance_metrics": {"loss": 0.5},
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "model_size_mb": 50.0,
            "tags": ["test"],
            "custom_metadata": {"key": "value"}
        }
        
        metadata = ModelMetadata.from_dict(metadata_dict)
        
        assert metadata.model_id == "test-model-1"
        assert metadata.name == "test_model"
        assert isinstance(metadata.created_at, datetime)
        assert isinstance(metadata.updated_at, datetime)
        assert metadata.tags == ["test"]
        assert metadata.custom_metadata == {"key": "value"}


class TestModelRegistryEntry:
    """Test cases for ModelRegistryEntry dataclass."""
    
    def test_registry_entry_creation(self):
        """Test creating registry entry."""
        now = datetime.now()
        metadata = ModelMetadata(
            model_id="test-model-1",
            name="test_model",
            version="v1",
            base_model="gpt2",
            training_config={"epochs": 3},
            dataset_info={"samples": 100},
            performance_metrics={"loss": 0.5},
            created_at=now,
            updated_at=now,
            model_size_mb=50.0
        )
        
        entry = ModelRegistryEntry(
            model_id="test-model-1",
            name="test_model",
            version="v1",
            path="/path/to/model",
            metadata=metadata
        )
        
        assert entry.model_id == "test-model-1"
        assert entry.name == "test_model"
        assert entry.version == "v1"
        assert entry.path == "/path/to/model"
        assert entry.is_active is True
    
    def test_registry_entry_serialization(self):
        """Test registry entry to/from dict conversion."""
        now = datetime.now()
        metadata = ModelMetadata(
            model_id="test-model-1",
            name="test_model",
            version="v1",
            base_model="gpt2",
            training_config={"epochs": 3},
            dataset_info={"samples": 100},
            performance_metrics={"loss": 0.5},
            created_at=now,
            updated_at=now,
            model_size_mb=50.0
        )
        
        entry = ModelRegistryEntry(
            model_id="test-model-1",
            name="test_model",
            version="v1",
            path="/path/to/model",
            metadata=metadata,
            is_active=False
        )
        
        # Convert to dict and back
        entry_dict = entry.to_dict()
        restored_entry = ModelRegistryEntry.from_dict(entry_dict)
        
        assert restored_entry.model_id == entry.model_id
        assert restored_entry.name == entry.name
        assert restored_entry.version == entry.version
        assert restored_entry.path == entry.path
        assert restored_entry.is_active == entry.is_active
        assert restored_entry.metadata.model_id == entry.metadata.model_id


class TestModelManager:
    """Test cases for ModelManager class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = ModelManager(
            base_model_dir=self.temp_dir,
            registry_file="test_registry.json"
        )
        
        # Create mock training config
        self.training_config = TrainingConfig(
            base_model="gpt2",
            epochs=3,
            batch_size=4
        )
        
        self.dataset_info = {
            "name": "test_dataset",
            "samples": 100,
            "format": "jsonl"
        }
        
        self.performance_metrics = {
            "train_loss": 0.5,
            "eval_loss": 0.6,
            "perplexity": 2.0
        }
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_manager_initialization(self):
        """Test manager initialization."""
        assert isinstance(self.manager, ModelManager)
        assert Path(self.manager.base_model_dir).exists()
        assert len(self.manager.registry) == 0
    
    def test_save_model_mock(self):
        """Test saving a model with mock implementation."""
        model_id = self.manager.save_model(
            model=None,  # Mock model
            tokenizer=None,  # Mock tokenizer
            name="test_model",
            training_config=self.training_config,
            dataset_info=self.dataset_info,
            performance_metrics=self.performance_metrics,
            description="Test model for unit testing",
            tags=["test", "gpt2"],
            training_duration=120.5
        )
        
        assert model_id is not None
        assert model_id in self.manager.registry
        
        entry = self.manager.registry[model_id]
        assert entry.name == "test_model"
        assert entry.version == "v1"
        assert entry.metadata.description == "Test model for unit testing"
        assert entry.metadata.tags == ["test", "gpt2"]
        assert entry.metadata.training_duration_seconds == 120.5
        
        # Check that files were created
        model_path = Path(entry.path)
        assert model_path.exists()
        assert (model_path / "pytorch_model.bin").exists()
        assert (model_path / "config.json").exists()
        assert (model_path / "tokenizer.json").exists()
        assert (model_path / "training_config.json").exists()
        assert (model_path / "metadata.json").exists()
    
    def test_save_multiple_versions(self):
        """Test saving multiple versions of the same model."""
        # Save first version
        model_id_v1 = self.manager.save_model(
            model=None,
            tokenizer=None,
            name="test_model",
            training_config=self.training_config,
            dataset_info=self.dataset_info,
            performance_metrics=self.performance_metrics
        )
        
        # Save second version
        model_id_v2 = self.manager.save_model(
            model=None,
            tokenizer=None,
            name="test_model",
            training_config=self.training_config,
            dataset_info=self.dataset_info,
            performance_metrics={"train_loss": 0.4, "eval_loss": 0.5}
        )
        
        assert model_id_v1 != model_id_v2
        assert self.manager.registry[model_id_v1].version == "v1"
        assert self.manager.registry[model_id_v2].version == "v2"
    
    def test_load_model_mock(self):
        """Test loading a model with mock implementation."""
        # First save a model
        model_id = self.manager.save_model(
            model=None,
            tokenizer=None,
            name="test_model",
            training_config=self.training_config,
            dataset_info=self.dataset_info,
            performance_metrics=self.performance_metrics
        )
        
        # Load the model
        model, tokenizer, metadata = self.manager.load_model(model_id)
        
        # With mock implementation, model and tokenizer will be None
        assert model is None
        assert tokenizer is None
        assert isinstance(metadata, ModelMetadata)
        assert metadata.model_id == model_id
        assert metadata.name == "test_model"
    
    def test_load_nonexistent_model(self):
        """Test loading a nonexistent model."""
        with pytest.raises(ModelLoadingError) as exc_info:
            self.manager.load_model("nonexistent-model")
        
        assert "not found in registry" in str(exc_info.value)
    
    def test_list_models_empty(self):
        """Test listing models when registry is empty."""
        models = self.manager.list_models()
        assert len(models) == 0
    
    def test_list_models_with_filters(self):
        """Test listing models with various filters."""
        # Save some test models
        model_id_1 = self.manager.save_model(
            model=None, tokenizer=None, name="gpt2_model",
            training_config=TrainingConfig(base_model="gpt2"),
            dataset_info=self.dataset_info,
            performance_metrics=self.performance_metrics,
            tags=["gpt2", "small"]
        )
        
        model_id_2 = self.manager.save_model(
            model=None, tokenizer=None, name="distilgpt2_model",
            training_config=TrainingConfig(base_model="distilgpt2"),
            dataset_info=self.dataset_info,
            performance_metrics=self.performance_metrics,
            tags=["distilgpt2", "fast"]
        )
        
        model_id_3 = self.manager.save_model(
            model=None, tokenizer=None, name="another_gpt2",
            training_config=TrainingConfig(base_model="gpt2"),
            dataset_info=self.dataset_info,
            performance_metrics=self.performance_metrics,
            tags=["gpt2", "large"]
        )
        
        # Test no filter
        all_models = self.manager.list_models()
        assert len(all_models) == 3
        
        # Test name filter
        gpt2_models = self.manager.list_models(name_filter="gpt2")
        assert len(gpt2_models) == 3  # All models contain "gpt2" in their names
        
        # Test base model filter
        distil_models = self.manager.list_models(base_model_filter="distilgpt2")
        assert len(distil_models) == 1
        assert distil_models[0].model_id == model_id_2
        
        # Test tag filter
        small_models = self.manager.list_models(tag_filter=["small"])
        assert len(small_models) == 1
        assert small_models[0].model_id == model_id_1
        
        # Test multiple tag filter
        gpt2_large = self.manager.list_models(tag_filter=["gpt2", "large"])
        assert len(gpt2_large) == 1
        assert gpt2_large[0].model_id == model_id_3
    
    def test_get_model_info(self):
        """Test getting model information."""
        model_id = self.manager.save_model(
            model=None, tokenizer=None, name="test_model",
            training_config=self.training_config,
            dataset_info=self.dataset_info,
            performance_metrics=self.performance_metrics
        )
        
        info = self.manager.get_model_info(model_id)
        assert info is not None
        assert info.model_id == model_id
        assert info.name == "test_model"
        
        # Test nonexistent model
        assert self.manager.get_model_info("nonexistent") is None
    
    def test_delete_model_soft(self):
        """Test soft deletion (marking as inactive)."""
        model_id = self.manager.save_model(
            model=None, tokenizer=None, name="test_model",
            training_config=self.training_config,
            dataset_info=self.dataset_info,
            performance_metrics=self.performance_metrics
        )
        
        # Soft delete
        result = self.manager.delete_model(model_id, permanent=False)
        assert result is True
        
        # Model should still exist but be inactive
        assert model_id in self.manager.registry
        assert self.manager.registry[model_id].is_active is False
        
        # Should not appear in active model list
        active_models = self.manager.list_models(active_only=True)
        assert len(active_models) == 0
        
        # Should appear in all models list
        all_models = self.manager.list_models(active_only=False)
        assert len(all_models) == 1
    
    def test_delete_model_permanent(self):
        """Test permanent deletion."""
        model_id = self.manager.save_model(
            model=None, tokenizer=None, name="test_model",
            training_config=self.training_config,
            dataset_info=self.dataset_info,
            performance_metrics=self.performance_metrics
        )
        
        model_path = Path(self.manager.registry[model_id].path)
        assert model_path.exists()
        
        # Permanent delete
        result = self.manager.delete_model(model_id, permanent=True)
        assert result is True
        
        # Model should be completely removed
        assert model_id not in self.manager.registry
        assert not model_path.exists()
    
    def test_update_model_metadata(self):
        """Test updating model metadata."""
        model_id = self.manager.save_model(
            model=None, tokenizer=None, name="test_model",
            training_config=self.training_config,
            dataset_info=self.dataset_info,
            performance_metrics=self.performance_metrics,
            description="Original description",
            tags=["original"]
        )
        
        # Update metadata
        result = self.manager.update_model_metadata(
            model_id,
            description="Updated description",
            tags=["updated", "new"],
            custom_metadata={"key": "value"}
        )
        
        assert result is True
        
        entry = self.manager.registry[model_id]
        assert entry.metadata.description == "Updated description"
        assert entry.metadata.tags == ["updated", "new"]
        assert entry.metadata.custom_metadata["key"] == "value"
        
        # Test nonexistent model
        result = self.manager.update_model_metadata("nonexistent")
        assert result is False
    
    def test_compare_models(self):
        """Test model comparison functionality."""
        # Save some models with different metrics
        model_id_1 = self.manager.save_model(
            model=None, tokenizer=None, name="model_1",
            training_config=self.training_config,
            dataset_info=self.dataset_info,
            performance_metrics={"train_loss": 0.5, "eval_loss": 0.6}
        )
        
        model_id_2 = self.manager.save_model(
            model=None, tokenizer=None, name="model_2",
            training_config=self.training_config,
            dataset_info=self.dataset_info,
            performance_metrics={"train_loss": 0.4, "eval_loss": 0.5}
        )
        
        comparison = self.manager.compare_models([model_id_1, model_id_2])
        
        assert "models" in comparison
        assert "metrics_comparison" in comparison
        assert "summary" in comparison
        
        assert len(comparison["models"]) == 2
        assert model_id_1 in comparison["models"]
        assert model_id_2 in comparison["models"]
        
        # Check metrics comparison
        assert "train_loss" in comparison["metrics_comparison"]
        assert "eval_loss" in comparison["metrics_comparison"]
        
        train_loss_comparison = comparison["metrics_comparison"]["train_loss"]
        assert train_loss_comparison["best"] == 0.4  # Lower is better for loss
        assert train_loss_comparison["worst"] == 0.5
        assert train_loss_comparison["average"] == 0.45
    
    def test_get_registry_stats(self):
        """Test getting registry statistics."""
        # Initially empty
        stats = self.manager.get_registry_stats()
        assert stats["total_models"] == 0
        assert stats["active_models"] == 0
        
        # Add some models
        model_id_1 = self.manager.save_model(
            model=None, tokenizer=None, name="gpt2_model",
            training_config=TrainingConfig(base_model="gpt2"),
            dataset_info=self.dataset_info,
            performance_metrics=self.performance_metrics
        )
        
        model_id_2 = self.manager.save_model(
            model=None, tokenizer=None, name="distilgpt2_model",
            training_config=TrainingConfig(base_model="distilgpt2"),
            dataset_info=self.dataset_info,
            performance_metrics=self.performance_metrics
        )
        
        # Soft delete one model
        self.manager.delete_model(model_id_2, permanent=False)
        
        stats = self.manager.get_registry_stats()
        assert stats["total_models"] == 2
        assert stats["active_models"] == 1
        assert stats["inactive_models"] == 1
        assert "gpt2" in stats["base_models"]
        assert stats["base_models"]["gpt2"] == 1
        assert stats["total_size_mb"] > 0
    
    def test_registry_persistence(self):
        """Test that registry is persisted across manager instances."""
        # Save a model
        model_id = self.manager.save_model(
            model=None, tokenizer=None, name="persistent_model",
            training_config=self.training_config,
            dataset_info=self.dataset_info,
            performance_metrics=self.performance_metrics
        )
        
        # Create new manager instance with same directory
        new_manager = ModelManager(
            base_model_dir=self.temp_dir,
            registry_file="test_registry.json"
        )
        
        # Should load the existing registry
        assert len(new_manager.registry) == 1
        assert model_id in new_manager.registry
        assert new_manager.registry[model_id].name == "persistent_model"


if __name__ == "__main__":
    pytest.main([__file__])