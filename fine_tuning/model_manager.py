"""
Model management system for fine-tuned models.

This module provides the ModelManager class for saving, loading, versioning,
and managing fine-tuned models with comprehensive metadata storage.
"""

import os
import json
import shutil
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
import uuid
import hashlib

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel, PeftConfig
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    import warnings
    warnings.warn("Transformers library not available. Using mock implementations for testing.")

from fine_tuning.training_config import TrainingConfig
from utils.exceptions import ModelLoadingError, ValidationError


@dataclass
class ModelMetadata:
    """Metadata for a fine-tuned model."""
    
    model_id: str
    name: str
    version: str
    base_model: str
    training_config: Dict[str, Any]
    dataset_info: Dict[str, Any]
    performance_metrics: Dict[str, float]
    created_at: datetime
    updated_at: datetime
    model_size_mb: float
    training_duration_seconds: Optional[float] = None
    training_job_id: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = None
    custom_metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.custom_metadata is None:
            self.custom_metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        # Convert datetime objects to strings
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """Create from dictionary."""
        # Parse datetime strings
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        return cls(**data)


@dataclass
class ModelRegistryEntry:
    """Entry in the model registry."""
    
    model_id: str
    name: str
    version: str
    path: str
    metadata: ModelMetadata
    is_active: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'model_id': self.model_id,
            'name': self.name,
            'version': self.version,
            'path': self.path,
            'metadata': self.metadata.to_dict(),
            'is_active': self.is_active
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelRegistryEntry':
        """Create from dictionary."""
        metadata = ModelMetadata.from_dict(data['metadata'])
        return cls(
            model_id=data['model_id'],
            name=data['name'],
            version=data['version'],
            path=data['path'],
            metadata=metadata,
            is_active=data.get('is_active', True)
        )


class ModelManager:
    """
    Manages fine-tuned models with versioning and metadata storage.
    
    Provides functionality for:
    - Saving and loading models with metadata
    - Model versioning and registry management
    - Search and filtering capabilities
    - Model comparison and analytics
    """
    
    def __init__(self, base_model_dir: str = "./models", registry_file: str = "model_registry.json"):
        """
        Initialize the model manager.
        
        Args:
            base_model_dir: Base directory for storing models.
            registry_file: Path to the model registry file.
        """
        self.base_model_dir = Path(base_model_dir)
        self.base_model_dir.mkdir(parents=True, exist_ok=True)
        
        self.registry_file = self.base_model_dir / registry_file
        self.logger = logging.getLogger(__name__)
        
        # Load existing registry
        self.registry: Dict[str, ModelRegistryEntry] = self._load_registry()
        
        if not TRANSFORMERS_AVAILABLE:
            self.logger.warning("Transformers not available. Model manager will use mock implementations.")
    
    def _load_registry(self) -> Dict[str, ModelRegistryEntry]:
        """Load model registry from file."""
        if not self.registry_file.exists():
            return {}
        
        try:
            with open(self.registry_file, 'r') as f:
                registry_data = json.load(f)
            
            registry = {}
            for model_id, entry_data in registry_data.items():
                registry[model_id] = ModelRegistryEntry.from_dict(entry_data)
            
            self.logger.info(f"Loaded {len(registry)} models from registry")
            return registry
            
        except Exception as e:
            self.logger.error(f"Failed to load model registry: {e}")
            return {}
    
    def _save_registry(self) -> None:
        """Save model registry to file."""
        try:
            registry_data = {}
            for model_id, entry in self.registry.items():
                registry_data[model_id] = entry.to_dict()
            
            with open(self.registry_file, 'w') as f:
                json.dump(registry_data, f, indent=2)
            
            self.logger.info(f"Saved registry with {len(self.registry)} models")
            
        except Exception as e:
            self.logger.error(f"Failed to save model registry: {e}")
            raise
    
    def _calculate_model_size(self, model_path: Path) -> float:
        """Calculate model size in MB."""
        total_size = 0
        for file_path in model_path.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size / (1024 * 1024)  # Convert to MB
    
    def _generate_model_id(self, name: str, base_model: str) -> str:
        """Generate unique model ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hash_input = f"{name}_{base_model}_{timestamp}_{uuid.uuid4()}"
        hash_id = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        return f"{name}_{hash_id}"
    
    def _get_next_version(self, name: str) -> str:
        """Get next version number for a model name."""
        existing_versions = []
        for entry in self.registry.values():
            if entry.name == name and entry.is_active:
                try:
                    version_num = int(entry.version.replace('v', ''))
                    existing_versions.append(version_num)
                except ValueError:
                    continue
        
        if not existing_versions:
            return "v1"
        
        return f"v{max(existing_versions) + 1}"
    
    def save_model(
        self,
        model: Any,
        tokenizer: Any,
        name: str,
        training_config: TrainingConfig,
        dataset_info: Dict[str, Any],
        performance_metrics: Dict[str, float],
        training_job_id: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        custom_metadata: Optional[Dict[str, Any]] = None,
        training_duration: Optional[float] = None
    ) -> str:
        """
        Save a fine-tuned model with metadata.
        
        Args:
            model: The fine-tuned model to save.
            tokenizer: The tokenizer associated with the model.
            name: Human-readable name for the model.
            training_config: Configuration used for training.
            dataset_info: Information about the training dataset.
            performance_metrics: Training and evaluation metrics.
            training_job_id: ID of the training job that created this model.
            description: Optional description of the model.
            tags: Optional tags for categorization.
            custom_metadata: Optional custom metadata.
            training_duration: Training duration in seconds.
            
        Returns:
            Model ID of the saved model.
        """
        # Generate model ID and version
        model_id = self._generate_model_id(name, training_config.base_model)
        version = self._get_next_version(name)
        
        # Create model directory
        model_dir = self.base_model_dir / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            if TRANSFORMERS_AVAILABLE and model is not None and tokenizer is not None:
                # Save model and tokenizer
                model.save_pretrained(str(model_dir))
                tokenizer.save_pretrained(str(model_dir))
                
                # Save training configuration
                config_path = model_dir / "training_config.json"
                with open(config_path, 'w') as f:
                    json.dump(training_config.to_dict(), f, indent=2)
            else:
                # Mock save for testing
                (model_dir / "pytorch_model.bin").touch()
                (model_dir / "config.json").touch()
                (model_dir / "tokenizer.json").touch()
                
                # Save training configuration
                config_path = model_dir / "training_config.json"
                with open(config_path, 'w') as f:
                    json.dump(training_config.to_dict(), f, indent=2)
            
            # Calculate model size
            model_size = self._calculate_model_size(model_dir)
            
            # Create metadata
            metadata = ModelMetadata(
                model_id=model_id,
                name=name,
                version=version,
                base_model=training_config.base_model,
                training_config=training_config.to_dict(),
                dataset_info=dataset_info,
                performance_metrics=performance_metrics,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                model_size_mb=model_size,
                training_duration_seconds=training_duration,
                training_job_id=training_job_id,
                description=description,
                tags=tags or [],
                custom_metadata=custom_metadata or {}
            )
            
            # Save metadata
            metadata_path = model_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata.to_dict(), f, indent=2)
            
            # Add to registry
            registry_entry = ModelRegistryEntry(
                model_id=model_id,
                name=name,
                version=version,
                path=str(model_dir),
                metadata=metadata,
                is_active=True
            )
            
            self.registry[model_id] = registry_entry
            self._save_registry()
            
            self.logger.info(f"Saved model {name} v{version} with ID {model_id}")
            return model_id
            
        except Exception as e:
            # Clean up on failure
            if model_dir.exists():
                shutil.rmtree(model_dir)
            raise ModelLoadingError(f"Failed to save model: {e}")
    
    def load_model(self, model_id: str) -> Tuple[Any, Any, ModelMetadata]:
        """
        Load a model by ID.
        
        Args:
            model_id: ID of the model to load.
            
        Returns:
            Tuple of (model, tokenizer, metadata).
            
        Raises:
            ModelLoadingError: If model cannot be loaded.
        """
        if model_id not in self.registry:
            raise ModelLoadingError(f"Model {model_id} not found in registry")
        
        entry = self.registry[model_id]
        model_path = Path(entry.path)
        
        if not model_path.exists():
            raise ModelLoadingError(f"Model path does not exist: {model_path}")
        
        try:
            if TRANSFORMERS_AVAILABLE:
                # Check if it's a PEFT model
                peft_config_path = model_path / "adapter_config.json"
                if peft_config_path.exists():
                    # Load PEFT model
                    peft_config = PeftConfig.from_pretrained(str(model_path))
                    base_model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path)
                    model = PeftModel.from_pretrained(base_model, str(model_path))
                else:
                    # Load regular model
                    model = AutoModelForCausalLM.from_pretrained(str(model_path))
                
                tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            else:
                # Mock load for testing
                model = None
                tokenizer = None
            
            self.logger.info(f"Loaded model {entry.name} v{entry.version}")
            return model, tokenizer, entry.metadata
            
        except Exception as e:
            raise ModelLoadingError(f"Failed to load model {model_id}: {e}")
    
    def list_models(
        self,
        name_filter: Optional[str] = None,
        tag_filter: Optional[List[str]] = None,
        base_model_filter: Optional[str] = None,
        active_only: bool = True
    ) -> List[ModelRegistryEntry]:
        """
        List models with optional filtering.
        
        Args:
            name_filter: Filter by model name (partial match).
            tag_filter: Filter by tags (must have all specified tags).
            base_model_filter: Filter by base model.
            active_only: Only return active models.
            
        Returns:
            List of matching model registry entries.
        """
        results = []
        
        for entry in self.registry.values():
            # Skip inactive models if requested
            if active_only and not entry.is_active:
                continue
            
            # Apply filters
            if name_filter and name_filter.lower() not in entry.name.lower():
                continue
            
            if base_model_filter and entry.metadata.base_model != base_model_filter:
                continue
            
            if tag_filter:
                if not all(tag in entry.metadata.tags for tag in tag_filter):
                    continue
            
            results.append(entry)
        
        # Sort by creation date (newest first)
        results.sort(key=lambda x: x.metadata.created_at, reverse=True)
        return results
    
    def get_model_info(self, model_id: str) -> Optional[ModelRegistryEntry]:
        """
        Get information about a specific model.
        
        Args:
            model_id: ID of the model.
            
        Returns:
            ModelRegistryEntry if found, None otherwise.
        """
        return self.registry.get(model_id)
    
    def delete_model(self, model_id: str, permanent: bool = False) -> bool:
        """
        Delete a model.
        
        Args:
            model_id: ID of the model to delete.
            permanent: If True, permanently delete files. If False, just mark as inactive.
            
        Returns:
            True if deletion was successful.
        """
        if model_id not in self.registry:
            return False
        
        entry = self.registry[model_id]
        
        try:
            if permanent:
                # Delete model files
                model_path = Path(entry.path)
                if model_path.exists():
                    shutil.rmtree(model_path)
                
                # Remove from registry
                del self.registry[model_id]
                self.logger.info(f"Permanently deleted model {model_id}")
            else:
                # Mark as inactive
                entry.is_active = False
                self.logger.info(f"Marked model {model_id} as inactive")
            
            self._save_registry()
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete model {model_id}: {e}")
            return False
    
    def update_model_metadata(
        self,
        model_id: str,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        custom_metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update model metadata.
        
        Args:
            model_id: ID of the model to update.
            description: New description.
            tags: New tags.
            custom_metadata: New custom metadata.
            
        Returns:
            True if update was successful.
        """
        if model_id not in self.registry:
            return False
        
        entry = self.registry[model_id]
        
        # Update metadata
        if description is not None:
            entry.metadata.description = description
        
        if tags is not None:
            entry.metadata.tags = tags
        
        if custom_metadata is not None:
            entry.metadata.custom_metadata.update(custom_metadata)
        
        entry.metadata.updated_at = datetime.now()
        
        # Save updated metadata to file
        try:
            metadata_path = Path(entry.path) / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(entry.metadata.to_dict(), f, indent=2)
            
            self._save_registry()
            self.logger.info(f"Updated metadata for model {model_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update metadata for model {model_id}: {e}")
            return False
    
    def compare_models(self, model_ids: List[str]) -> Dict[str, Any]:
        """
        Compare multiple models.
        
        Args:
            model_ids: List of model IDs to compare.
            
        Returns:
            Dictionary with comparison data.
        """
        if not model_ids:
            return {}
        
        comparison = {
            'models': {},
            'metrics_comparison': {},
            'config_comparison': {},
            'summary': {}
        }
        
        # Collect model data
        models_data = []
        for model_id in model_ids:
            if model_id in self.registry:
                entry = self.registry[model_id]
                models_data.append(entry)
                comparison['models'][model_id] = {
                    'name': entry.name,
                    'version': entry.version,
                    'base_model': entry.metadata.base_model,
                    'created_at': entry.metadata.created_at.isoformat(),
                    'model_size_mb': entry.metadata.model_size_mb,
                    'performance_metrics': entry.metadata.performance_metrics
                }
        
        if not models_data:
            return comparison
        
        # Compare metrics
        all_metrics = set()
        for entry in models_data:
            all_metrics.update(entry.metadata.performance_metrics.keys())
        
        for metric in all_metrics:
            comparison['metrics_comparison'][metric] = {}
            values = []
            for entry in models_data:
                value = entry.metadata.performance_metrics.get(metric)
                comparison['metrics_comparison'][metric][entry.model_id] = value
                if value is not None:
                    values.append(value)
            
            if values:
                comparison['metrics_comparison'][metric]['best'] = min(values) if 'loss' in metric else max(values)
                comparison['metrics_comparison'][metric]['worst'] = max(values) if 'loss' in metric else min(values)
                comparison['metrics_comparison'][metric]['average'] = sum(values) / len(values)
        
        # Summary statistics
        comparison['summary'] = {
            'total_models': len(models_data),
            'base_models': list(set(entry.metadata.base_model for entry in models_data)),
            'size_range_mb': {
                'min': min(entry.metadata.model_size_mb for entry in models_data),
                'max': max(entry.metadata.model_size_mb for entry in models_data),
                'average': sum(entry.metadata.model_size_mb for entry in models_data) / len(models_data)
            }
        }
        
        return comparison
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the model registry.
        
        Returns:
            Dictionary with registry statistics.
        """
        active_models = [entry for entry in self.registry.values() if entry.is_active]
        
        if not active_models:
            return {
                'total_models': 0,
                'active_models': 0,
                'inactive_models': len(self.registry),
                'base_models': [],
                'total_size_mb': 0
            }
        
        base_models = {}
        total_size = 0
        
        for entry in active_models:
            base_model = entry.metadata.base_model
            base_models[base_model] = base_models.get(base_model, 0) + 1
            total_size += entry.metadata.model_size_mb
        
        return {
            'total_models': len(self.registry),
            'active_models': len(active_models),
            'inactive_models': len(self.registry) - len(active_models),
            'base_models': base_models,
            'total_size_mb': total_size,
            'average_size_mb': total_size / len(active_models) if active_models else 0
        }