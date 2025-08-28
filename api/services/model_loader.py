"""
Model loading and caching service.
"""
import os
import json
import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import threading
from collections import OrderedDict

from utils.cache_manager import cache_manager, cache_result
from utils.async_processor import async_task, task_manager

logger = logging.getLogger(__name__)


class ModelCache:
    """LRU cache for loaded models."""
    
    def __init__(self, max_size: int = 3):
        self.max_size = max_size
        self.cache = OrderedDict()
        self.lock = threading.Lock()
    
    def get(self, model_id: str) -> Optional[Any]:
        """Get model from cache."""
        with self.lock:
            if model_id in self.cache:
                # Move to end (most recently used)
                self.cache.move_to_end(model_id)
                return self.cache[model_id]
            return None
    
    def put(self, model_id: str, model: Any) -> None:
        """Put model in cache."""
        with self.lock:
            if model_id in self.cache:
                # Update existing
                self.cache[model_id] = model
                self.cache.move_to_end(model_id)
            else:
                # Add new
                self.cache[model_id] = model
                
                # Remove oldest if cache is full
                if len(self.cache) > self.max_size:
                    oldest_key = next(iter(self.cache))
                    removed_model = self.cache.pop(oldest_key)
                    logger.info(f"Evicted model {oldest_key} from cache")
                    
                    # Clean up model if it has a cleanup method
                    if hasattr(removed_model, 'cleanup'):
                        try:
                            removed_model.cleanup()
                        except Exception as e:
                            logger.warning(f"Error cleaning up model {oldest_key}: {e}")
    
    def clear(self) -> None:
        """Clear all models from cache."""
        with self.lock:
            for model_id, model in self.cache.items():
                if hasattr(model, 'cleanup'):
                    try:
                        model.cleanup()
                    except Exception as e:
                        logger.warning(f"Error cleaning up model {model_id}: {e}")
            self.cache.clear()
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'models': list(self.cache.keys())
            }


class ModelLoader:
    """Service for loading and managing fine-tuned models."""
    
    def __init__(self, cache_size: int = 3):
        self.cache = ModelCache(cache_size)
        self.models_dir = os.path.join(os.getcwd(), 'models')
        self.lock = threading.Lock()
    
    @cache_result(key_prefix="model_load", ttl=3600)
    def load_model(self, model_id: str) -> Tuple[Any, Any]:
        """
        Load a fine-tuned model and tokenizer.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Tuple of (model, tokenizer)
            
        Raises:
            ValueError: If model not found or invalid
            RuntimeError: If model loading fails
        """
        # Check local cache first
        cached_model = self.cache.get(model_id)
        if cached_model:
            logger.info(f"Loaded model {model_id} from local cache")
            return cached_model['model'], cached_model['tokenizer']
        
        with self.lock:
            # Double-check cache after acquiring lock
            cached_model = self.cache.get(model_id)
            if cached_model:
                return cached_model['model'], cached_model['tokenizer']
            
            # Load model from disk
            model_path = os.path.join(self.models_dir, model_id)
            
            if not os.path.exists(model_path):
                raise ValueError(f"Model {model_id} not found at {model_path}")
            
            # Check if it's a valid model directory
            config_path = os.path.join(model_path, 'config.json')
            if not os.path.exists(config_path):
                raise ValueError(f"Invalid model directory: missing config.json")
            
            try:
                # Import transformers here to avoid import errors if not installed
                from transformers import AutoModelForCausalLM, AutoTokenizer
                
                logger.info(f"Loading model {model_id} from {model_path}")
                
                # Load tokenizer
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                
                # Load model
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype='auto',
                    device_map='auto' if self._has_gpu() else None,
                    low_cpu_mem_usage=True
                )
                
                # Cache the loaded model
                model_data = {
                    'model': model,
                    'tokenizer': tokenizer,
                    'loaded_at': datetime.utcnow(),
                    'path': model_path
                }
                
                self.cache.put(model_id, model_data)
                
                logger.info(f"Successfully loaded model {model_id}")
                return model, tokenizer
                
            except ImportError as e:
                raise RuntimeError(f"Transformers library not available: {e}")
            except Exception as e:
                raise RuntimeError(f"Failed to load model {model_id}: {e}")
    
    def load_model_async(self, model_id: str) -> str:
        """
        Load a model asynchronously.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Task ID for tracking progress
        """
        return task_manager.submit_task(
            self.load_model,
            model_id,
            task_id=f"load_model_{model_id}",
            metadata={'model_id': model_id, 'operation': 'load_model'}
        )
    
    def unload_model(self, model_id: str) -> bool:
        """
        Unload a model from cache.
        
        Args:
            model_id: Model identifier
            
        Returns:
            True if model was unloaded, False if not in cache
        """
        with self.lock:
            cached_model = self.cache.get(model_id)
            if cached_model:
                # Remove from cache
                del self.cache.cache[model_id]
                
                # Clean up if possible
                if hasattr(cached_model['model'], 'cleanup'):
                    try:
                        cached_model['model'].cleanup()
                    except Exception as e:
                        logger.warning(f"Error cleaning up model {model_id}: {e}")
                
                logger.info(f"Unloaded model {model_id}")
                return True
            
            return False
    
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a loaded model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Model information dictionary or None if not loaded
        """
        cached_model = self.cache.get(model_id)
        if cached_model:
            return {
                'model_id': model_id,
                'loaded_at': cached_model['loaded_at'].isoformat(),
                'path': cached_model['path'],
                'in_cache': True
            }
        return None
    
    def list_available_models(self) -> list:
        """
        List all available fine-tuned models.
        
        Returns:
            List of model identifiers
        """
        if not os.path.exists(self.models_dir):
            return []
        
        models = []
        for item in os.listdir(self.models_dir):
            model_path = os.path.join(self.models_dir, item)
            if os.path.isdir(model_path):
                config_path = os.path.join(model_path, 'config.json')
                if os.path.exists(config_path):
                    models.append(item)
        
        return models
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.get_cache_info()
    
    def clear_cache(self) -> None:
        """Clear all models from cache."""
        self.cache.clear()
        logger.info("Cleared model cache")
    
    def _has_gpu(self) -> bool:
        """Check if GPU is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False