"""
Centralized caching system for the LLM optimization platform.
"""
import os
import json
import pickle
import hashlib
import logging
from typing import Any, Optional, Dict, Union
from datetime import datetime, timedelta
from threading import Lock
from collections import OrderedDict
from functools import wraps

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    redis = None
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)


class MemoryCache:
    """In-memory LRU cache with TTL support."""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache = OrderedDict()
        self.timestamps = {}
        self.lock = Lock()
    
    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired."""
        if key not in self.timestamps:
            return True
        
        timestamp, ttl = self.timestamps[key]
        return datetime.utcnow() > timestamp + timedelta(seconds=ttl)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self.lock:
            if key in self.cache and not self._is_expired(key):
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                return self.cache[key]
            elif key in self.cache:
                # Remove expired entry
                del self.cache[key]
                del self.timestamps[key]
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache."""
        with self.lock:
            ttl = ttl or self.default_ttl
            
            if key in self.cache:
                # Update existing
                self.cache[key] = value
                self.cache.move_to_end(key)
            else:
                # Add new
                self.cache[key] = value
                
                # Remove oldest if cache is full
                if len(self.cache) >= self.max_size:
                    oldest_key = next(iter(self.cache))
                    del self.cache[oldest_key]
                    del self.timestamps[oldest_key]
            
            self.timestamps[key] = (datetime.utcnow(), ttl)
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                del self.timestamps[key]
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_entries = len(self.cache)
            expired_entries = sum(1 for key in self.cache if self._is_expired(key))
            
            return {
                'total_entries': total_entries,
                'active_entries': total_entries - expired_entries,
                'expired_entries': expired_entries,
                'max_size': self.max_size,
                'hit_rate': getattr(self, '_hit_count', 0) / max(getattr(self, '_access_count', 1), 1)
            }


class RedisCache:
    """Redis-based distributed cache."""
    
    def __init__(self, redis_url: str = None, default_ttl: int = 3600):
        self.redis_url = redis_url or os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        self.default_ttl = default_ttl
        self.client = None
        self._connect()
    
    def _connect(self) -> None:
        """Connect to Redis."""
        if not REDIS_AVAILABLE:
            logger.warning("Redis not available, skipping connection")
            self.client = None
            return
            
        try:
            self.client = redis.from_url(self.redis_url, decode_responses=False)
            self.client.ping()
            logger.info("Connected to Redis cache")
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}")
            self.client = None
    
    def _serialize(self, value: Any) -> bytes:
        """Serialize value for storage."""
        return pickle.dumps(value)
    
    def _deserialize(self, data: bytes) -> Any:
        """Deserialize value from storage."""
        return pickle.loads(data)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if not self.client:
            return None
        
        try:
            data = self.client.get(key)
            if data:
                return self._deserialize(data)
        except Exception as e:
            logger.error(f"Redis get error: {e}")
        
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        if not self.client:
            return False
        
        try:
            ttl = ttl or self.default_ttl
            data = self._serialize(value)
            return self.client.setex(key, ttl, data)
        except Exception as e:
            logger.error(f"Redis set error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        if not self.client:
            return False
        
        try:
            return bool(self.client.delete(key))
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        if not self.client:
            return
        
        try:
            self.client.flushdb()
        except Exception as e:
            logger.error(f"Redis clear error: {e}")


class CacheManager:
    """Unified cache manager with fallback support."""
    
    def __init__(self, use_redis: bool = True, memory_cache_size: int = 1000):
        self.memory_cache = MemoryCache(max_size=memory_cache_size)
        self.redis_cache = RedisCache() if use_redis else None
        
        # Use Redis if available, otherwise fallback to memory
        self.primary_cache = self.redis_cache if (self.redis_cache and self.redis_cache.client) else self.memory_cache
        self.fallback_cache = self.memory_cache if self.primary_cache != self.memory_cache else None
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache with fallback."""
        # Try primary cache first
        value = self.primary_cache.get(key)
        if value is not None:
            return value
        
        # Try fallback cache
        if self.fallback_cache:
            value = self.fallback_cache.get(key)
            if value is not None:
                # Populate primary cache
                self.primary_cache.set(key, value)
                return value
        
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache."""
        self.primary_cache.set(key, value, ttl)
        
        # Also set in fallback if different
        if self.fallback_cache and self.fallback_cache != self.primary_cache:
            self.fallback_cache.set(key, value, ttl)
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        result = self.primary_cache.delete(key)
        
        if self.fallback_cache:
            self.fallback_cache.delete(key)
        
        return result
    
    def clear(self) -> None:
        """Clear all caches."""
        self.primary_cache.clear()
        if self.fallback_cache:
            self.fallback_cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {
            'primary_cache': type(self.primary_cache).__name__,
            'fallback_cache': type(self.fallback_cache).__name__ if self.fallback_cache else None
        }
        
        if hasattr(self.primary_cache, 'get_stats'):
            stats['primary_stats'] = self.primary_cache.get_stats()
        
        if self.fallback_cache and hasattr(self.fallback_cache, 'get_stats'):
            stats['fallback_stats'] = self.fallback_cache.get_stats()
        
        return stats


# Global cache manager instance
cache_manager = CacheManager()


def cache_result(key_prefix: str = "", ttl: int = 3600):
    """
    Decorator for caching function results.
    
    Args:
        key_prefix: Prefix for cache key
        ttl: Time to live in seconds
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            key_parts = [key_prefix, func.__name__]
            
            # Add args to key
            for arg in args:
                if isinstance(arg, (str, int, float, bool)):
                    key_parts.append(str(arg))
                else:
                    key_parts.append(hashlib.md5(str(arg).encode()).hexdigest()[:8])
            
            # Add kwargs to key
            for k, v in sorted(kwargs.items()):
                if isinstance(v, (str, int, float, bool)):
                    key_parts.append(f"{k}:{v}")
                else:
                    key_parts.append(f"{k}:{hashlib.md5(str(v).encode()).hexdigest()[:8]}")
            
            cache_key = ":".join(key_parts)
            
            # Try to get from cache
            result = cache_manager.get(cache_key)
            if result is not None:
                logger.debug(f"Cache hit for {cache_key}")
                return result
            
            # Execute function and cache result
            logger.debug(f"Cache miss for {cache_key}")
            result = func(*args, **kwargs)
            cache_manager.set(cache_key, result, ttl)
            
            return result
        
        return wrapper
    return decorator


def invalidate_cache_pattern(pattern: str) -> None:
    """
    Invalidate cache entries matching a pattern.
    
    Args:
        pattern: Pattern to match cache keys
    """
    # This is a simplified implementation
    # In production, you might want to use Redis SCAN for pattern matching
    logger.info(f"Cache invalidation requested for pattern: {pattern}")