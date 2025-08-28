"""
Connection pooling and resource management utilities.
"""
import logging
import threading
import time
from typing import Any, Dict, Optional, Callable, ContextManager
from contextlib import contextmanager
from queue import Queue, Empty, Full
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class PooledConnection:
    """Wrapper for pooled connections."""
    connection: Any
    created_at: datetime
    last_used: datetime
    use_count: int = 0
    is_healthy: bool = True


class ConnectionPool:
    """Generic connection pool implementation."""
    
    def __init__(self, 
                 create_connection: Callable[[], Any],
                 close_connection: Callable[[Any], None],
                 validate_connection: Optional[Callable[[Any], bool]] = None,
                 max_size: int = 10,
                 min_size: int = 2,
                 max_idle_time: int = 300,
                 max_lifetime: int = 3600):
        """
        Initialize connection pool.
        
        Args:
            create_connection: Function to create new connections
            close_connection: Function to close connections
            validate_connection: Function to validate connection health
            max_size: Maximum pool size
            min_size: Minimum pool size
            max_idle_time: Maximum idle time before connection is closed
            max_lifetime: Maximum connection lifetime
        """
        self.create_connection = create_connection
        self.close_connection = close_connection
        self.validate_connection = validate_connection or (lambda x: True)
        self.max_size = max_size
        self.min_size = min_size
        self.max_idle_time = max_idle_time
        self.max_lifetime = max_lifetime
        
        self.pool = Queue(maxsize=max_size)
        self.active_connections = 0
        self.total_connections = 0
        self.lock = threading.Lock()
        
        # Statistics
        self.stats = {
            'created': 0,
            'destroyed': 0,
            'borrowed': 0,
            'returned': 0,
            'validation_failures': 0
        }
        
        # Initialize minimum connections
        self._initialize_pool()
        
        # Start maintenance thread
        self.maintenance_thread = threading.Thread(target=self._maintenance_worker, daemon=True)
        self.maintenance_thread.start()
    
    def _initialize_pool(self) -> None:
        """Initialize pool with minimum connections."""
        for _ in range(self.min_size):
            try:
                conn = self._create_pooled_connection()
                self.pool.put_nowait(conn)
            except Exception as e:
                logger.error(f"Failed to initialize connection: {e}")
    
    def _create_pooled_connection(self) -> PooledConnection:
        """Create a new pooled connection."""
        try:
            connection = self.create_connection()
            pooled_conn = PooledConnection(
                connection=connection,
                created_at=datetime.utcnow(),
                last_used=datetime.utcnow()
            )
            
            with self.lock:
                self.total_connections += 1
                self.stats['created'] += 1
            
            logger.debug("Created new pooled connection")
            return pooled_conn
            
        except Exception as e:
            logger.error(f"Failed to create connection: {e}")
            raise
    
    def _destroy_connection(self, pooled_conn: PooledConnection) -> None:
        """Destroy a pooled connection."""
        try:
            self.close_connection(pooled_conn.connection)
            
            with self.lock:
                self.total_connections -= 1
                self.stats['destroyed'] += 1
            
            logger.debug("Destroyed pooled connection")
            
        except Exception as e:
            logger.error(f"Error destroying connection: {e}")
    
    def _is_connection_stale(self, pooled_conn: PooledConnection) -> bool:
        """Check if connection is stale and should be replaced."""
        now = datetime.utcnow()
        
        # Check lifetime
        if now - pooled_conn.created_at > timedelta(seconds=self.max_lifetime):
            return True
        
        # Check idle time
        if now - pooled_conn.last_used > timedelta(seconds=self.max_idle_time):
            return True
        
        return False
    
    @contextmanager
    def get_connection(self, timeout: float = 30.0) -> ContextManager[Any]:
        """
        Get a connection from the pool.
        
        Args:
            timeout: Timeout for getting connection
            
        Yields:
            Connection object
        """
        pooled_conn = None
        start_time = time.time()
        
        try:
            # Try to get connection from pool
            while time.time() - start_time < timeout:
                try:
                    pooled_conn = self.pool.get(timeout=1.0)
                    break
                except Empty:
                    # Try to create new connection if under limit
                    with self.lock:
                        if self.total_connections < self.max_size:
                            try:
                                pooled_conn = self._create_pooled_connection()
                                break
                            except Exception as e:
                                logger.error(f"Failed to create connection: {e}")
                                continue
            
            if pooled_conn is None:
                raise TimeoutError("Timeout waiting for connection")
            
            # Validate connection
            if not self.validate_connection(pooled_conn.connection) or self._is_connection_stale(pooled_conn):
                logger.debug("Connection validation failed or stale, creating new one")
                self._destroy_connection(pooled_conn)
                pooled_conn = self._create_pooled_connection()
                
                with self.lock:
                    self.stats['validation_failures'] += 1
            
            # Update usage stats
            pooled_conn.last_used = datetime.utcnow()
            pooled_conn.use_count += 1
            
            with self.lock:
                self.active_connections += 1
                self.stats['borrowed'] += 1
            
            yield pooled_conn.connection
            
        finally:
            if pooled_conn:
                # Return connection to pool
                try:
                    self.pool.put_nowait(pooled_conn)
                    
                    with self.lock:
                        self.active_connections -= 1
                        self.stats['returned'] += 1
                        
                except Full:
                    # Pool is full, destroy connection
                    self._destroy_connection(pooled_conn)
                    
                    with self.lock:
                        self.active_connections -= 1
    
    def _maintenance_worker(self) -> None:
        """Background maintenance worker."""
        while True:
            try:
                time.sleep(60)  # Run every minute
                
                # Get all connections from pool for maintenance
                connections_to_check = []
                while not self.pool.empty():
                    try:
                        conn = self.pool.get_nowait()
                        connections_to_check.append(conn)
                    except Empty:
                        break
                
                # Check each connection
                valid_connections = []
                for pooled_conn in connections_to_check:
                    if (self.validate_connection(pooled_conn.connection) and 
                        not self._is_connection_stale(pooled_conn)):
                        valid_connections.append(pooled_conn)
                    else:
                        self._destroy_connection(pooled_conn)
                
                # Return valid connections to pool
                for conn in valid_connections:
                    try:
                        self.pool.put_nowait(conn)
                    except Full:
                        self._destroy_connection(conn)
                
                # Ensure minimum connections
                current_size = self.pool.qsize()
                if current_size < self.min_size:
                    for _ in range(self.min_size - current_size):
                        try:
                            conn = self._create_pooled_connection()
                            self.pool.put_nowait(conn)
                        except Exception as e:
                            logger.error(f"Failed to create maintenance connection: {e}")
                            break
                
            except Exception as e:
                logger.error(f"Error in connection pool maintenance: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        with self.lock:
            return {
                'total_connections': self.total_connections,
                'active_connections': self.active_connections,
                'pool_size': self.pool.qsize(),
                'max_size': self.max_size,
                'min_size': self.min_size,
                'stats': self.stats.copy()
            }
    
    def close(self) -> None:
        """Close all connections in the pool."""
        logger.info("Closing connection pool")
        
        # Close all pooled connections
        while not self.pool.empty():
            try:
                pooled_conn = self.pool.get_nowait()
                self._destroy_connection(pooled_conn)
            except Empty:
                break


class HTTPConnectionPool(ConnectionPool):
    """HTTP connection pool using requests.Session."""
    
    def __init__(self, **kwargs):
        import requests
        
        def create_session():
            session = requests.Session()
            # Configure session defaults
            session.headers.update({
                'User-Agent': 'LLM-Optimization-Platform/1.0'
            })
            return session
        
        def close_session(session):
            session.close()
        
        def validate_session(session):
            # Simple validation - check if session is still usable
            return hasattr(session, 'get')
        
        super().__init__(
            create_connection=create_session,
            close_connection=close_session,
            validate_connection=validate_session,
            **kwargs
        )


class DatabaseConnectionPool(ConnectionPool):
    """Database connection pool."""
    
    def __init__(self, database_url: str, **kwargs):
        self.database_url = database_url
        
        def create_db_connection():
            from sqlalchemy import create_engine
            from sqlalchemy.orm import sessionmaker
            
            engine = create_engine(database_url)
            Session = sessionmaker(bind=engine)
            return Session()
        
        def close_db_connection(session):
            session.close()
        
        def validate_db_connection(session):
            try:
                session.execute("SELECT 1")
                return True
            except Exception:
                return False
        
        super().__init__(
            create_connection=create_db_connection,
            close_connection=close_db_connection,
            validate_connection=validate_db_connection,
            **kwargs
        )


# Global connection pools
http_pool = HTTPConnectionPool(max_size=20, min_size=5)