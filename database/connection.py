"""
Database connection management with connection pooling and migration support.
"""

import os
import logging
from typing import Optional, Generator
from contextlib import contextmanager
from sqlalchemy import create_engine, Engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool, QueuePool
from sqlalchemy.exc import SQLAlchemyError

from .models import Base
from utils.exceptions import DatabaseError
from utils.connection_pool import DatabaseConnectionPool

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manages database connections, sessions, and migrations.
    
    Provides connection pooling, session management, and database initialization
    for the LLM optimization platform.
    """
    
    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize database manager.
        
        Args:
            database_url: Database connection URL. If None, uses SQLite default.
        """
        self.database_url = database_url or self._get_default_database_url()
        self.engine: Optional[Engine] = None
        self.session_factory: Optional[sessionmaker] = None
        
    def _get_default_database_url(self) -> str:
        """Get default SQLite database URL."""
        db_path = os.getenv('DATABASE_PATH', 'llm_optimization.db')
        return f"sqlite:///{db_path}"
    
    def initialize(self) -> None:
        """
        Initialize database engine and create tables.
        
        Raises:
            DatabaseError: If database initialization fails.
        """
        try:
            # Create engine with optimized configuration
            if self.database_url.startswith('sqlite'):
                # SQLite specific configuration
                self.engine = create_engine(
                    self.database_url,
                    poolclass=StaticPool,
                    connect_args={
                        'check_same_thread': False,
                        'timeout': 30,
                        'isolation_level': None  # Enable autocommit mode
                    },
                    echo=os.getenv('DATABASE_DEBUG', 'false').lower() == 'true'
                )
            else:
                # PostgreSQL/MySQL configuration with optimized pooling
                self.engine = create_engine(
                    self.database_url,
                    poolclass=QueuePool,
                    pool_size=20,  # Increased pool size
                    max_overflow=30,  # Increased overflow
                    pool_pre_ping=True,
                    pool_recycle=3600,  # Recycle connections every hour
                    echo=os.getenv('DATABASE_DEBUG', 'false').lower() == 'true'
                )
            
            # Create session factory with optimized settings
            self.session_factory = sessionmaker(
                bind=self.engine,
                expire_on_commit=False,  # Keep objects accessible after commit
                autoflush=True,
                autocommit=False
            )
            
            # Create all tables
            Base.metadata.create_all(self.engine)
            
            # Create indexes for performance
            self._create_indexes()
            
            logger.info(f"Database initialized successfully: {self.database_url}")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise DatabaseError(f"Database initialization failed: {e}")
    
    def _create_indexes(self) -> None:
        """Create database indexes for performance optimization."""
        try:
            with self.engine.connect() as conn:
                # Create indexes for common queries
                indexes = [
                    "CREATE INDEX IF NOT EXISTS idx_experiments_created_at ON experiments(created_at)",
                    "CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status)",
                    "CREATE INDEX IF NOT EXISTS idx_evaluations_experiment_id ON evaluations(experiment_id)",
                    "CREATE INDEX IF NOT EXISTS idx_evaluations_model_id ON evaluations(model_id)",
                    "CREATE INDEX IF NOT EXISTS idx_evaluations_created_at ON evaluations(created_at)",
                    "CREATE INDEX IF NOT EXISTS idx_models_type ON models(type)",
                    "CREATE INDEX IF NOT EXISTS idx_models_created_at ON models(created_at)",
                    "CREATE INDEX IF NOT EXISTS idx_feedback_evaluation_id ON feedback(evaluation_id)",
                    "CREATE INDEX IF NOT EXISTS idx_feedback_rating ON feedback(rating)",
                ]
                
                for index_sql in indexes:
                    try:
                        conn.execute(text(index_sql))
                    except Exception as e:
                        logger.warning(f"Failed to create index: {e}")
                
                conn.commit()
                logger.info("Database indexes created successfully")
                
        except Exception as e:
            logger.warning(f"Failed to create indexes: {e}")
    
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """
        Get database session with automatic cleanup.
        
        Yields:
            Session: SQLAlchemy session instance.
            
        Raises:
            DatabaseError: If session creation fails.
        """
        if not self.session_factory:
            raise DatabaseError("Database not initialized. Call initialize() first.")
        
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise DatabaseError(f"Database operation failed: {e}")
        finally:
            session.close()
    
    def health_check(self) -> bool:
        """
        Check database connectivity.
        
        Returns:
            bool: True if database is accessible, False otherwise.
        """
        try:
            with self.get_session() as session:
                session.execute(text("SELECT 1"))
                return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    def migrate_schema(self) -> None:
        """
        Apply database schema migrations.
        
        Raises:
            DatabaseError: If migration fails.
        """
        try:
            if not self.engine:
                raise DatabaseError("Database not initialized")
            
            # Create all tables (handles new tables and columns)
            Base.metadata.create_all(self.engine)
            
            logger.info("Database schema migration completed")
            
        except Exception as e:
            logger.error(f"Schema migration failed: {e}")
            raise DatabaseError(f"Schema migration failed: {e}")
    
    def close(self) -> None:
        """Close database connections."""
        if self.engine:
            self.engine.dispose()
            logger.info("Database connections closed")


# Global database manager instance
db_manager = DatabaseManager()