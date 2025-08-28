"""
Tests for database connection management.
"""

import pytest
import tempfile
import os
from unittest.mock import patch, MagicMock
from sqlalchemy.exc import SQLAlchemyError

from database.connection import DatabaseManager
from database.models import Base
from utils.exceptions import DatabaseError


class TestDatabaseManager:
    """Test database manager functionality."""
    
    def test_default_database_url(self):
        """Test default SQLite database URL generation."""
        db_manager = DatabaseManager()
        assert db_manager.database_url == "sqlite:///llm_optimization.db"
    
    def test_custom_database_url(self):
        """Test custom database URL."""
        custom_url = "sqlite:///test.db"
        db_manager = DatabaseManager(custom_url)
        assert db_manager.database_url == custom_url
    
    def test_environment_database_path(self):
        """Test database path from environment variable."""
        with patch.dict(os.environ, {'DATABASE_PATH': 'custom.db'}):
            db_manager = DatabaseManager()
            assert db_manager.database_url == "sqlite:///custom.db"
    
    def test_initialize_sqlite(self):
        """Test SQLite database initialization."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_url = f"sqlite:///{tmp.name}"
            db_manager = DatabaseManager(db_url)
            
            try:
                db_manager.initialize()
                
                assert db_manager.engine is not None
                assert db_manager.session_factory is not None
                
                # Test that tables were created
                with db_manager.get_session() as session:
                    # Should not raise an error
                    from sqlalchemy import text
                    session.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
                    
            finally:
                db_manager.close()
                os.unlink(tmp.name)
    
    def test_initialize_failure(self):
        """Test database initialization failure."""
        # Use invalid database URL
        db_manager = DatabaseManager("invalid://invalid")
        
        with pytest.raises(DatabaseError, match="Database initialization failed"):
            db_manager.initialize()
    
    def test_session_context_manager(self):
        """Test session context manager."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_url = f"sqlite:///{tmp.name}"
            db_manager = DatabaseManager(db_url)
            
            try:
                db_manager.initialize()
                
                # Test successful session
                with db_manager.get_session() as session:
                    from sqlalchemy import text
                    result = session.execute(text("SELECT 1")).scalar()
                    assert result == 1
                    
            finally:
                db_manager.close()
                os.unlink(tmp.name)
    
    def test_session_without_initialization(self):
        """Test session creation without initialization."""
        db_manager = DatabaseManager()
        
        with pytest.raises(DatabaseError, match="Database not initialized"):
            with db_manager.get_session():
                pass
    
    def test_session_rollback_on_error(self):
        """Test session rollback on error."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_url = f"sqlite:///{tmp.name}"
            db_manager = DatabaseManager(db_url)
            
            try:
                db_manager.initialize()
                
                with pytest.raises(DatabaseError):
                    with db_manager.get_session() as session:
                        # Force an error
                        session.execute("INVALID SQL")
                        
            finally:
                db_manager.close()
                os.unlink(tmp.name)
    
    def test_health_check_success(self):
        """Test successful health check."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_url = f"sqlite:///{tmp.name}"
            db_manager = DatabaseManager(db_url)
            
            try:
                db_manager.initialize()
                assert db_manager.health_check() is True
                
            finally:
                db_manager.close()
                os.unlink(tmp.name)
    
    def test_health_check_failure(self):
        """Test health check failure."""
        db_manager = DatabaseManager()
        assert db_manager.health_check() is False
    
    def test_migrate_schema(self):
        """Test schema migration."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_url = f"sqlite:///{tmp.name}"
            db_manager = DatabaseManager(db_url)
            
            try:
                db_manager.initialize()
                
                # Migration should succeed
                db_manager.migrate_schema()
                
                # Verify tables exist
                with db_manager.get_session() as session:
                    from sqlalchemy import text
                    tables = session.execute(
                        text("SELECT name FROM sqlite_master WHERE type='table'")
                    ).fetchall()
                    table_names = [table[0] for table in tables]
                    
                    expected_tables = ['datasets', 'models', 'experiments', 'evaluations', 'comparison_results']
                    for table in expected_tables:
                        assert table in table_names
                        
            finally:
                db_manager.close()
                os.unlink(tmp.name)
    
    def test_migrate_schema_without_initialization(self):
        """Test schema migration without initialization."""
        db_manager = DatabaseManager()
        
        with pytest.raises(DatabaseError, match="Database not initialized"):
            db_manager.migrate_schema()
    
    def test_close_connections(self):
        """Test closing database connections."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_url = f"sqlite:///{tmp.name}"
            db_manager = DatabaseManager(db_url)
            
            try:
                db_manager.initialize()
                assert db_manager.engine is not None
                
                db_manager.close()
                # Engine should be disposed but still accessible
                assert db_manager.engine is not None
                
            finally:
                os.unlink(tmp.name)
    
    @patch.dict(os.environ, {'DATABASE_DEBUG': 'true'})
    def test_debug_mode(self):
        """Test database debug mode."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_url = f"sqlite:///{tmp.name}"
            db_manager = DatabaseManager(db_url)
            
            try:
                db_manager.initialize()
                # In debug mode, echo should be True
                assert db_manager.engine.echo is True
                
            finally:
                db_manager.close()
                os.unlink(tmp.name)


@pytest.fixture
def temp_db():
    """Fixture for temporary database."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_url = f"sqlite:///{tmp.name}"
        db_manager = DatabaseManager(db_url)
        db_manager.initialize()
        
        yield db_manager
        
        db_manager.close()
        os.unlink(tmp.name)


class TestDatabaseIntegration:
    """Integration tests for database operations."""
    
    def test_full_database_lifecycle(self, temp_db):
        """Test complete database lifecycle."""
        db_manager = temp_db
        
        # Test health check
        assert db_manager.health_check() is True
        
        # Test session operations
        with db_manager.get_session() as session:
            # Test basic query
            from sqlalchemy import text
            result = session.execute(text("SELECT COUNT(*) FROM datasets")).scalar()
            assert result == 0
            
        # Test migration
        db_manager.migrate_schema()
        
        # Verify migration worked
        with db_manager.get_session() as session:
            from sqlalchemy import text
            result = session.execute(text("SELECT COUNT(*) FROM datasets")).scalar()
            assert result == 0