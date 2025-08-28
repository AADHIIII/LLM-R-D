"""
Tests for comprehensive logging system functionality and performance.
"""

import json
import logging
import os
import tempfile
import time
import unittest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import sqlite3

from utils.logging import (
    JSONFormatter, StructuredLogger, setup_logging, get_structured_logger,
    log_function_call, log_error_with_context
)
from utils.log_aggregator import LogAggregator, LogSearchQuery, LogWatcher
from api.middleware.logging_middleware import RequestLogger


class TestJSONFormatter(unittest.TestCase):
    """Test JSON formatter for structured logging."""
    
    def setUp(self):
        self.formatter = JSONFormatter()
    
    def test_basic_log_formatting(self):
        """Test basic log record formatting to JSON."""
        record = logging.LogRecord(
            name='test_logger',
            level=logging.INFO,
            pathname='/test/path.py',
            lineno=42,
            msg='Test message',
            args=(),
            exc_info=None
        )
        
        formatted = self.formatter.format(record)
        log_data = json.loads(formatted)
        
        self.assertEqual(log_data['level'], 'INFO')
        self.assertEqual(log_data['logger'], 'test_logger')
        self.assertEqual(log_data['message'], 'Test message')
        self.assertEqual(log_data['line'], 42)
        self.assertIn('timestamp', log_data)
    
    def test_exception_formatting(self):
        """Test exception information in JSON format."""
        try:
            raise ValueError("Test exception")
        except ValueError:
            import sys
            exc_info = sys.exc_info()
            record = logging.LogRecord(
                name='test_logger',
                level=logging.ERROR,
                pathname='/test/path.py',
                lineno=42,
                msg='Error occurred',
                args=(),
                exc_info=exc_info
            )
        
        formatted = self.formatter.format(record)
        log_data = json.loads(formatted)
        
        self.assertIn('exception', log_data)
        self.assertEqual(log_data['exception']['type'], 'ValueError')
        self.assertEqual(log_data['exception']['message'], 'Test exception')
        self.assertIsInstance(log_data['exception']['traceback'], list)
    
    def test_extra_fields(self):
        """Test extra fields in log record."""
        record = logging.LogRecord(
            name='test_logger',
            level=logging.INFO,
            pathname='/test/path.py',
            lineno=42,
            msg='Test message',
            args=(),
            exc_info=None
        )
        
        # Add extra fields
        record.request_id = 'req-123'
        record.user_id = 'user-456'
        
        formatted = self.formatter.format(record)
        log_data = json.loads(formatted)
        
        self.assertEqual(log_data['request_id'], 'req-123')
        self.assertEqual(log_data['user_id'], 'user-456')


class TestStructuredLogger(unittest.TestCase):
    """Test structured logger functionality."""
    
    def setUp(self):
        self.logger = StructuredLogger('test_logger')
    
    def test_context_management(self):
        """Test context setting and clearing."""
        self.logger.set_context(request_id='req-123', user_id='user-456')
        self.assertEqual(self.logger.context['request_id'], 'req-123')
        self.assertEqual(self.logger.context['user_id'], 'user-456')
        
        self.logger.clear_context()
        self.assertEqual(len(self.logger.context), 0)
    
    @patch('utils.logging.get_logger')
    def test_logging_with_context(self, mock_get_logger):
        """Test logging with context fields."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        logger = StructuredLogger('test')
        logger.set_context(request_id='req-123')
        logger.info('Test message', extra_field='value')
        
        mock_logger.log.assert_called_once()
        args, kwargs = mock_logger.log.call_args
        
        self.assertEqual(args[0], logging.INFO)
        self.assertEqual(args[1], 'Test message')
        self.assertEqual(kwargs['extra']['request_id'], 'req-123')
        self.assertEqual(kwargs['extra']['extra_field'], 'value')


class TestLogFunctionDecorator(unittest.TestCase):
    """Test function call logging decorator."""
    
    @patch('utils.logging.get_logger')
    def test_successful_function_call(self, mock_get_logger):
        """Test logging of successful function calls."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        @log_function_call
        def test_function(x, y=10):
            return x + y
        
        result = test_function(5, y=15)
        
        self.assertEqual(result, 20)
        self.assertEqual(mock_logger.debug.call_count, 2)  # Entry and exit
    
    @patch('utils.logging.get_logger')
    def test_function_call_with_exception(self, mock_get_logger):
        """Test logging of function calls that raise exceptions."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        @log_function_call
        def failing_function():
            raise ValueError("Test error")
        
        with self.assertRaises(ValueError):
            failing_function()
        
        # Should log entry and error
        self.assertTrue(mock_logger.debug.called)
        self.assertTrue(mock_logger.error.called)


class TestLogAggregator(unittest.TestCase):
    """Test log aggregation and search functionality."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, 'test_logs.db')
        self.aggregator = LogAggregator(self.db_path)
    
    def tearDown(self):
        # Clean up temporary files
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_database_initialization(self):
        """Test database schema creation."""
        self.assertTrue(os.path.exists(self.db_path))
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='log_entries'"
            )
            self.assertIsNotNone(cursor.fetchone())
    
    def test_log_file_ingestion(self):
        """Test ingesting logs from JSON log file."""
        # Create test log file
        log_file = os.path.join(self.temp_dir, 'test.log')
        
        test_logs = [
            {
                'timestamp': '2024-01-01T10:00:00Z',
                'level': 'INFO',
                'logger': 'test.module',
                'message': 'Test message 1',
                'module': 'test_module',
                'function': 'test_func',
                'line': 10,
                'thread': 123,
                'process': 456,
                'request_id': 'req-001'
            },
            {
                'timestamp': '2024-01-01T10:01:00Z',
                'level': 'ERROR',
                'logger': 'test.module',
                'message': 'Test error',
                'module': 'test_module',
                'function': 'error_func',
                'line': 20,
                'thread': 123,
                'process': 456,
                'exception': {
                    'type': 'ValueError',
                    'message': 'Test exception',
                    'traceback': ['line1', 'line2']
                }
            }
        ]
        
        with open(log_file, 'w') as f:
            for log_entry in test_logs:
                f.write(json.dumps(log_entry) + '\n')
        
        # Ingest logs
        count = self.aggregator.ingest_log_file(log_file)
        self.assertEqual(count, 2)
        
        # Verify logs were stored
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM log_entries")
            self.assertEqual(cursor.fetchone()[0], 2)
    
    def test_log_search(self):
        """Test log search functionality."""
        # First ingest some test data
        log_file = os.path.join(self.temp_dir, 'search_test.log')
        
        test_logs = [
            {
                'timestamp': '2024-01-01T10:00:00Z',
                'level': 'INFO',
                'logger': 'api.requests',
                'message': 'Request started',
                'request_id': 'req-001'
            },
            {
                'timestamp': '2024-01-01T10:01:00Z',
                'level': 'ERROR',
                'logger': 'api.errors',
                'message': 'Database connection failed',
                'request_id': 'req-002'
            },
            {
                'timestamp': '2024-01-01T10:02:00Z',
                'level': 'INFO',
                'logger': 'api.requests',
                'message': 'Request completed',
                'request_id': 'req-001'
            }
        ]
        
        with open(log_file, 'w') as f:
            for log_entry in test_logs:
                f.write(json.dumps(log_entry) + '\n')
        
        self.aggregator.ingest_log_file(log_file)
        
        # Test search by level
        query = LogSearchQuery(level='ERROR')
        results = self.aggregator.search_logs(query)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['message'], 'Database connection failed')
        
        # Test search by logger
        query = LogSearchQuery(logger='api.requests')
        results = self.aggregator.search_logs(query)
        self.assertEqual(len(results), 2)
        
        # Test search by request ID
        query = LogSearchQuery(request_id='req-001')
        results = self.aggregator.search_logs(query)
        self.assertEqual(len(results), 2)
        
        # Test search by message pattern
        query = LogSearchQuery(message_pattern='Request')
        results = self.aggregator.search_logs(query)
        self.assertEqual(len(results), 2)
    
    def test_log_statistics(self):
        """Test log statistics generation."""
        # Ingest test data with different levels
        log_file = os.path.join(self.temp_dir, 'stats_test.log')
        
        now = datetime.utcnow()
        test_logs = []
        
        # Create logs with different levels
        for i in range(10):
            test_logs.append({
                'timestamp': (now - timedelta(minutes=i)).isoformat() + 'Z',
                'level': 'INFO',
                'logger': 'test.module',
                'message': f'Info message {i}'
            })
        
        for i in range(3):
            test_logs.append({
                'timestamp': (now - timedelta(minutes=i)).isoformat() + 'Z',
                'level': 'ERROR',
                'logger': 'test.module',
                'message': f'Error message {i}'
            })
        
        with open(log_file, 'w') as f:
            for log_entry in test_logs:
                f.write(json.dumps(log_entry) + '\n')
        
        self.aggregator.ingest_log_file(log_file)
        
        # Get statistics
        stats = self.aggregator.get_log_statistics(hours=1)
        
        self.assertEqual(stats['total_entries'], 13)
        self.assertEqual(stats['error_count'], 3)
        self.assertAlmostEqual(stats['error_rate_percent'], 23.08, places=1)
        self.assertEqual(stats['level_distribution']['INFO'], 10)
        self.assertEqual(stats['level_distribution']['ERROR'], 3)


class TestRequestLogger(unittest.TestCase):
    """Test request logging middleware."""
    
    def setUp(self):
        from flask import Flask
        self.app = Flask(__name__)
        
        with self.app.app_context():
            self.request_logger = RequestLogger(self.app)
    
    def test_request_sanitization(self):
        """Test sensitive data sanitization in requests."""
        with self.app.app_context():
            sensitive_data = {
                'username': 'testuser',
                'password': 'secret123',
                'api_key': 'key123',
                'normal_field': 'normal_value',
                'nested': {
                    'token': 'secret_token',
                    'safe_field': 'safe_value'
                }
            }
            
            sanitized = self.request_logger._sanitize_request_data(sensitive_data)
            
            self.assertEqual(sanitized['username'], 'testuser')
            self.assertEqual(sanitized['password'], '[REDACTED]')
            self.assertEqual(sanitized['api_key'], '[REDACTED]')
            self.assertEqual(sanitized['normal_field'], 'normal_value')
            self.assertEqual(sanitized['nested']['token'], '[REDACTED]')
            self.assertEqual(sanitized['nested']['safe_field'], 'safe_value')


class TestLogWatcher(unittest.TestCase):
    """Test log file watching functionality."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, 'watcher_test.db')
        self.aggregator = LogAggregator(self.db_path)
        self.watcher = LogWatcher(self.aggregator)
    
    def tearDown(self):
        # Clean up temporary files
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_file_watching(self):
        """Test watching log files for new entries."""
        log_file = os.path.join(self.temp_dir, 'watched.log')
        
        # Create initial log file
        with open(log_file, 'w') as f:
            f.write(json.dumps({
                'timestamp': '2024-01-01T10:00:00Z',
                'level': 'INFO',
                'logger': 'test',
                'message': 'Initial message'
            }) + '\n')
        
        # Start watching
        self.watcher.watch_file(log_file)
        self.assertIn(log_file, self.watcher.watched_files)
        
        # Add new entry
        with open(log_file, 'a') as f:
            f.write(json.dumps({
                'timestamp': '2024-01-01T10:01:00Z',
                'level': 'INFO',
                'logger': 'test',
                'message': 'New message'
            }) + '\n')
        
        # Check for updates
        self.watcher.check_for_updates()
        
        # Verify new entry was ingested
        query = LogSearchQuery(message_pattern='New message')
        results = self.aggregator.search_logs(query)
        self.assertEqual(len(results), 1)


class TestLoggingPerformance(unittest.TestCase):
    """Test logging system performance."""
    
    def test_json_formatter_performance(self):
        """Test JSON formatter performance with large number of records."""
        formatter = JSONFormatter()
        
        start_time = time.time()
        
        for i in range(1000):
            record = logging.LogRecord(
                name='perf_test',
                level=logging.INFO,
                pathname='/test/path.py',
                lineno=i,
                msg=f'Performance test message {i}',
                args=(),
                exc_info=None
            )
            record.request_id = f'req-{i}'
            formatted = formatter.format(record)
            
            # Verify it's valid JSON
            json.loads(formatted)
        
        duration = time.time() - start_time
        
        # Should format 1000 records in less than 1 second
        self.assertLess(duration, 1.0)
    
    def test_log_aggregation_performance(self):
        """Test log aggregation performance with large datasets."""
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, 'perf_test.db')
        
        try:
            aggregator = LogAggregator(db_path)
            
            # Create large log file
            log_file = os.path.join(temp_dir, 'large.log')
            
            start_time = time.time()
            
            with open(log_file, 'w') as f:
                for i in range(5000):
                    log_entry = {
                        'timestamp': datetime.utcnow().isoformat() + 'Z',
                        'level': 'INFO',
                        'logger': 'perf.test',
                        'message': f'Performance test message {i}',
                        'request_id': f'req-{i % 100}'  # Simulate some duplicate request IDs
                    }
                    f.write(json.dumps(log_entry) + '\n')
            
            # Ingest logs
            ingestion_start = time.time()
            count = aggregator.ingest_log_file(log_file)
            ingestion_duration = time.time() - ingestion_start
            
            self.assertEqual(count, 5000)
            
            # Should ingest 5000 records in less than 5 seconds
            self.assertLess(ingestion_duration, 5.0)
            
            # Test search performance
            search_start = time.time()
            query = LogSearchQuery(logger='perf.test', limit=100)
            results = aggregator.search_logs(query)
            search_duration = time.time() - search_start
            
            self.assertEqual(len(results), 100)
            
            # Should search through 5000 records in less than 0.1 seconds
            self.assertLess(search_duration, 0.1)
            
        finally:
            # Clean up
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)


if __name__ == '__main__':
    unittest.main()