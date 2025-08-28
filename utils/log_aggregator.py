"""
Log aggregation and search utilities for centralized log management.
"""

import json
import os
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import re
import gzip
from pathlib import Path

from utils.logging import get_structured_logger


@dataclass
class LogEntry:
    """Structured log entry for aggregation."""
    timestamp: datetime
    level: str
    logger: str
    message: str
    module: str
    function: str
    line: int
    thread: int
    process: int
    extra_fields: Dict[str, Any]
    exception: Optional[Dict[str, Any]] = None


@dataclass
class LogSearchQuery:
    """Search query parameters for log aggregation."""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    level: Optional[str] = None
    logger: Optional[str] = None
    message_pattern: Optional[str] = None
    request_id: Optional[str] = None
    limit: int = 1000
    offset: int = 0


class LogAggregator:
    """Centralized log aggregation and search system."""
    
    def __init__(self, db_path: str = "logs/log_aggregator.db"):
        self.db_path = db_path
        self.logger = get_structured_logger('log_aggregator')
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize SQLite database for log storage."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS log_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    level TEXT NOT NULL,
                    logger TEXT NOT NULL,
                    message TEXT NOT NULL,
                    module TEXT,
                    function TEXT,
                    line INTEGER,
                    thread INTEGER,
                    process INTEGER,
                    request_id TEXT,
                    exception_type TEXT,
                    exception_message TEXT,
                    exception_traceback TEXT,
                    extra_fields TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for better search performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON log_entries(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_level ON log_entries(level)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_logger ON log_entries(logger)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_request_id ON log_entries(request_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON log_entries(created_at)")
    
    def ingest_log_file(self, log_file_path: str) -> int:
        """
        Ingest logs from a JSON log file.
        
        Args:
            log_file_path: Path to the log file
            
        Returns:
            Number of log entries ingested
        """
        ingested_count = 0
        
        try:
            # Handle gzipped files
            if log_file_path.endswith('.gz'):
                file_opener = gzip.open
            else:
                file_opener = open
            
            with file_opener(log_file_path, 'rt', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        log_data = json.loads(line)
                        entry = self._parse_log_entry(log_data)
                        self._store_log_entry(entry)
                        ingested_count += 1
                        
                    except json.JSONDecodeError as e:
                        self.logger.warning(
                            f"Failed to parse JSON on line {line_num}",
                            file_path=log_file_path,
                            line_number=line_num,
                            error=str(e)
                        )
                    except Exception as e:
                        self.logger.error(
                            f"Error processing log entry on line {line_num}",
                            file_path=log_file_path,
                            line_number=line_num,
                            error=str(e)
                        )
        
        except Exception as e:
            self.logger.error(
                f"Failed to ingest log file",
                file_path=log_file_path,
                error=str(e)
            )
            raise
        
        self.logger.info(
            f"Ingested {ingested_count} log entries",
            file_path=log_file_path,
            entries_count=ingested_count
        )
        
        return ingested_count
    
    def _parse_log_entry(self, log_data: Dict[str, Any]) -> LogEntry:
        """Parse log data into LogEntry object."""
        timestamp_str = log_data.get('timestamp', '')
        try:
            # Handle different timestamp formats
            if timestamp_str.endswith('Z'):
                timestamp = datetime.fromisoformat(timestamp_str[:-1])
            else:
                timestamp = datetime.fromisoformat(timestamp_str)
        except ValueError:
            timestamp = datetime.utcnow()
        
        # Extract exception information
        exception = None
        if 'exception' in log_data:
            exception = log_data['exception']
        
        # Extract extra fields (everything not in standard fields)
        standard_fields = {
            'timestamp', 'level', 'logger', 'message', 'module',
            'function', 'line', 'thread', 'process', 'exception'
        }
        extra_fields = {
            k: v for k, v in log_data.items()
            if k not in standard_fields
        }
        
        return LogEntry(
            timestamp=timestamp,
            level=log_data.get('level', 'INFO'),
            logger=log_data.get('logger', 'unknown'),
            message=log_data.get('message', ''),
            module=log_data.get('module', ''),
            function=log_data.get('function', ''),
            line=log_data.get('line', 0),
            thread=log_data.get('thread', 0),
            process=log_data.get('process', 0),
            extra_fields=extra_fields,
            exception=exception
        )
    
    def _store_log_entry(self, entry: LogEntry) -> None:
        """Store log entry in database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO log_entries (
                    timestamp, level, logger, message, module, function,
                    line, thread, process, request_id, exception_type,
                    exception_message, exception_traceback, extra_fields
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.timestamp.isoformat(),
                entry.level,
                entry.logger,
                entry.message,
                entry.module,
                entry.function,
                entry.line,
                entry.thread,
                entry.process,
                entry.extra_fields.get('request_id'),
                entry.exception.get('type') if entry.exception else None,
                entry.exception.get('message') if entry.exception else None,
                json.dumps(entry.exception.get('traceback')) if entry.exception else None,
                json.dumps(entry.extra_fields)
            ))
    
    def search_logs(self, query: LogSearchQuery) -> List[Dict[str, Any]]:
        """
        Search logs based on query parameters.
        
        Args:
            query: Search query parameters
            
        Returns:
            List of matching log entries
        """
        sql_parts = ["SELECT * FROM log_entries WHERE 1=1"]
        params = []
        
        # Add time range filter
        if query.start_time:
            sql_parts.append("AND timestamp >= ?")
            params.append(query.start_time.isoformat())
        
        if query.end_time:
            sql_parts.append("AND timestamp <= ?")
            params.append(query.end_time.isoformat())
        
        # Add level filter
        if query.level:
            sql_parts.append("AND level = ?")
            params.append(query.level)
        
        # Add logger filter
        if query.logger:
            sql_parts.append("AND logger LIKE ?")
            params.append(f"%{query.logger}%")
        
        # Add message pattern filter
        if query.message_pattern:
            sql_parts.append("AND message LIKE ?")
            params.append(f"%{query.message_pattern}%")
        
        # Add request ID filter
        if query.request_id:
            sql_parts.append("AND request_id = ?")
            params.append(query.request_id)
        
        # Add ordering and pagination
        sql_parts.append("ORDER BY timestamp DESC")
        sql_parts.append("LIMIT ? OFFSET ?")
        params.extend([query.limit, query.offset])
        
        sql = " ".join(sql_parts)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(sql, params)
            
            results = []
            for row in cursor.fetchall():
                result = dict(row)
                
                # Parse JSON fields
                if result['extra_fields']:
                    result['extra_fields'] = json.loads(result['extra_fields'])
                
                if result['exception_traceback']:
                    result['exception_traceback'] = json.loads(result['exception_traceback'])
                
                results.append(result)
            
            return results
    
    def get_log_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get log statistics for the specified time period.
        
        Args:
            hours: Number of hours to analyze
            
        Returns:
            Dictionary with log statistics
        """
        start_time = datetime.utcnow() - timedelta(hours=hours)
        
        with sqlite3.connect(self.db_path) as conn:
            # Total log count
            total_count = conn.execute(
                "SELECT COUNT(*) FROM log_entries WHERE timestamp >= ?",
                (start_time.isoformat(),)
            ).fetchone()[0]
            
            # Count by level
            level_counts = {}
            for row in conn.execute("""
                SELECT level, COUNT(*) as count 
                FROM log_entries 
                WHERE timestamp >= ? 
                GROUP BY level
            """, (start_time.isoformat(),)):
                level_counts[row[0]] = row[1]
            
            # Count by logger
            logger_counts = {}
            for row in conn.execute("""
                SELECT logger, COUNT(*) as count 
                FROM log_entries 
                WHERE timestamp >= ? 
                GROUP BY logger 
                ORDER BY count DESC 
                LIMIT 10
            """, (start_time.isoformat(),)):
                logger_counts[row[0]] = row[1]
            
            # Error rate
            error_count = conn.execute(
                "SELECT COUNT(*) FROM log_entries WHERE timestamp >= ? AND level IN ('ERROR', 'CRITICAL')",
                (start_time.isoformat(),)
            ).fetchone()[0]
            
            error_rate = (error_count / total_count * 100) if total_count > 0 else 0
            
            return {
                'time_period_hours': hours,
                'total_entries': total_count,
                'error_count': error_count,
                'error_rate_percent': round(error_rate, 2),
                'level_distribution': level_counts,
                'top_loggers': logger_counts
            }
    
    def cleanup_old_logs(self, days: int = 30) -> int:
        """
        Remove log entries older than specified days.
        
        Args:
            days: Number of days to retain logs
            
        Returns:
            Number of deleted entries
        """
        cutoff_time = datetime.utcnow() - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM log_entries WHERE timestamp < ?",
                (cutoff_time.isoformat(),)
            )
            deleted_count = cursor.rowcount
        
        self.logger.info(
            f"Cleaned up old log entries",
            deleted_count=deleted_count,
            cutoff_days=days
        )
        
        return deleted_count


class LogWatcher:
    """Watch log files for new entries and automatically ingest them."""
    
    def __init__(self, aggregator: LogAggregator):
        self.aggregator = aggregator
        self.logger = get_structured_logger('log_watcher')
        self.watched_files = {}
    
    def watch_file(self, file_path: str) -> None:
        """
        Start watching a log file for new entries.
        
        Args:
            file_path: Path to the log file to watch
        """
        if not os.path.exists(file_path):
            self.logger.warning(f"Log file does not exist: {file_path}")
            return
        
        # Store the current file size to track new entries
        self.watched_files[file_path] = os.path.getsize(file_path)
        self.logger.info(f"Started watching log file: {file_path}")
    
    def check_for_updates(self) -> None:
        """Check watched files for new log entries."""
        for file_path, last_size in self.watched_files.items():
            if not os.path.exists(file_path):
                self.logger.warning(f"Watched file no longer exists: {file_path}")
                continue
            
            current_size = os.path.getsize(file_path)
            if current_size > last_size:
                # File has grown, ingest new entries
                try:
                    self._ingest_new_entries(file_path, last_size)
                    self.watched_files[file_path] = current_size
                except Exception as e:
                    self.logger.error(
                        f"Failed to ingest new entries from {file_path}",
                        error=str(e)
                    )
    
    def _ingest_new_entries(self, file_path: str, start_position: int) -> None:
        """Ingest only new entries from a log file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            f.seek(start_position)
            
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    log_data = json.loads(line)
                    entry = self.aggregator._parse_log_entry(log_data)
                    self.aggregator._store_log_entry(entry)
                except Exception as e:
                    self.logger.warning(
                        f"Failed to process log line",
                        file_path=file_path,
                        error=str(e)
                    )