"""
Asynchronous processing utilities for long-running tasks.
"""
import asyncio
import logging
import uuid
from typing import Any, Dict, Optional, Callable, Awaitable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import queue
import time

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskResult:
    """Task execution result."""
    task_id: str
    status: TaskStatus
    result: Any = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class AsyncTaskManager:
    """Manager for asynchronous task execution."""
    
    def __init__(self, max_workers: int = 4, max_process_workers: int = 2):
        self.max_workers = max_workers
        self.max_process_workers = max_process_workers
        self.thread_executor = ThreadPoolExecutor(max_workers=max_workers)
        self.process_executor = ProcessPoolExecutor(max_workers=max_process_workers)
        self.tasks: Dict[str, TaskResult] = {}
        self.task_futures: Dict[str, Any] = {}
        self.lock = threading.Lock()
        
        # Cleanup thread for old tasks
        self.cleanup_thread = threading.Thread(target=self._cleanup_old_tasks, daemon=True)
        self.cleanup_thread.start()
    
    def submit_task(self, 
                   func: Callable, 
                   *args, 
                   use_process: bool = False,
                   task_id: Optional[str] = None,
                   metadata: Optional[Dict[str, Any]] = None,
                   **kwargs) -> str:
        """
        Submit a task for asynchronous execution.
        
        Args:
            func: Function to execute
            *args: Function arguments
            use_process: Whether to use process pool instead of thread pool
            task_id: Optional custom task ID
            metadata: Optional task metadata
            **kwargs: Function keyword arguments
            
        Returns:
            Task ID for tracking
        """
        task_id = task_id or str(uuid.uuid4())
        
        with self.lock:
            if task_id in self.tasks:
                raise ValueError(f"Task {task_id} already exists")
            
            # Create task result
            task_result = TaskResult(
                task_id=task_id,
                status=TaskStatus.PENDING,
                metadata=metadata or {}
            )
            self.tasks[task_id] = task_result
            
            # Submit to appropriate executor
            executor = self.process_executor if use_process else self.thread_executor
            
            # Wrap function to update task status
            def wrapped_func():
                try:
                    with self.lock:
                        self.tasks[task_id].status = TaskStatus.RUNNING
                        self.tasks[task_id].started_at = datetime.utcnow()
                    
                    result = func(*args, **kwargs)
                    
                    with self.lock:
                        self.tasks[task_id].status = TaskStatus.COMPLETED
                        self.tasks[task_id].result = result
                        self.tasks[task_id].completed_at = datetime.utcnow()
                        self.tasks[task_id].progress = 1.0
                    
                    return result
                    
                except Exception as e:
                    logger.error(f"Task {task_id} failed: {e}")
                    with self.lock:
                        self.tasks[task_id].status = TaskStatus.FAILED
                        self.tasks[task_id].error = str(e)
                        self.tasks[task_id].completed_at = datetime.utcnow()
                    raise
            
            future = executor.submit(wrapped_func)
            self.task_futures[task_id] = future
            
            logger.info(f"Submitted task {task_id} to {'process' if use_process else 'thread'} pool")
            return task_id
    
    def get_task_status(self, task_id: str) -> Optional[TaskResult]:
        """Get task status and result."""
        with self.lock:
            return self.tasks.get(task_id)
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or running task."""
        with self.lock:
            if task_id not in self.tasks:
                return False
            
            task = self.tasks[task_id]
            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                return False
            
            # Try to cancel the future
            future = self.task_futures.get(task_id)
            if future and future.cancel():
                task.status = TaskStatus.CANCELLED
                task.completed_at = datetime.utcnow()
                return True
            
            return False
    
    def update_task_progress(self, task_id: str, progress: float, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Update task progress and metadata."""
        with self.lock:
            if task_id in self.tasks:
                self.tasks[task_id].progress = max(0.0, min(1.0, progress))
                if metadata:
                    self.tasks[task_id].metadata.update(metadata)
    
    def list_tasks(self, status: Optional[TaskStatus] = None) -> Dict[str, TaskResult]:
        """List all tasks, optionally filtered by status."""
        with self.lock:
            if status:
                return {tid: task for tid, task in self.tasks.items() if task.status == status}
            return self.tasks.copy()
    
    def cleanup_task(self, task_id: str) -> bool:
        """Remove a completed task from memory."""
        with self.lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                    del self.tasks[task_id]
                    if task_id in self.task_futures:
                        del self.task_futures[task_id]
                    return True
            return False
    
    def _cleanup_old_tasks(self) -> None:
        """Background thread to cleanup old completed tasks."""
        while True:
            try:
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                
                with self.lock:
                    tasks_to_remove = []
                    for task_id, task in self.tasks.items():
                        if (task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED] and
                            task.completed_at and task.completed_at < cutoff_time):
                            tasks_to_remove.append(task_id)
                    
                    for task_id in tasks_to_remove:
                        del self.tasks[task_id]
                        if task_id in self.task_futures:
                            del self.task_futures[task_id]
                    
                    if tasks_to_remove:
                        logger.info(f"Cleaned up {len(tasks_to_remove)} old tasks")
                
                time.sleep(3600)  # Check every hour
                
            except Exception as e:
                logger.error(f"Error in task cleanup: {e}")
                time.sleep(300)  # Wait 5 minutes on error
    
    def shutdown(self) -> None:
        """Shutdown the task manager."""
        logger.info("Shutting down async task manager")
        self.thread_executor.shutdown(wait=True)
        self.process_executor.shutdown(wait=True)


class TaskQueue:
    """Simple task queue for background processing."""
    
    def __init__(self, max_workers: int = 2):
        self.queue = queue.Queue()
        self.workers = []
        self.running = True
        
        # Start worker threads
        for i in range(max_workers):
            worker = threading.Thread(target=self._worker, daemon=True)
            worker.start()
            self.workers.append(worker)
    
    def _worker(self) -> None:
        """Worker thread function."""
        while self.running:
            try:
                task = self.queue.get(timeout=1)
                if task is None:  # Shutdown signal
                    break
                
                func, args, kwargs = task
                try:
                    func(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Task queue error: {e}")
                finally:
                    self.queue.task_done()
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker error: {e}")
    
    def submit(self, func: Callable, *args, **kwargs) -> None:
        """Submit a task to the queue."""
        if self.running:
            self.queue.put((func, args, kwargs))
    
    def shutdown(self) -> None:
        """Shutdown the task queue."""
        self.running = False
        
        # Send shutdown signals
        for _ in self.workers:
            self.queue.put(None)
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=5)


# Global instances
task_manager = AsyncTaskManager()
background_queue = TaskQueue()


def async_task(use_process: bool = False, task_id: Optional[str] = None):
    """
    Decorator to make a function run asynchronously.
    
    Args:
        use_process: Whether to use process pool
        task_id: Optional custom task ID
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            return task_manager.submit_task(
                func, *args, 
                use_process=use_process, 
                task_id=task_id,
                **kwargs
            )
        return wrapper
    return decorator


def background_task(func: Callable) -> Callable:
    """
    Decorator to run a function in the background queue.
    """
    def wrapper(*args, **kwargs):
        background_queue.submit(func, *args, **kwargs)
    return wrapper