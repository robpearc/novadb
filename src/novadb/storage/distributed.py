"""Distributed processing coordinator for large-scale dataset processing.

Provides distributed work coordination using Redis for:
- Job distribution across multiple workers
- Work stealing for load balancing
- Fault tolerance with automatic task reassignment
- Progress aggregation across workers
- Distributed checkpointing
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import signal
import socket
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


class TaskStatus(Enum):
    """Status of a distributed task."""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    DEAD_LETTER = "dead_letter"


class WorkerStatus(Enum):
    """Status of a distributed worker."""
    STARTING = "starting"
    IDLE = "idle"
    BUSY = "busy"
    STOPPING = "stopping"
    STOPPED = "stopped"
    DEAD = "dead"


@dataclass
class DistributedConfig:
    """Configuration for distributed processing."""
    # Redis connection
    redis_url: str = "redis://localhost:6379"
    redis_db: int = 0
    redis_password: Optional[str] = None
    
    # Worker configuration
    worker_id: Optional[str] = None  # Auto-generated if not provided
    worker_heartbeat_interval: float = 5.0
    worker_timeout: float = 30.0  # Consider worker dead after this
    max_tasks_per_worker: int = 10
    
    # Task configuration
    task_timeout: float = 300.0  # 5 minutes default
    max_task_retries: int = 3
    task_visibility_timeout: float = 60.0  # Time before reassigning stuck task
    
    # Queue configuration
    queue_name: str = "novadb:tasks"
    results_ttl: int = 86400  # 24 hours
    
    # Batch processing
    prefetch_count: int = 5
    batch_size: int = 10


@dataclass
class Task:
    """A distributed processing task."""
    task_id: str
    job_id: str
    payload: Dict[str, Any]
    status: TaskStatus = TaskStatus.PENDING
    attempts: int = 0
    worker_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "task_id": self.task_id,
            "job_id": self.job_id,
            "payload": self.payload,
            "status": self.status.value,
            "attempts": self.attempts,
            "worker_id": self.worker_id,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": self.error,
            "result": self.result,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        """Create from dictionary."""
        return cls(
            task_id=data["task_id"],
            job_id=data["job_id"],
            payload=data["payload"],
            status=TaskStatus(data["status"]),
            attempts=data["attempts"],
            worker_id=data.get("worker_id"),
            created_at=datetime.fromisoformat(data["created_at"]),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            error=data.get("error"),
            result=data.get("result"),
        )


@dataclass
class Job:
    """A distributed processing job containing multiple tasks."""
    job_id: str
    name: str
    source_bucket: str
    source_prefix: str
    output_bucket: str
    output_prefix: str
    config: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "job_id": self.job_id,
            "name": self.name,
            "source_bucket": self.source_bucket,
            "source_prefix": self.source_prefix,
            "output_bucket": self.output_bucket,
            "output_prefix": self.output_prefix,
            "config": self.config,
            "status": self.status,
            "total_tasks": self.total_tasks,
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Job":
        """Create from dictionary."""
        return cls(
            job_id=data["job_id"],
            name=data["name"],
            source_bucket=data["source_bucket"],
            source_prefix=data["source_prefix"],
            output_bucket=data["output_bucket"],
            output_prefix=data["output_prefix"],
            config=data.get("config", {}),
            status=data["status"],
            total_tasks=data["total_tasks"],
            completed_tasks=data["completed_tasks"],
            failed_tasks=data["failed_tasks"],
            created_at=datetime.fromisoformat(data["created_at"]),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
        )


@dataclass
class WorkerInfo:
    """Information about a distributed worker."""
    worker_id: str
    hostname: str
    pid: int
    status: WorkerStatus
    current_tasks: List[str] = field(default_factory=list)
    tasks_completed: int = 0
    tasks_failed: int = 0
    started_at: datetime = field(default_factory=datetime.now)
    last_heartbeat: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "worker_id": self.worker_id,
            "hostname": self.hostname,
            "pid": self.pid,
            "status": self.status.value,
            "current_tasks": self.current_tasks,
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "started_at": self.started_at.isoformat(),
            "last_heartbeat": self.last_heartbeat.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkerInfo":
        """Create from dictionary."""
        return cls(
            worker_id=data["worker_id"],
            hostname=data["hostname"],
            pid=data["pid"],
            status=WorkerStatus(data["status"]),
            current_tasks=data.get("current_tasks", []),
            tasks_completed=data.get("tasks_completed", 0),
            tasks_failed=data.get("tasks_failed", 0),
            started_at=datetime.fromisoformat(data["started_at"]),
            last_heartbeat=datetime.fromisoformat(data["last_heartbeat"]),
        )


class DistributedCoordinator:
    """Coordinator for distributed processing jobs.
    
    Manages job submission, task distribution, and progress tracking
    across multiple workers using Redis as the coordination backend.
    
    Example usage:
        ```python
        # Create coordinator
        coordinator = DistributedCoordinator(DistributedConfig(
            redis_url="redis://localhost:6379"
        ))
        
        # Submit a job
        job_id = await coordinator.submit_job(
            name="process_pdb",
            source_bucket="pdb-structures",
            source_prefix="mmcif/",
            output_bucket="processed-data",
            output_prefix="features/",
        )
        
        # Monitor progress
        while True:
            status = await coordinator.get_job_status(job_id)
            print(f"Progress: {status.completed_tasks}/{status.total_tasks}")
            if status.status in ("completed", "failed"):
                break
            await asyncio.sleep(5)
        ```
    """
    
    def __init__(self, config: Optional[DistributedConfig] = None):
        self.config = config or DistributedConfig()
        self._redis = None
        self._initialized = False
    
    async def _ensure_initialized(self) -> None:
        """Ensure Redis connection is initialized."""
        if self._initialized:
            return
        
        try:
            import redis.asyncio as redis
        except ImportError:
            raise ImportError(
                "redis is required for distributed processing. "
                "Install it with: pip install redis[hiredis]"
            )
        
        self._redis = redis.from_url(
            self.config.redis_url,
            db=self.config.redis_db,
            password=self.config.redis_password,
            decode_responses=True,
        )
        self._initialized = True
    
    async def close(self) -> None:
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            self._initialized = False
    
    def _job_key(self, job_id: str) -> str:
        """Get Redis key for a job."""
        return f"novadb:job:{job_id}"
    
    def _task_key(self, task_id: str) -> str:
        """Get Redis key for a task."""
        return f"novadb:task:{task_id}"
    
    def _worker_key(self, worker_id: str) -> str:
        """Get Redis key for a worker."""
        return f"novadb:worker:{worker_id}"
    
    def _queue_key(self, job_id: str) -> str:
        """Get Redis key for a job's task queue."""
        return f"novadb:queue:{job_id}"
    
    def _processing_key(self, job_id: str) -> str:
        """Get Redis key for a job's processing set."""
        return f"novadb:processing:{job_id}"
    
    async def submit_job(
        self,
        name: str,
        source_bucket: str,
        source_prefix: str,
        output_bucket: str,
        output_prefix: str,
        task_keys: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Submit a new processing job.
        
        Args:
            name: Job name for identification
            source_bucket: S3 bucket containing source files
            source_prefix: Prefix to filter source files
            output_bucket: S3 bucket for output files
            output_prefix: Prefix for output files
            task_keys: Optional list of specific keys to process
            config: Optional job configuration
            
        Returns:
            Job ID
        """
        await self._ensure_initialized()
        
        job_id = str(uuid.uuid4())[:8]
        job = Job(
            job_id=job_id,
            name=name,
            source_bucket=source_bucket,
            source_prefix=source_prefix,
            output_bucket=output_bucket,
            output_prefix=output_prefix,
            config=config or {},
        )
        
        # Store job
        await self._redis.set(
            self._job_key(job_id),
            json.dumps(job.to_dict()),
        )
        
        # Create tasks from keys or list bucket
        if task_keys:
            await self._create_tasks_from_keys(job_id, task_keys)
        else:
            # Queue a scan task to populate the queue
            scan_task = Task(
                task_id=f"{job_id}_scan",
                job_id=job_id,
                payload={
                    "type": "scan",
                    "bucket": source_bucket,
                    "prefix": source_prefix,
                },
            )
            await self._redis.rpush(
                self._queue_key(job_id),
                json.dumps(scan_task.to_dict()),
            )
        
        logger.info(f"Submitted job {job_id}: {name}")
        return job_id
    
    async def _create_tasks_from_keys(
        self,
        job_id: str,
        keys: List[str],
    ) -> None:
        """Create tasks from a list of S3 keys."""
        job_data = await self._redis.get(self._job_key(job_id))
        job = Job.from_dict(json.loads(job_data))
        
        pipeline = self._redis.pipeline()
        
        for i, key in enumerate(keys):
            task = Task(
                task_id=f"{job_id}_{i:06d}",
                job_id=job_id,
                payload={
                    "type": "process",
                    "source_bucket": job.source_bucket,
                    "source_key": key,
                    "output_bucket": job.output_bucket,
                    "output_prefix": job.output_prefix,
                },
            )
            pipeline.rpush(
                self._queue_key(job_id),
                json.dumps(task.to_dict()),
            )
        
        # Update job total
        job.total_tasks = len(keys)
        job.status = "running"
        job.started_at = datetime.now()
        pipeline.set(self._job_key(job_id), json.dumps(job.to_dict()))
        
        await pipeline.execute()
    
    async def get_job_status(self, job_id: str) -> Optional[Job]:
        """Get current status of a job."""
        await self._ensure_initialized()
        
        job_data = await self._redis.get(self._job_key(job_id))
        if not job_data:
            return None
        
        return Job.from_dict(json.loads(job_data))
    
    async def list_jobs(
        self,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List[Job]:
        """List all jobs, optionally filtered by status."""
        await self._ensure_initialized()
        
        # Scan for all job keys
        jobs = []
        async for key in self._redis.scan_iter("novadb:job:*"):
            job_data = await self._redis.get(key)
            if job_data:
                job = Job.from_dict(json.loads(job_data))
                if status is None or job.status == status:
                    jobs.append(job)
                    if len(jobs) >= limit:
                        break
        
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job."""
        await self._ensure_initialized()
        
        job_data = await self._redis.get(self._job_key(job_id))
        if not job_data:
            return False
        
        job = Job.from_dict(json.loads(job_data))
        job.status = "cancelled"
        job.completed_at = datetime.now()
        
        # Clear queue
        await self._redis.delete(self._queue_key(job_id))
        
        # Update job
        await self._redis.set(self._job_key(job_id), json.dumps(job.to_dict()))
        
        logger.info(f"Cancelled job {job_id}")
        return True
    
    async def get_workers(self) -> List[WorkerInfo]:
        """Get list of all registered workers."""
        await self._ensure_initialized()
        
        workers = []
        async for key in self._redis.scan_iter("novadb:worker:*"):
            worker_data = await self._redis.get(key)
            if worker_data:
                worker = WorkerInfo.from_dict(json.loads(worker_data))
                workers.append(worker)
        
        return workers
    
    async def cleanup_dead_workers(self) -> int:
        """Clean up dead workers and reassign their tasks."""
        await self._ensure_initialized()
        
        cleaned = 0
        now = datetime.now()
        timeout = timedelta(seconds=self.config.worker_timeout)
        
        async for key in self._redis.scan_iter("novadb:worker:*"):
            worker_data = await self._redis.get(key)
            if not worker_data:
                continue
            
            worker = WorkerInfo.from_dict(json.loads(worker_data))
            
            if now - worker.last_heartbeat > timeout:
                logger.warning(f"Worker {worker.worker_id} is dead, cleaning up")
                
                # Reassign tasks
                for task_id in worker.current_tasks:
                    await self._reassign_task(task_id)
                
                # Remove worker
                await self._redis.delete(key)
                cleaned += 1
        
        return cleaned
    
    async def _reassign_task(self, task_id: str) -> None:
        """Reassign a task back to the queue."""
        task_data = await self._redis.get(self._task_key(task_id))
        if not task_data:
            return
        
        task = Task.from_dict(json.loads(task_data))
        
        if task.attempts < self.config.max_task_retries:
            task.status = TaskStatus.PENDING
            task.worker_id = None
            task.attempts += 1
            
            # Re-queue
            await self._redis.rpush(
                self._queue_key(task.job_id),
                json.dumps(task.to_dict()),
            )
        else:
            task.status = TaskStatus.DEAD_LETTER
            
        await self._redis.set(self._task_key(task_id), json.dumps(task.to_dict()))


class DistributedWorker:
    """Worker for distributed processing.
    
    Fetches tasks from the coordinator and processes them.
    
    Example usage:
        ```python
        async def process_structure(task: Task) -> dict:
            key = task.payload["source_key"]
            # Download and process
            return {"status": "success"}
        
        worker = DistributedWorker(
            DistributedConfig(redis_url="redis://localhost:6379"),
            process_func=process_structure,
        )
        
        await worker.run()
        ```
    """
    
    def __init__(
        self,
        config: DistributedConfig,
        process_func: Callable[[Task], Any],
    ):
        self.config = config
        self.process_func = process_func
        self._redis = None
        self._worker_info: Optional[WorkerInfo] = None
        self._running = False
        self._shutdown_event = asyncio.Event()
    
    async def _ensure_initialized(self) -> None:
        """Ensure Redis connection is initialized."""
        if self._redis:
            return
        
        try:
            import redis.asyncio as redis
        except ImportError:
            raise ImportError(
                "redis is required for distributed processing. "
                "Install it with: pip install redis[hiredis]"
            )
        
        self._redis = redis.from_url(
            self.config.redis_url,
            db=self.config.redis_db,
            password=self.config.redis_password,
            decode_responses=True,
        )
        
        # Register worker
        worker_id = self.config.worker_id or f"{socket.gethostname()}_{os.getpid()}_{uuid.uuid4().hex[:6]}"
        self._worker_info = WorkerInfo(
            worker_id=worker_id,
            hostname=socket.gethostname(),
            pid=os.getpid(),
            status=WorkerStatus.STARTING,
        )
        
        await self._update_worker_info()
    
    async def _update_worker_info(self) -> None:
        """Update worker info in Redis."""
        self._worker_info.last_heartbeat = datetime.now()
        await self._redis.set(
            f"novadb:worker:{self._worker_info.worker_id}",
            json.dumps(self._worker_info.to_dict()),
            ex=int(self.config.worker_timeout * 2),
        )
    
    async def _heartbeat_loop(self) -> None:
        """Background task to send heartbeats."""
        while not self._shutdown_event.is_set():
            try:
                await self._update_worker_info()
            except Exception as e:
                logger.error(f"Heartbeat failed: {e}")
            
            await asyncio.sleep(self.config.worker_heartbeat_interval)
    
    async def _fetch_task(self, job_id: str) -> Optional[Task]:
        """Fetch a task from the queue."""
        queue_key = f"novadb:queue:{job_id}"
        
        # Blocking pop with timeout
        result = await self._redis.blpop(queue_key, timeout=1)
        if not result:
            return None
        
        _, task_data = result
        task = Task.from_dict(json.loads(task_data))
        
        # Mark as assigned
        task.status = TaskStatus.ASSIGNED
        task.worker_id = self._worker_info.worker_id
        task.started_at = datetime.now()
        task.attempts += 1
        
        # Store task and add to worker's current tasks
        self._worker_info.current_tasks.append(task.task_id)
        
        await self._redis.set(
            f"novadb:task:{task.task_id}",
            json.dumps(task.to_dict()),
        )
        
        return task
    
    async def _complete_task(
        self,
        task: Task,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        """Mark a task as completed or failed."""
        if error:
            task.status = TaskStatus.FAILED
            task.error = error
            self._worker_info.tasks_failed += 1
        else:
            task.status = TaskStatus.COMPLETED
            task.result = result
            self._worker_info.tasks_completed += 1
        
        task.completed_at = datetime.now()
        
        # Update task
        await self._redis.set(
            f"novadb:task:{task.task_id}",
            json.dumps(task.to_dict()),
            ex=self.config.results_ttl,
        )
        
        # Remove from worker's current tasks
        if task.task_id in self._worker_info.current_tasks:
            self._worker_info.current_tasks.remove(task.task_id)
        
        # Update job counters
        job_key = f"novadb:job:{task.job_id}"
        job_data = await self._redis.get(job_key)
        if job_data:
            job = Job.from_dict(json.loads(job_data))
            if error:
                job.failed_tasks += 1
            else:
                job.completed_tasks += 1
            
            # Check if job is complete
            if job.completed_tasks + job.failed_tasks >= job.total_tasks:
                job.status = "completed" if job.failed_tasks == 0 else "completed_with_errors"
                job.completed_at = datetime.now()
            
            await self._redis.set(job_key, json.dumps(job.to_dict()))
    
    async def run(self, job_ids: Optional[List[str]] = None) -> None:
        """Run the worker, processing tasks until shutdown.
        
        Args:
            job_ids: Optional list of job IDs to process. If None, processes all jobs.
        """
        await self._ensure_initialized()
        
        self._running = True
        self._worker_info.status = WorkerStatus.IDLE
        await self._update_worker_info()
        
        # Start heartbeat
        heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        
        # Set up signal handlers
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(self.shutdown()))
        
        logger.info(f"Worker {self._worker_info.worker_id} started")
        
        try:
            while self._running:
                try:
                    # Get list of active jobs
                    if job_ids:
                        active_jobs = job_ids
                    else:
                        # Find running jobs
                        active_jobs = []
                        async for key in self._redis.scan_iter("novadb:job:*"):
                            job_data = await self._redis.get(key)
                            if job_data:
                                job = Job.from_dict(json.loads(job_data))
                                if job.status == "running":
                                    active_jobs.append(job.job_id)
                    
                    if not active_jobs:
                        await asyncio.sleep(1)
                        continue
                    
                    # Try to fetch a task from each job
                    task = None
                    for job_id in active_jobs:
                        task = await self._fetch_task(job_id)
                        if task:
                            break
                    
                    if not task:
                        await asyncio.sleep(0.1)
                        continue
                    
                    # Process task
                    self._worker_info.status = WorkerStatus.BUSY
                    await self._update_worker_info()
                    
                    try:
                        result = await asyncio.wait_for(
                            self._process_task(task),
                            timeout=self.config.task_timeout,
                        )
                        await self._complete_task(task, result=result)
                    except asyncio.TimeoutError:
                        await self._complete_task(task, error="Task timed out")
                    except Exception as e:
                        await self._complete_task(task, error=str(e))
                    
                    self._worker_info.status = WorkerStatus.IDLE
                    await self._update_worker_info()
                    
                except Exception as e:
                    logger.error(f"Worker error: {e}")
                    await asyncio.sleep(1)
        
        finally:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass
            
            self._worker_info.status = WorkerStatus.STOPPED
            await self._update_worker_info()
            await self._redis.close()
    
    async def _process_task(self, task: Task) -> Optional[Dict[str, Any]]:
        """Process a single task."""
        if asyncio.iscoroutinefunction(self.process_func):
            return await self.process_func(task)
        else:
            return await asyncio.get_event_loop().run_in_executor(
                None, self.process_func, task
            )
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the worker."""
        logger.info(f"Worker {self._worker_info.worker_id} shutting down")
        self._running = False
        self._shutdown_event.set()
        self._worker_info.status = WorkerStatus.STOPPING
        await self._update_worker_info()


class TaskBatcher:
    """Utility for batching tasks for efficient processing."""
    
    def __init__(self, batch_size: int = 10):
        self.batch_size = batch_size
        self._batch: List[Task] = []
    
    def add(self, task: Task) -> Optional[List[Task]]:
        """Add a task, returning a batch if ready."""
        self._batch.append(task)
        if len(self._batch) >= self.batch_size:
            batch = self._batch
            self._batch = []
            return batch
        return None
    
    def flush(self) -> List[Task]:
        """Flush any remaining tasks."""
        batch = self._batch
        self._batch = []
        return batch


class ProgressAggregator:
    """Aggregates progress across multiple workers and jobs."""
    
    def __init__(self, coordinator: DistributedCoordinator):
        self.coordinator = coordinator
    
    async def get_overall_progress(self) -> Dict[str, Any]:
        """Get aggregated progress across all jobs."""
        jobs = await self.coordinator.list_jobs()
        workers = await self.coordinator.get_workers()
        
        total_tasks = sum(j.total_tasks for j in jobs)
        completed_tasks = sum(j.completed_tasks for j in jobs)
        failed_tasks = sum(j.failed_tasks for j in jobs)
        
        active_workers = len([
            w for w in workers
            if w.status in (WorkerStatus.IDLE, WorkerStatus.BUSY)
        ])
        
        return {
            "total_jobs": len(jobs),
            "running_jobs": len([j for j in jobs if j.status == "running"]),
            "completed_jobs": len([j for j in jobs if j.status == "completed"]),
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "failed_tasks": failed_tasks,
            "progress_percent": (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0,
            "active_workers": active_workers,
            "total_workers": len(workers),
        }
    
    async def get_job_progress(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed progress for a specific job."""
        job = await self.coordinator.get_job_status(job_id)
        if not job:
            return None
        
        progress = job.completed_tasks / job.total_tasks if job.total_tasks > 0 else 0
        
        elapsed = None
        eta = None
        if job.started_at:
            elapsed = (datetime.now() - job.started_at).total_seconds()
            if progress > 0:
                eta = elapsed / progress * (1 - progress)
        
        return {
            "job_id": job.job_id,
            "name": job.name,
            "status": job.status,
            "total_tasks": job.total_tasks,
            "completed_tasks": job.completed_tasks,
            "failed_tasks": job.failed_tasks,
            "progress_percent": progress * 100,
            "elapsed_seconds": elapsed,
            "eta_seconds": eta,
        }
