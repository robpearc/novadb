"""S3 streaming processor for large-scale dataset processing.

Provides streaming capabilities for processing PDB/mmCIF files directly
from S3 without requiring full local downloads.

Features:
- Async I/O with aioboto3
- Configurable batch processing
- Progress tracking with ETA
- Automatic retries with exponential backoff
- Dead letter queue for failed items
- Checkpointing for job resumption
"""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import io
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    AsyncIterator,
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

# Type variable for generic processing
T = TypeVar("T")


class ProcessingStatus(Enum):
    """Status of a processing job or item."""
    PENDING = auto()
    IN_PROGRESS = auto()
    COMPLETED = auto()
    FAILED = auto()
    RETRYING = auto()
    DEAD_LETTER = auto()


@dataclass
class ProcessingStats:
    """Statistics for a processing job."""
    total_items: int = 0
    processed: int = 0
    failed: int = 0
    retried: int = 0
    dead_letter: int = 0
    bytes_downloaded: int = 0
    bytes_uploaded: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    @property
    def elapsed_seconds(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()
    
    @property
    def items_per_second(self) -> float:
        """Get processing rate."""
        elapsed = self.elapsed_seconds
        if elapsed == 0:
            return 0.0
        return self.processed / elapsed
    
    @property
    def eta_seconds(self) -> Optional[float]:
        """Get estimated time remaining in seconds."""
        if self.items_per_second == 0:
            return None
        remaining = self.total_items - self.processed
        return remaining / self.items_per_second
    
    @property
    def success_rate(self) -> float:
        """Get success rate as percentage."""
        if self.processed == 0:
            return 0.0
        return (self.processed - self.failed) / self.processed * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_items": self.total_items,
            "processed": self.processed,
            "failed": self.failed,
            "retried": self.retried,
            "dead_letter": self.dead_letter,
            "bytes_downloaded": self.bytes_downloaded,
            "bytes_uploaded": self.bytes_uploaded,
            "elapsed_seconds": self.elapsed_seconds,
            "items_per_second": self.items_per_second,
            "eta_seconds": self.eta_seconds,
            "success_rate": self.success_rate,
        }


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    
    def get_delay(self, attempt: int) -> float:
        """Get delay for a given attempt number."""
        import random
        delay = min(
            self.initial_delay * (self.exponential_base ** attempt),
            self.max_delay
        )
        if self.jitter:
            delay *= (0.5 + random.random())
        return delay


@dataclass
class S3StreamConfig:
    """Configuration for S3 streaming processor."""
    # AWS credentials (if not using default chain)
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_region: str = "us-east-1"
    endpoint_url: Optional[str] = None  # For S3-compatible services
    
    # Processing parameters
    max_concurrent: int = 50
    batch_size: int = 100
    buffer_size: int = 8 * 1024 * 1024  # 8MB buffer
    
    # Retry configuration
    retry: RetryConfig = field(default_factory=RetryConfig)
    
    # Checkpointing
    checkpoint_interval: int = 100  # Items between checkpoints
    checkpoint_path: Optional[Path] = None
    
    # Dead letter queue
    enable_dlq: bool = True
    dlq_bucket: Optional[str] = None
    dlq_prefix: str = "dead-letter/"
    
    # Progress reporting
    progress_interval: float = 5.0  # Seconds between progress updates
    
    # Filtering
    file_extensions: List[str] = field(default_factory=lambda: [".cif", ".cif.gz", ".pdb", ".pdb.gz"])
    max_file_size: int = 100 * 1024 * 1024  # 100MB max file size


@dataclass
class Checkpoint:
    """Checkpoint for resuming interrupted jobs."""
    job_id: str
    source_bucket: str
    source_prefix: str
    output_bucket: str
    output_prefix: str
    processed_keys: Set[str] = field(default_factory=set)
    failed_keys: Dict[str, str] = field(default_factory=dict)  # key -> error message
    stats: ProcessingStats = field(default_factory=ProcessingStats)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def save(self, path: Path) -> None:
        """Save checkpoint to file."""
        data = {
            "job_id": self.job_id,
            "source_bucket": self.source_bucket,
            "source_prefix": self.source_prefix,
            "output_bucket": self.output_bucket,
            "output_prefix": self.output_prefix,
            "processed_keys": list(self.processed_keys),
            "failed_keys": self.failed_keys,
            "stats": self.stats.to_dict(),
            "created_at": self.created_at.isoformat(),
            "updated_at": datetime.now().isoformat(),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> "Checkpoint":
        """Load checkpoint from file."""
        with open(path) as f:
            data = json.load(f)
        
        stats = ProcessingStats(**{
            k: v for k, v in data["stats"].items()
            if k in ProcessingStats.__dataclass_fields__
        })
        
        return cls(
            job_id=data["job_id"],
            source_bucket=data["source_bucket"],
            source_prefix=data["source_prefix"],
            output_bucket=data["output_bucket"],
            output_prefix=data["output_prefix"],
            processed_keys=set(data["processed_keys"]),
            failed_keys=data["failed_keys"],
            stats=stats,
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
        )


class S3StreamingProcessor:
    """Async S3 streaming processor for large-scale dataset processing.
    
    Example usage:
        ```python
        config = S3StreamConfig(
            max_concurrent=100,
            batch_size=50,
        )
        
        processor = S3StreamingProcessor(config)
        
        async def process_structure(key: str, data: bytes) -> dict:
            # Parse and process the structure
            structure = parse_mmcif(data)
            features = extract_features(structure)
            return features
        
        stats = await processor.process_bucket(
            source_bucket="pdb-structures",
            source_prefix="mmcif/",
            output_bucket="processed-data",
            output_prefix="features/",
            process_func=process_structure,
        )
        ```
    """
    
    def __init__(self, config: Optional[S3StreamConfig] = None):
        self.config = config or S3StreamConfig()
        self._session = None
        self._client = None
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._stats = ProcessingStats()
        self._checkpoint: Optional[Checkpoint] = None
        self._progress_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
    async def __aenter__(self) -> "S3StreamingProcessor":
        """Async context manager entry."""
        await self._init_client()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self._close_client()
    
    async def _init_client(self) -> None:
        """Initialize async S3 client."""
        try:
            import aioboto3
        except ImportError:
            raise ImportError(
                "aioboto3 is required for async S3 operations. "
                "Install it with: pip install aioboto3"
            )
        
        self._session = aioboto3.Session(
            aws_access_key_id=self.config.aws_access_key_id,
            aws_secret_access_key=self.config.aws_secret_access_key,
            region_name=self.config.aws_region,
        )
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent)
    
    async def _close_client(self) -> None:
        """Close async S3 client."""
        if self._progress_task:
            self._shutdown_event.set()
            self._progress_task.cancel()
            try:
                await self._progress_task
            except asyncio.CancelledError:
                pass
    
    async def _get_client(self):
        """Get S3 client from session."""
        client_kwargs = {}
        if self.config.endpoint_url:
            client_kwargs["endpoint_url"] = self.config.endpoint_url
        return self._session.client("s3", **client_kwargs)
    
    async def list_objects(
        self,
        bucket: str,
        prefix: str = "",
        filter_func: Optional[Callable[[str], bool]] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """List objects in S3 bucket with optional filtering.
        
        Args:
            bucket: S3 bucket name
            prefix: Key prefix to filter by
            filter_func: Optional function to filter keys
            
        Yields:
            Object metadata dictionaries
        """
        async with await self._get_client() as client:
            paginator = client.get_paginator("list_objects_v2")
            
            async for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    
                    # Apply extension filter
                    if self.config.file_extensions:
                        if not any(key.endswith(ext) for ext in self.config.file_extensions):
                            continue
                    
                    # Apply size filter
                    if obj["Size"] > self.config.max_file_size:
                        logger.warning(f"Skipping {key}: exceeds max size ({obj['Size']} bytes)")
                        continue
                    
                    # Apply custom filter
                    if filter_func and not filter_func(key):
                        continue
                    
                    yield obj
    
    async def download_object(
        self,
        bucket: str,
        key: str,
        decompress: bool = True,
    ) -> bytes:
        """Download object from S3.
        
        Args:
            bucket: S3 bucket name
            key: Object key
            decompress: Whether to decompress gzipped files
            
        Returns:
            Object data as bytes
        """
        async with self._semaphore:
            async with await self._get_client() as client:
                response = await client.get_object(Bucket=bucket, Key=key)
                data = await response["Body"].read()
                self._stats.bytes_downloaded += len(data)
                
                # Decompress if needed
                if decompress and key.endswith(".gz"):
                    data = gzip.decompress(data)
                
                return data
    
    async def upload_object(
        self,
        bucket: str,
        key: str,
        data: bytes,
        compress: bool = True,
        content_type: str = "application/octet-stream",
    ) -> None:
        """Upload object to S3.
        
        Args:
            bucket: S3 bucket name
            key: Object key
            data: Data to upload
            compress: Whether to gzip compress the data
            content_type: Content type header
        """
        async with self._semaphore:
            if compress:
                data = gzip.compress(data)
                key = key if key.endswith(".gz") else f"{key}.gz"
                content_type = "application/gzip"
            
            async with await self._get_client() as client:
                await client.put_object(
                    Bucket=bucket,
                    Key=key,
                    Body=data,
                    ContentType=content_type,
                )
                self._stats.bytes_uploaded += len(data)
    
    async def _process_item(
        self,
        source_bucket: str,
        key: str,
        output_bucket: str,
        output_prefix: str,
        process_func: Callable[[str, bytes], Any],
        serialize_func: Optional[Callable[[Any], bytes]] = None,
    ) -> Tuple[str, bool, Optional[str]]:
        """Process a single item with retries.
        
        Returns:
            Tuple of (key, success, error_message)
        """
        attempt = 0
        last_error = None
        
        while attempt <= self.config.retry.max_retries:
            try:
                # Download
                data = await self.download_object(source_bucket, key)
                
                # Process
                result = await asyncio.get_event_loop().run_in_executor(
                    None, process_func, key, data
                )
                
                # Upload result if we have one
                if result is not None and output_bucket:
                    # Serialize result
                    if serialize_func:
                        output_data = serialize_func(result)
                    elif isinstance(result, bytes):
                        output_data = result
                    elif isinstance(result, dict):
                        output_data = json.dumps(result).encode("utf-8")
                    else:
                        output_data = str(result).encode("utf-8")
                    
                    # Generate output key
                    base_name = Path(key).stem
                    if base_name.endswith(".cif"):
                        base_name = base_name[:-4]
                    output_key = f"{output_prefix}{base_name}.json"
                    
                    await self.upload_object(
                        output_bucket,
                        output_key,
                        output_data,
                        compress=True,
                    )
                
                return key, True, None
                
            except Exception as e:
                last_error = str(e)
                attempt += 1
                
                if attempt <= self.config.retry.max_retries:
                    delay = self.config.retry.get_delay(attempt)
                    logger.warning(
                        f"Retry {attempt}/{self.config.retry.max_retries} for {key}: {e}"
                    )
                    self._stats.retried += 1
                    await asyncio.sleep(delay)
        
        return key, False, last_error
    
    async def _send_to_dlq(
        self,
        bucket: str,
        key: str,
        error: str,
    ) -> None:
        """Send failed item to dead letter queue."""
        if not self.config.enable_dlq:
            return
        
        dlq_bucket = self.config.dlq_bucket or bucket
        dlq_key = f"{self.config.dlq_prefix}{key}.error.json"
        
        error_record = {
            "original_bucket": bucket,
            "original_key": key,
            "error": error,
            "timestamp": datetime.now().isoformat(),
            "job_id": self._checkpoint.job_id if self._checkpoint else None,
        }
        
        try:
            await self.upload_object(
                dlq_bucket,
                dlq_key,
                json.dumps(error_record).encode("utf-8"),
                compress=False,
            )
            self._stats.dead_letter += 1
        except Exception as e:
            logger.error(f"Failed to send to DLQ: {e}")
    
    async def _progress_reporter(self) -> None:
        """Background task to report progress."""
        while not self._shutdown_event.is_set():
            await asyncio.sleep(self.config.progress_interval)
            
            stats = self._stats
            eta = stats.eta_seconds
            eta_str = str(timedelta(seconds=int(eta))) if eta else "unknown"
            
            logger.info(
                f"Progress: {stats.processed}/{stats.total_items} "
                f"({stats.processed/stats.total_items*100:.1f}%) | "
                f"Rate: {stats.items_per_second:.1f}/s | "
                f"ETA: {eta_str} | "
                f"Failed: {stats.failed} | "
                f"Downloaded: {stats.bytes_downloaded / 1024 / 1024:.1f} MB"
            )
    
    async def process_bucket(
        self,
        source_bucket: str,
        source_prefix: str,
        output_bucket: str,
        output_prefix: str,
        process_func: Callable[[str, bytes], Any],
        serialize_func: Optional[Callable[[Any], bytes]] = None,
        filter_func: Optional[Callable[[str], bool]] = None,
        resume_from: Optional[Path] = None,
    ) -> ProcessingStats:
        """Process all matching objects in an S3 bucket.
        
        Args:
            source_bucket: Source S3 bucket
            source_prefix: Prefix to filter source objects
            output_bucket: Destination S3 bucket
            output_prefix: Prefix for output objects
            process_func: Function to process each object (key, data) -> result
            serialize_func: Optional function to serialize results
            filter_func: Optional function to filter source keys
            resume_from: Path to checkpoint file to resume from
            
        Returns:
            Processing statistics
        """
        await self._init_client()
        
        # Load or create checkpoint
        if resume_from and resume_from.exists():
            self._checkpoint = Checkpoint.load(resume_from)
            self._stats = self._checkpoint.stats
            logger.info(
                f"Resuming from checkpoint: {len(self._checkpoint.processed_keys)} "
                f"already processed"
            )
        else:
            job_id = hashlib.md5(
                f"{source_bucket}/{source_prefix}/{datetime.now().isoformat()}".encode()
            ).hexdigest()[:12]
            
            self._checkpoint = Checkpoint(
                job_id=job_id,
                source_bucket=source_bucket,
                source_prefix=source_prefix,
                output_bucket=output_bucket,
                output_prefix=output_prefix,
            )
        
        # Start progress reporter
        self._shutdown_event.clear()
        self._progress_task = asyncio.create_task(self._progress_reporter())
        
        try:
            # Collect all objects to process
            objects = []
            async for obj in self.list_objects(source_bucket, source_prefix, filter_func):
                key = obj["Key"]
                if key not in self._checkpoint.processed_keys:
                    objects.append(obj)
            
            self._stats.total_items = len(objects) + len(self._checkpoint.processed_keys)
            self._stats.start_time = self._stats.start_time or datetime.now()
            
            logger.info(
                f"Starting processing of {len(objects)} objects "
                f"({len(self._checkpoint.processed_keys)} already done)"
            )
            
            # Process in batches
            batch = []
            items_since_checkpoint = 0
            
            for obj in objects:
                batch.append(obj)
                
                if len(batch) >= self.config.batch_size:
                    await self._process_batch(
                        batch,
                        source_bucket,
                        output_bucket,
                        output_prefix,
                        process_func,
                        serialize_func,
                    )
                    batch = []
                    items_since_checkpoint += self.config.batch_size
                    
                    # Save checkpoint periodically
                    if (
                        self.config.checkpoint_path
                        and items_since_checkpoint >= self.config.checkpoint_interval
                    ):
                        self._checkpoint.save(self.config.checkpoint_path)
                        items_since_checkpoint = 0
            
            # Process remaining batch
            if batch:
                await self._process_batch(
                    batch,
                    source_bucket,
                    output_bucket,
                    output_prefix,
                    process_func,
                    serialize_func,
                )
            
            self._stats.end_time = datetime.now()
            
            # Save final checkpoint
            if self.config.checkpoint_path:
                self._checkpoint.save(self.config.checkpoint_path)
            
            return self._stats
            
        finally:
            await self._close_client()
    
    async def _process_batch(
        self,
        objects: List[Dict[str, Any]],
        source_bucket: str,
        output_bucket: str,
        output_prefix: str,
        process_func: Callable[[str, bytes], Any],
        serialize_func: Optional[Callable[[Any], bytes]],
    ) -> None:
        """Process a batch of objects concurrently."""
        tasks = [
            self._process_item(
                source_bucket,
                obj["Key"],
                output_bucket,
                output_prefix,
                process_func,
                serialize_func,
            )
            for obj in objects
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Unexpected error: {result}")
                self._stats.failed += 1
            else:
                key, success, error = result
                self._stats.processed += 1
                
                if success:
                    self._checkpoint.processed_keys.add(key)
                else:
                    self._stats.failed += 1
                    self._checkpoint.failed_keys[key] = error
                    await self._send_to_dlq(source_bucket, key, error)


class S3SyncManager:
    """Manage synchronization between S3 and local storage.
    
    Supports bidirectional sync with conflict resolution.
    """
    
    def __init__(self, config: S3StreamConfig):
        self.config = config
        self._processor = S3StreamingProcessor(config)
    
    async def sync_to_local(
        self,
        bucket: str,
        prefix: str,
        local_path: Path,
        filter_func: Optional[Callable[[str], bool]] = None,
        overwrite: bool = False,
    ) -> ProcessingStats:
        """Sync S3 objects to local filesystem.
        
        Args:
            bucket: Source S3 bucket
            prefix: S3 prefix to sync
            local_path: Local destination directory
            filter_func: Optional filter function
            overwrite: Whether to overwrite existing files
            
        Returns:
            Sync statistics
        """
        local_path.mkdir(parents=True, exist_ok=True)
        stats = ProcessingStats()
        stats.start_time = datetime.now()
        
        async with self._processor:
            async for obj in self._processor.list_objects(bucket, prefix, filter_func):
                key = obj["Key"]
                rel_path = key[len(prefix):].lstrip("/")
                dest_path = local_path / rel_path
                
                stats.total_items += 1
                
                # Skip if exists and not overwriting
                if dest_path.exists() and not overwrite:
                    stats.processed += 1
                    continue
                
                try:
                    data = await self._processor.download_object(bucket, key)
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    dest_path.write_bytes(data)
                    stats.processed += 1
                    stats.bytes_downloaded += len(data)
                except Exception as e:
                    logger.error(f"Failed to sync {key}: {e}")
                    stats.failed += 1
        
        stats.end_time = datetime.now()
        return stats
    
    async def sync_to_s3(
        self,
        local_path: Path,
        bucket: str,
        prefix: str,
        filter_func: Optional[Callable[[Path], bool]] = None,
        overwrite: bool = False,
    ) -> ProcessingStats:
        """Sync local files to S3.
        
        Args:
            local_path: Local source directory
            bucket: Destination S3 bucket
            prefix: S3 prefix for uploaded files
            filter_func: Optional filter function
            overwrite: Whether to overwrite existing objects
            
        Returns:
            Sync statistics
        """
        stats = ProcessingStats()
        stats.start_time = datetime.now()
        
        # Collect files
        files = list(local_path.rglob("*"))
        files = [f for f in files if f.is_file()]
        
        if filter_func:
            files = [f for f in files if filter_func(f)]
        
        stats.total_items = len(files)
        
        async with self._processor:
            # Get existing keys if not overwriting
            existing_keys = set()
            if not overwrite:
                async for obj in self._processor.list_objects(bucket, prefix):
                    existing_keys.add(obj["Key"])
            
            for file_path in files:
                rel_path = file_path.relative_to(local_path)
                key = f"{prefix}{rel_path}"
                
                if key in existing_keys and not overwrite:
                    stats.processed += 1
                    continue
                
                try:
                    data = file_path.read_bytes()
                    await self._processor.upload_object(bucket, key, data)
                    stats.processed += 1
                    stats.bytes_uploaded += len(data)
                except Exception as e:
                    logger.error(f"Failed to upload {file_path}: {e}")
                    stats.failed += 1
        
        stats.end_time = datetime.now()
        return stats
