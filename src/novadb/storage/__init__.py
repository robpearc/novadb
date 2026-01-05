"""Storage module for NovaDB.

Provides storage backends and serialization for processed data
with support for local and cloud storage, streaming, and distributed processing.
"""

from novadb.storage.backends import (
    StorageConfig,
    StorageBackend,
    LocalStorage,
    S3Storage,
    GCSStorage,
    AzureStorage,
    create_storage,
)
from novadb.storage.serialization import (
    DataSerializer,
    DatasetWriter,
    DatasetReader,
)
from novadb.storage.streaming import (
    ProcessingStatus,
    ProcessingStats,
    RetryConfig,
    S3StreamConfig,
    Checkpoint,
    S3StreamingProcessor,
    S3SyncManager,
)
from novadb.storage.distributed import (
    TaskStatus,
    WorkerStatus,
    DistributedConfig,
    Task,
    Job,
    WorkerInfo,
    DistributedCoordinator,
    DistributedWorker,
    TaskBatcher,
    ProgressAggregator,
)

__all__ = [
    # Backends
    "StorageConfig",
    "StorageBackend",
    "LocalStorage",
    "S3Storage",
    "GCSStorage",
    "AzureStorage",
    "create_storage",
    # Serialization
    "DataSerializer",
    "DatasetWriter",
    "DatasetReader",
    # Streaming
    "ProcessingStatus",
    "ProcessingStats",
    "RetryConfig",
    "S3StreamConfig",
    "Checkpoint",
    "S3StreamingProcessor",
    "S3SyncManager",
    # Distributed
    "TaskStatus",
    "WorkerStatus",
    "DistributedConfig",
    "Task",
    "Job",
    "WorkerInfo",
    "DistributedCoordinator",
    "DistributedWorker",
    "TaskBatcher",
    "ProgressAggregator",
]
