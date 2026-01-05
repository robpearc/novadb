"""Storage backends for processed data.

Provides unified interface for local and cloud storage:
- Local filesystem
- Amazon S3
- Google Cloud Storage
- Azure Blob Storage
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Union
import numpy as np


@dataclass
class StorageConfig:
    """Configuration for storage backend."""
    backend: str = "local"  # local, s3, gcs, azure
    bucket: str = ""
    prefix: str = ""
    region: str = ""

    # Local storage
    local_path: str = "./data"

    # S3 specific
    s3_access_key: Optional[str] = None
    s3_secret_key: Optional[str] = None
    s3_endpoint_url: Optional[str] = None

    # GCS specific
    gcs_project: Optional[str] = None
    gcs_credentials_file: Optional[str] = None

    # Azure specific
    azure_connection_string: Optional[str] = None
    azure_account_name: Optional[str] = None
    azure_account_key: Optional[str] = None


class StorageBackend(ABC):
    """Abstract base class for storage backends."""

    @abstractmethod
    def put(self, key: str, data: bytes) -> None:
        """Store data at key."""
        pass

    @abstractmethod
    def get(self, key: str) -> bytes:
        """Retrieve data from key."""
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists."""
        pass

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete data at key."""
        pass

    @abstractmethod
    def list(self, prefix: str = "") -> List[str]:
        """List keys with given prefix."""
        pass

    def put_json(self, key: str, data: Dict[str, Any]) -> None:
        """Store JSON data."""
        self.put(key, json.dumps(data).encode("utf-8"))

    def get_json(self, key: str) -> Dict[str, Any]:
        """Retrieve JSON data."""
        return json.loads(self.get(key).decode("utf-8"))

    def put_numpy(self, key: str, arrays: Dict[str, np.ndarray]) -> None:
        """Store numpy arrays."""
        import io
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **arrays)
        buffer.seek(0)
        self.put(key, buffer.read())

    def get_numpy(self, key: str) -> Dict[str, np.ndarray]:
        """Retrieve numpy arrays."""
        import io
        buffer = io.BytesIO(self.get(key))
        data = np.load(buffer, allow_pickle=False)
        return dict(data)


class LocalStorage(StorageBackend):
    """Local filesystem storage."""

    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _resolve_path(self, key: str) -> Path:
        """Resolve key to full path."""
        return self.base_path / key

    def put(self, key: str, data: bytes) -> None:
        """Store data at key."""
        path = self._resolve_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def get(self, key: str) -> bytes:
        """Retrieve data from key."""
        path = self._resolve_path(key)
        if not path.exists():
            raise FileNotFoundError(f"Key not found: {key}")
        return path.read_bytes()

    def exists(self, key: str) -> bool:
        """Check if key exists."""
        return self._resolve_path(key).exists()

    def delete(self, key: str) -> None:
        """Delete data at key."""
        path = self._resolve_path(key)
        if path.exists():
            path.unlink()

    def list(self, prefix: str = "") -> List[str]:
        """List keys with given prefix."""
        search_path = self.base_path / prefix
        if not search_path.exists():
            return []

        keys = []
        for path in search_path.rglob("*"):
            if path.is_file():
                rel_path = path.relative_to(self.base_path)
                keys.append(str(rel_path))

        return sorted(keys)


class S3Storage(StorageBackend):
    """Amazon S3 storage backend."""

    def __init__(self, config: StorageConfig):
        import boto3

        self.bucket = config.bucket
        self.prefix = config.prefix

        # Create S3 client
        client_kwargs = {}
        if config.s3_access_key and config.s3_secret_key:
            client_kwargs["aws_access_key_id"] = config.s3_access_key
            client_kwargs["aws_secret_access_key"] = config.s3_secret_key
        if config.region:
            client_kwargs["region_name"] = config.region
        if config.s3_endpoint_url:
            client_kwargs["endpoint_url"] = config.s3_endpoint_url

        self.client = boto3.client("s3", **client_kwargs)

    def _make_key(self, key: str) -> str:
        """Make full S3 key with prefix."""
        if self.prefix:
            return f"{self.prefix}/{key}"
        return key

    def put(self, key: str, data: bytes) -> None:
        """Store data at key."""
        self.client.put_object(
            Bucket=self.bucket,
            Key=self._make_key(key),
            Body=data,
        )

    def get(self, key: str) -> bytes:
        """Retrieve data from key."""
        response = self.client.get_object(
            Bucket=self.bucket,
            Key=self._make_key(key),
        )
        return response["Body"].read()

    def exists(self, key: str) -> bool:
        """Check if key exists."""
        try:
            self.client.head_object(
                Bucket=self.bucket,
                Key=self._make_key(key),
            )
            return True
        except self.client.exceptions.ClientError:
            return False

    def delete(self, key: str) -> None:
        """Delete data at key."""
        self.client.delete_object(
            Bucket=self.bucket,
            Key=self._make_key(key),
        )

    def list(self, prefix: str = "") -> List[str]:
        """List keys with given prefix."""
        full_prefix = self._make_key(prefix)
        paginator = self.client.get_paginator("list_objects_v2")

        keys = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=full_prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if self.prefix:
                    key = key[len(self.prefix) + 1 :]
                keys.append(key)

        return sorted(keys)


class GCSStorage(StorageBackend):
    """Google Cloud Storage backend."""

    def __init__(self, config: StorageConfig):
        from google.cloud import storage

        self.bucket_name = config.bucket
        self.prefix = config.prefix

        # Create client
        if config.gcs_credentials_file:
            self.client = storage.Client.from_service_account_json(
                config.gcs_credentials_file
            )
        else:
            self.client = storage.Client(project=config.gcs_project)

        self.bucket = self.client.bucket(self.bucket_name)

    def _make_key(self, key: str) -> str:
        """Make full GCS key with prefix."""
        if self.prefix:
            return f"{self.prefix}/{key}"
        return key

    def put(self, key: str, data: bytes) -> None:
        """Store data at key."""
        blob = self.bucket.blob(self._make_key(key))
        blob.upload_from_string(data)

    def get(self, key: str) -> bytes:
        """Retrieve data from key."""
        blob = self.bucket.blob(self._make_key(key))
        return blob.download_as_bytes()

    def exists(self, key: str) -> bool:
        """Check if key exists."""
        blob = self.bucket.blob(self._make_key(key))
        return blob.exists()

    def delete(self, key: str) -> None:
        """Delete data at key."""
        blob = self.bucket.blob(self._make_key(key))
        blob.delete()

    def list(self, prefix: str = "") -> List[str]:
        """List keys with given prefix."""
        full_prefix = self._make_key(prefix)
        blobs = self.client.list_blobs(
            self.bucket_name, prefix=full_prefix
        )

        keys = []
        for blob in blobs:
            key = blob.name
            if self.prefix:
                key = key[len(self.prefix) + 1 :]
            keys.append(key)

        return sorted(keys)


class AzureStorage(StorageBackend):
    """Azure Blob Storage backend."""

    def __init__(self, config: StorageConfig):
        from azure.storage.blob import BlobServiceClient

        self.container = config.bucket
        self.prefix = config.prefix

        # Create client
        if config.azure_connection_string:
            self.service = BlobServiceClient.from_connection_string(
                config.azure_connection_string
            )
        else:
            account_url = f"https://{config.azure_account_name}.blob.core.windows.net"
            self.service = BlobServiceClient(
                account_url=account_url,
                credential=config.azure_account_key,
            )

        self.container_client = self.service.get_container_client(
            self.container
        )

    def _make_key(self, key: str) -> str:
        """Make full Azure key with prefix."""
        if self.prefix:
            return f"{self.prefix}/{key}"
        return key

    def put(self, key: str, data: bytes) -> None:
        """Store data at key."""
        blob = self.container_client.get_blob_client(self._make_key(key))
        blob.upload_blob(data, overwrite=True)

    def get(self, key: str) -> bytes:
        """Retrieve data from key."""
        blob = self.container_client.get_blob_client(self._make_key(key))
        return blob.download_blob().readall()

    def exists(self, key: str) -> bool:
        """Check if key exists."""
        blob = self.container_client.get_blob_client(self._make_key(key))
        return blob.exists()

    def delete(self, key: str) -> None:
        """Delete data at key."""
        blob = self.container_client.get_blob_client(self._make_key(key))
        blob.delete_blob()

    def list(self, prefix: str = "") -> List[str]:
        """List keys with given prefix."""
        full_prefix = self._make_key(prefix)
        blobs = self.container_client.list_blobs(name_starts_with=full_prefix)

        keys = []
        for blob in blobs:
            key = blob.name
            if self.prefix:
                key = key[len(self.prefix) + 1 :]
            keys.append(key)

        return sorted(keys)


def create_storage(config: StorageConfig) -> StorageBackend:
    """Create storage backend from configuration.
    
    Args:
        config: Storage configuration
        
    Returns:
        Appropriate storage backend instance
    """
    if config.backend == "local":
        return LocalStorage(config.local_path)
    elif config.backend == "s3":
        return S3Storage(config)
    elif config.backend == "gcs":
        return GCSStorage(config)
    elif config.backend == "azure":
        return AzureStorage(config)
    else:
        raise ValueError(f"Unknown storage backend: {config.backend}")
