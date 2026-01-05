"""
Utility functions and classes for NovaDB.

This module provides common utilities for:
- Timing and profiling
- Memory tracking
- File operations
- Sequence processing
- Retry logic with exponential backoff
- Logging configuration
"""

from __future__ import annotations

import functools
import hashlib
import logging
import os
import random
import re
import shutil
import sys
import tempfile
import time
from collections.abc import Callable, Generator, Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


# =============================================================================
# Timing Utilities
# =============================================================================


class Timer:
    """Context manager for timing code blocks with optional logging.
    
    Examples
    --------
    >>> with Timer("data processing"):
    ...     process_data()
    [Timer] data processing: 1.234s
    
    >>> with Timer("silent operation", log=False) as t:
    ...     do_work()
    >>> print(f"Elapsed: {t.elapsed:.2f}s")
    
    >>> timer = Timer()
    >>> timer.start()
    >>> do_work()
    >>> timer.stop()
    >>> print(timer.elapsed)
    """
    
    def __init__(
        self,
        name: str = "Timer",
        log: bool = True,
        log_level: int = logging.INFO,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize the timer.
        
        Parameters
        ----------
        name : str
            Name to display in log messages.
        log : bool
            Whether to log timing information.
        log_level : int
            Logging level to use.
        logger : logging.Logger, optional
            Logger to use. If None, uses module logger.
        """
        self.name = name
        self.log = log
        self.log_level = log_level
        self._logger = logger or globals()["logger"]
        self._start_time: float | None = None
        self._end_time: float | None = None
        self._elapsed: float = 0.0
    
    @property
    def elapsed(self) -> float:
        """Return elapsed time in seconds."""
        if self._start_time is None:
            return self._elapsed
        if self._end_time is None:
            return time.perf_counter() - self._start_time
        return self._elapsed
    
    def start(self) -> Timer:
        """Start the timer."""
        self._start_time = time.perf_counter()
        self._end_time = None
        return self
    
    def stop(self) -> float:
        """Stop the timer and return elapsed time."""
        if self._start_time is None:
            raise RuntimeError("Timer was not started")
        self._end_time = time.perf_counter()
        self._elapsed = self._end_time - self._start_time
        return self._elapsed
    
    def reset(self) -> None:
        """Reset the timer."""
        self._start_time = None
        self._end_time = None
        self._elapsed = 0.0
    
    def __enter__(self) -> Timer:
        """Enter the context manager."""
        self.start()
        return self
    
    def __exit__(self, *args: Any) -> None:
        """Exit the context manager."""
        self.stop()
        if self.log:
            self._logger.log(
                self.log_level,
                "[Timer] %s: %.3fs",
                self.name,
                self._elapsed,
            )
    
    def __repr__(self) -> str:
        return f"Timer(name={self.name!r}, elapsed={self.elapsed:.3f}s)"


def timed(
    name: str | None = None,
    log_level: int = logging.DEBUG,
    logger: logging.Logger | None = None,
) -> Callable[[F], F]:
    """Decorator for timing function execution.
    
    Parameters
    ----------
    name : str, optional
        Name to use in log messages. Defaults to function name.
    log_level : int
        Logging level to use.
    logger : logging.Logger, optional
        Logger to use. If None, uses module logger.
    
    Returns
    -------
    Callable
        Decorated function.
    
    Examples
    --------
    >>> @timed()
    ... def slow_function():
    ...     time.sleep(1)
    
    >>> @timed("custom_name", log_level=logging.INFO)
    ... def another_function():
    ...     pass
    """
    def decorator(func: F) -> F:
        func_name = name or func.__qualname__
        log = logger or globals()["logger"]
        
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                log.log(
                    log_level,
                    "[Timed] %s: %.3fs",
                    func_name,
                    elapsed,
                )
        
        return wrapper  # type: ignore[return-value]
    
    return decorator


# =============================================================================
# Memory Utilities
# =============================================================================


def get_memory_usage() -> dict[str, float]:
    """Get current process memory usage.
    
    Returns
    -------
    dict[str, float]
        Dictionary with memory metrics in MB:
        - rss: Resident Set Size (physical memory)
        - vms: Virtual Memory Size
        - percent: Percentage of system memory used
    
    Examples
    --------
    >>> mem = get_memory_usage()
    >>> print(f"Memory: {mem['rss']:.1f} MB")
    """
    try:
        import psutil
        
        process = psutil.Process()
        mem_info = process.memory_info()
        
        return {
            "rss": mem_info.rss / (1024 * 1024),  # MB
            "vms": mem_info.vms / (1024 * 1024),  # MB
            "percent": process.memory_percent(),
        }
    except ImportError:
        # Fallback if psutil is not available
        import resource
        
        usage = resource.getrusage(resource.RUSAGE_SELF)
        # maxrss is in KB on Linux, bytes on macOS
        rss_kb = usage.ru_maxrss
        if sys.platform == "darwin":
            rss_kb = rss_kb / 1024  # Convert bytes to KB on macOS
        
        return {
            "rss": rss_kb / 1024,  # MB
            "vms": 0.0,  # Not available without psutil
            "percent": 0.0,  # Not available without psutil
        }


class MemoryTracker:
    """Track memory usage over time.
    
    Examples
    --------
    >>> tracker = MemoryTracker()
    >>> tracker.snapshot("start")
    >>> process_data()
    >>> tracker.snapshot("after_processing")
    >>> print(tracker.report())
    """
    
    def __init__(self) -> None:
        """Initialize the memory tracker."""
        self._snapshots: list[tuple[str, float, dict[str, float]]] = []
        self._start_time = time.perf_counter()
    
    def snapshot(self, label: str = "") -> dict[str, float]:
        """Take a memory snapshot.
        
        Parameters
        ----------
        label : str
            Label for this snapshot.
        
        Returns
        -------
        dict[str, float]
            Current memory usage.
        """
        mem = get_memory_usage()
        elapsed = time.perf_counter() - self._start_time
        self._snapshots.append((label or f"snapshot_{len(self._snapshots)}", elapsed, mem))
        return mem
    
    def reset(self) -> None:
        """Reset all snapshots."""
        self._snapshots.clear()
        self._start_time = time.perf_counter()
    
    @property
    def snapshots(self) -> list[tuple[str, float, dict[str, float]]]:
        """Return all snapshots."""
        return self._snapshots.copy()
    
    @property
    def peak_memory(self) -> float:
        """Return peak RSS memory in MB."""
        if not self._snapshots:
            return 0.0
        return max(s[2]["rss"] for s in self._snapshots)
    
    def report(self) -> str:
        """Generate a memory usage report.
        
        Returns
        -------
        str
            Formatted report of memory snapshots.
        """
        if not self._snapshots:
            return "No snapshots recorded."
        
        lines = ["Memory Usage Report", "=" * 50]
        
        for label, elapsed, mem in self._snapshots:
            lines.append(
                f"[{elapsed:7.2f}s] {label:20s} | "
                f"RSS: {mem['rss']:8.1f} MB | "
                f"VMS: {mem['vms']:8.1f} MB"
            )
        
        lines.append("=" * 50)
        lines.append(f"Peak RSS: {self.peak_memory:.1f} MB")
        
        return "\n".join(lines)


# =============================================================================
# File Utilities
# =============================================================================


@contextmanager
def atomic_write(
    filepath: str | Path,
    mode: str = "w",
    encoding: str | None = "utf-8",
    **kwargs: Any,
) -> Generator[Any, None, None]:
    """Write to a file atomically using a temporary file.
    
    The file is written to a temporary location first, then moved to the
    target location. This ensures the target file is never in a partially
    written state.
    
    Parameters
    ----------
    filepath : str or Path
        Target file path.
    mode : str
        File mode ('w' for text, 'wb' for binary).
    encoding : str, optional
        File encoding for text mode.
    **kwargs
        Additional arguments passed to open().
    
    Yields
    ------
    file object
        File handle for writing.
    
    Examples
    --------
    >>> with atomic_write("data.json") as f:
    ...     json.dump(data, f)
    
    >>> with atomic_write("image.png", mode="wb") as f:
    ...     f.write(image_bytes)
    """
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    
    # Determine if binary mode
    is_binary = "b" in mode
    
    # Create temp file in same directory for atomic rename
    fd, temp_path = tempfile.mkstemp(
        dir=filepath.parent,
        prefix=f".{filepath.name}.",
        suffix=".tmp",
    )
    temp_path = Path(temp_path)
    
    try:
        os.close(fd)
        
        open_kwargs: dict[str, Any] = {"mode": mode, **kwargs}
        if not is_binary and encoding:
            open_kwargs["encoding"] = encoding
        
        with open(temp_path, **open_kwargs) as f:
            yield f
        
        # Atomic move
        shutil.move(str(temp_path), str(filepath))
        
    except Exception:
        # Clean up temp file on error
        if temp_path.exists():
            temp_path.unlink()
        raise


def ensure_dir(path: str | Path, mode: int = 0o755) -> Path:
    """Ensure a directory exists, creating it if necessary.
    
    Parameters
    ----------
    path : str or Path
        Directory path to create.
    mode : int
        Directory permissions (default: 0o755).
    
    Returns
    -------
    Path
        The directory path.
    
    Examples
    --------
    >>> ensure_dir("/path/to/data")
    PosixPath('/path/to/data')
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True, mode=mode)
    return path


def compute_checksum(
    filepath: str | Path,
    algorithm: str = "md5",
    chunk_size: int = 8192,
) -> str:
    """Compute the checksum of a file.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the file.
    algorithm : str
        Hash algorithm ('md5', 'sha256', 'sha1', etc.).
    chunk_size : int
        Size of chunks to read at a time.
    
    Returns
    -------
    str
        Hexadecimal checksum string.
    
    Examples
    --------
    >>> checksum = compute_checksum("data.tar.gz", algorithm="sha256")
    >>> print(checksum)
    'a1b2c3...'
    """
    filepath = Path(filepath)
    
    hasher = hashlib.new(algorithm)
    
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    
    return hasher.hexdigest()


def safe_filename(
    filename: str,
    replacement: str = "_",
    max_length: int = 255,
) -> str:
    """Sanitize a filename to be safe for all filesystems.
    
    Parameters
    ----------
    filename : str
        Original filename.
    replacement : str
        Character to replace invalid characters with.
    max_length : int
        Maximum length of the resulting filename.
    
    Returns
    -------
    str
        Sanitized filename.
    
    Examples
    --------
    >>> safe_filename("my file: version <1>.txt")
    'my_file__version__1_.txt'
    
    >>> safe_filename("../../../etc/passwd")
    '______etc_passwd'
    """
    # Characters not allowed in filenames on various systems
    invalid_chars = r'[<>:"/\\|?*\x00-\x1f]'
    
    # Replace invalid characters
    sanitized = re.sub(invalid_chars, replacement, filename)
    
    # Remove leading/trailing dots and spaces
    sanitized = sanitized.strip(". ")
    
    # Collapse multiple replacements
    if replacement:
        sanitized = re.sub(f"{re.escape(replacement)}+", replacement, sanitized)
    
    # Truncate if necessary, preserving extension
    if len(sanitized) > max_length:
        # Try to preserve extension
        if "." in sanitized:
            name, ext = sanitized.rsplit(".", 1)
            ext = "." + ext
            max_name_length = max_length - len(ext)
            if max_name_length > 0:
                sanitized = name[:max_name_length] + ext
            else:
                sanitized = sanitized[:max_length]
        else:
            sanitized = sanitized[:max_length]
    
    # Fallback if completely empty
    if not sanitized:
        sanitized = "unnamed"
    
    return sanitized


# =============================================================================
# Sequence Utilities
# =============================================================================


def chunk_iterable(
    iterable: Iterable[T],
    chunk_size: int,
) -> Generator[list[T], None, None]:
    """Split an iterable into chunks of a specified size.
    
    Parameters
    ----------
    iterable : Iterable[T]
        Input iterable to chunk.
    chunk_size : int
        Size of each chunk.
    
    Yields
    ------
    list[T]
        Chunks of the input iterable.
    
    Examples
    --------
    >>> list(chunk_iterable(range(10), 3))
    [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    
    >>> for batch in chunk_iterable(large_dataset, 100):
    ...     process_batch(batch)
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    
    iterator = iter(iterable)
    while True:
        chunk = []
        try:
            for _ in range(chunk_size):
                chunk.append(next(iterator))
        except StopIteration:
            if chunk:
                yield chunk
            return
        yield chunk


def flatten(
    iterable: Iterable[Any],
    depth: int = -1,
) -> Generator[Any, None, None]:
    """Flatten nested iterables to a specified depth.
    
    Parameters
    ----------
    iterable : Iterable
        Nested iterable to flatten.
    depth : int
        Maximum depth to flatten. -1 means unlimited depth.
        0 means no flattening (yields items as-is).
    
    Yields
    ------
    Any
        Flattened items.
    
    Examples
    --------
    >>> list(flatten([[1, 2], [3, [4, 5]]]))
    [1, 2, 3, 4, 5]
    
    >>> list(flatten([[1, 2], [3, [4, 5]]], depth=1))
    [1, 2, 3, [4, 5]]
    
    >>> list(flatten("abc"))  # Strings are not flattened
    ['abc']
    """
    for item in iterable:
        # Don't flatten strings or bytes
        if isinstance(item, (str, bytes)):
            yield item
        elif depth != 0 and isinstance(item, Iterable):
            yield from flatten(item, depth=depth - 1 if depth > 0 else -1)
        else:
            yield item


def unique_ordered(iterable: Iterable[T]) -> Iterator[T]:
    """Remove duplicates from an iterable while preserving order.
    
    Parameters
    ----------
    iterable : Iterable[T]
        Input iterable.
    
    Yields
    ------
    T
        Unique items in order of first appearance.
    
    Examples
    --------
    >>> list(unique_ordered([1, 2, 1, 3, 2, 4]))
    [1, 2, 3, 4]
    
    >>> list(unique_ordered("abracadabra"))
    ['a', 'b', 'r', 'c', 'd']
    """
    seen: set[T] = set()
    for item in iterable:
        if item not in seen:
            seen.add(item)
            yield item


# =============================================================================
# Retry Logic
# =============================================================================


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    max_delay: float = 60.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
    jitter: bool = True,
    on_retry: Callable[[Exception, int, float], None] | None = None,
) -> Callable[[F], F]:
    """Decorator for retrying functions with exponential backoff.
    
    Parameters
    ----------
    max_attempts : int
        Maximum number of attempts (including the initial call).
    delay : float
        Initial delay between retries in seconds.
    backoff : float
        Multiplier for delay after each retry.
    max_delay : float
        Maximum delay between retries.
    exceptions : tuple[type[Exception], ...]
        Exception types to catch and retry on.
    jitter : bool
        Add random jitter to delays to avoid thundering herd.
    on_retry : Callable, optional
        Callback function called before each retry with
        (exception, attempt_number, next_delay).
    
    Returns
    -------
    Callable
        Decorated function.
    
    Examples
    --------
    >>> @retry(max_attempts=3, delay=1.0, exceptions=(ConnectionError,))
    ... def fetch_data(url):
    ...     return requests.get(url)
    
    >>> @retry(max_attempts=5, backoff=2.0, on_retry=lambda e, n, d: print(f"Retry {n}"))
    ... def unreliable_operation():
    ...     pass
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            current_delay = delay
            last_exception: Exception | None = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_attempts:
                        # No more retries
                        raise
                    
                    # Calculate next delay with optional jitter
                    wait_time = min(current_delay, max_delay)
                    if jitter:
                        wait_time *= (0.5 + random.random())
                    
                    # Call retry callback if provided
                    if on_retry:
                        on_retry(e, attempt, wait_time)
                    else:
                        logger.warning(
                            "Retry %d/%d for %s after %.2fs: %s",
                            attempt,
                            max_attempts,
                            func.__qualname__,
                            wait_time,
                            str(e),
                        )
                    
                    time.sleep(wait_time)
                    current_delay *= backoff
            
            # Should never reach here, but just in case
            if last_exception:
                raise last_exception
            raise RuntimeError("Unexpected retry loop exit")
        
        return wrapper  # type: ignore[return-value]
    
    return decorator


# =============================================================================
# Logging
# =============================================================================


def setup_logging(
    level: int | str = logging.INFO,
    format_string: str | None = None,
    log_file: str | Path | None = None,
    log_file_level: int | str | None = None,
    name: str | None = None,
    propagate: bool = True,
) -> logging.Logger:
    """Configure logging with console and optional file output.
    
    Parameters
    ----------
    level : int or str
        Logging level for console output.
    format_string : str, optional
        Custom format string. If None, uses a default format.
    log_file : str or Path, optional
        Path to log file for file output.
    log_file_level : int or str, optional
        Logging level for file output. Defaults to same as console.
    name : str, optional
        Logger name. If None, configures the root logger.
    propagate : bool
        Whether to propagate messages to parent loggers.
    
    Returns
    -------
    logging.Logger
        Configured logger instance.
    
    Examples
    --------
    >>> # Basic setup
    >>> logger = setup_logging(level=logging.DEBUG)
    
    >>> # With file output
    >>> logger = setup_logging(
    ...     level=logging.INFO,
    ...     log_file="app.log",
    ...     log_file_level=logging.DEBUG,
    ... )
    
    >>> # Named logger
    >>> logger = setup_logging(name="myapp", level="DEBUG")
    """
    # Convert string level to int if needed
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    if isinstance(log_file_level, str):
        log_file_level = getattr(logging, log_file_level.upper())
    elif log_file_level is None:
        log_file_level = level
    
    # Default format
    if format_string is None:
        format_string = (
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        )
    
    # Get or create logger
    log = logging.getLogger(name)
    log.setLevel(min(level, log_file_level) if log_file else level)
    log.propagate = propagate
    
    # Remove existing handlers to avoid duplicates
    log.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    log.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_file = Path(log_file)
        ensure_dir(log_file.parent)
        
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(log_file_level)
        file_handler.setFormatter(formatter)
        log.addHandler(file_handler)
    
    return log


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # Timing
    "Timer",
    "timed",
    # Memory
    "get_memory_usage",
    "MemoryTracker",
    # File
    "atomic_write",
    "ensure_dir",
    "compute_checksum",
    "safe_filename",
    # Sequence
    "chunk_iterable",
    "flatten",
    "unique_ordered",
    # Retry
    "retry",
    # Logging
    "setup_logging",
]
