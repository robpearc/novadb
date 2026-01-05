"""Pytest configuration and fixtures for NovaDB tests."""

from __future__ import annotations

import tempfile
from datetime import date
from pathlib import Path
from typing import Dict, Generator, List

import numpy as np
import pytest


# =============================================================================
# Test Data Fixtures
# =============================================================================


@pytest.fixture
def sample_protein_sequence() -> str:
    """A sample protein sequence for testing."""
    return "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH"


@pytest.fixture
def sample_rna_sequence() -> str:
    """A sample RNA sequence for testing."""
    return "GCUAGCUAGCUAGCUAGCUA"


@pytest.fixture
def sample_dna_sequence() -> str:
    """A sample DNA sequence for testing."""
    return "ACGTACGTACGTACGTACGT"


@pytest.fixture
def sample_coordinates() -> np.ndarray:
    """Sample 3D coordinates for testing."""
    return np.array([
        [0.0, 0.0, 0.0],
        [1.5, 0.0, 0.0],
        [2.0, 1.5, 0.0],
        [3.5, 1.5, 0.0],
        [4.0, 0.0, 0.0],
    ], dtype=np.float32)


@pytest.fixture
def sample_msa_sequences() -> List[str]:
    """Sample MSA sequences for testing."""
    return [
        "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH",  # Query
        "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFD---",
        "MVLSPA-KTNVKAAWGKVGAHAGEYGAEAL-RMFLSFPTTKTYFPHFDLSH",
        "MVLSPADKTNVKAAWGKVG----EYGAEALERMFLSFPTTKTYFPHFDLSH",
        "---SPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH",
    ]


# =============================================================================
# Temporary Directory Fixtures
# =============================================================================


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_mmcif_dir(temp_dir: Path) -> Path:
    """Create a temporary directory for mmCIF files."""
    mmcif_dir = temp_dir / "mmcif"
    mmcif_dir.mkdir()
    return mmcif_dir


@pytest.fixture
def temp_output_dir(temp_dir: Path) -> Path:
    """Create a temporary directory for output files."""
    output_dir = temp_dir / "output"
    output_dir.mkdir()
    return output_dir


# =============================================================================
# Configuration Fixtures
# =============================================================================


@pytest.fixture
def minimal_config() -> Dict:
    """Minimal configuration for testing."""
    return {
        "databases": {
            "pdb_mmcif_dir": "/tmp/pdb",
        },
        "num_workers": 1,
        "batch_size": 4,
        "seed": 42,
    }


@pytest.fixture
def test_date() -> date:
    """A fixed date for reproducible testing."""
    return date(2021, 9, 30)


# =============================================================================
# Skip Markers
# =============================================================================


def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "requires_external: marks tests that require external tools"
    )


# =============================================================================
# Helper Fixtures
# =============================================================================


@pytest.fixture
def assert_arrays_equal():
    """Fixture providing array comparison helper."""
    def _assert_arrays_equal(a: np.ndarray, b: np.ndarray, rtol: float = 1e-5):
        np.testing.assert_allclose(a, b, rtol=rtol)
    return _assert_arrays_equal


@pytest.fixture
def random_seed() -> int:
    """Fixed random seed for reproducibility."""
    return 42


@pytest.fixture
def set_random_seed(random_seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(random_seed)
    return random_seed
