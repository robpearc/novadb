# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NovaDB is a Python framework that replicates the AlphaFold3 dataset curation, processing, and feature engineering pipeline for training biomolecular structure prediction models. It processes PDB structures through parsing, MSA generation, template search, tokenization, feature extraction, and cropping.

## Common Commands

```bash
# Install in development mode
pip install -e ".[all]"

# Run all tests
pytest

# Run tests with coverage
pytest --cov=novadb --cov-report=term-missing

# Run a single test file
pytest tests/test_pipeline.py

# Run a specific test class or function
pytest tests/test_pipeline.py::TestMMCIFParser
pytest tests/test_pipeline.py::TestMMCIFParser::test_parse_basic_structure

# Skip slow/integration tests
pytest -m "not slow and not integration"

# Lint with ruff
ruff check src/

# Format with black
black src/ tests/

# Type check
mypy src/novadb/
```

## Architecture

The pipeline processes structures through these stages (see `src/novadb/pipeline/pipeline.py`):

1. **Parse** (`data/parsers/mmcif_parser.py`) - mmCIF files to `Structure` objects
2. **Filter** (`processing/curation/filtering.py`) - Resolution, chain count, clash detection
3. **Tokenize** (`processing/tokenization/tokenizer.py`) - Residues/atoms to tokens (1 token per standard residue, 1 token per atom for ligands)
4. **MSA Search** (`search/msa/`) - Jackhmmer, HHBlits, nhmmer for sequence alignments
5. **Template Search** (`search/templates/`) - HMM-based template retrieval
6. **Feature Extraction** (`processing/features/`) - Token, MSA, template, bond features
7. **Cropping** (`processing/cropping/cropping.py`) - Contiguous, spatial, spatial-interface strategies
8. **Storage** (`storage/backends.py`) - Local, S3, GCS, Azure backends

### Key Data Flow

```
mmCIF → Structure → TokenizedStructure → InputFeatures → Storage
```

### Configuration

All pipeline parameters are defined in `src/novadb/config.py` using Pydantic models. The `Config` class aggregates sub-configs for databases, MSA search (Jackhmmer/HHBlits/nhmmer), templates, filtering, clustering, cropping, and storage. Configs can be loaded from YAML via `Config.from_yaml()`.

### Entry Points

- `DataPipeline` class in `pipeline/pipeline.py` orchestrates the full workflow
- CLI commands defined in `src/novadb/cli.py` (novadb, novadb-pipeline, novadb-download)

## Test Markers

Tests use pytest markers defined in `pytest.ini`:
- `@pytest.mark.slow` - Long-running tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.requires_external` - Tests needing HMMER tools

## External Dependencies

MSA/template search requires HMMER tools (jackhmmer, hmmbuild, hmmsearch, nhmmer) and optionally HHBlits. These are system dependencies, not Python packages.
