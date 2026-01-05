"""NovaDB: AlphaFold3-style dataset curation and feature engineering pipeline.

This package provides tools for:
- Parsing and processing mmCIF structure files
- Generating multiple sequence alignments (MSAs)
- Template search and processing
- Feature engineering for structure prediction training
- Dataset curation with clustering and weighting
- Cloud storage integration (S3, GCS, Azure)
"""

from novadb.config import Config
from novadb.pipeline.pipeline import DataPipeline, create_pipeline

__version__ = "0.1.0"
__all__ = ["Config", "DataPipeline", "create_pipeline", "__version__"]
