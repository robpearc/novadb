"""Template search and processing module for NovaDB.

Provides template search and processing functionality for structure prediction
according to AlphaFold3 Section 2.4:

- Template search using HMM-based methods
- Template HMM building from UniRef90 MSAs
- Template date filtering for training/inference separation
- Template deduplication by sequence and structure
- Template feature extraction (coordinates, distances, masks)
"""

from novadb.search.templates.template_processing import (
    DateFilterConfig,
    DeduplicationConfig,
    TemplateDateFilter,
    TemplateDeduplicator,
    TemplateFeatureConfig,
    TemplateFeatureExtractor,
    TemplateFeatures,
    TemplateHMMBuilder,
    TemplateHMMConfig,
    TemplateProcessingPipeline,
    create_template_pipeline,
)
from novadb.search.templates.template_search import (
    TemplateSearchResult,
    TemplateSearcher,
    realign_template,
)
# Import TemplateHit from processing (more complete version)
from novadb.search.templates.template_processing import TemplateHit

__all__ = [
    # Search
    "TemplateSearcher",
    "TemplateSearchResult",
    "TemplateHit",
    "realign_template",
    # HMM Building
    "TemplateHMMBuilder",
    "TemplateHMMConfig",
    # Date Filtering
    "TemplateDateFilter",
    "DateFilterConfig",
    # Deduplication
    "TemplateDeduplicator",
    "DeduplicationConfig",
    # Feature Extraction
    "TemplateFeatureExtractor",
    "TemplateFeatureConfig",
    "TemplateFeatures",
    # Pipeline
    "TemplateProcessingPipeline",
    "create_template_pipeline",
]
