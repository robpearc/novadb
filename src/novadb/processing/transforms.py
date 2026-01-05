"""Transform re-exports for backwards compatibility.

This module provides backwards-compatible imports from the new
curation.transforms location.
"""

from novadb.processing.curation.transforms import (
    # Core types
    CurationConfig,
    # Base classes
    BaseTransform,
    ChainTransform,
    StructureTransform,
    BatchTransform,
    # Per-chain transforms
    ChainLengthFilterTransform,
    ChainSequenceExtractTransform,
    ChainTypeFilterTransform,
    # Per-structure transforms
    DateFilterTransform,
    ResolutionFilterTransform,
    ChainCompositionFilterTransform,
    ExclusionListFilterTransform,
    LeavingAtomsRemovalTransform,
    CurationFilterPipeline,
    # Per-sequence transforms
    ClusterTransform,
    # Batch transforms
    InverseClusterWeightTransform,
    DatasetWeightTransform,
    SamplerTransform,
    BatchFilterTransform,
    # Pipeline
    CurationPipeline,
    # Utilities
    filter_structure,
    filter_structures_batch,
)

# Aliases for compatibility with __init__.py expected imports
TransformLevel = str  # Placeholder
TransformResult = dict  # Placeholder

# Token-level transforms (not yet implemented)
TokenTransform = BaseTransform
ResidueTransform = BaseTransform
SequenceTransform = BaseTransform
PairTransform = BaseTransform

# Composition helpers
TransformPipeline = CurationFilterPipeline
ParallelTransform = BatchTransform
CachedTransform = BaseTransform

# Utility transforms
LambdaTransform = BaseTransform
FilterTransform = BaseTransform
MapTransform = BaseTransform

# Level lifting (not yet implemented)
LiftToChain = BaseTransform
LiftToStructure = BaseTransform
LiftTokenToStructure = BaseTransform
AggregateChains = BaseTransform
AggregateTokens = BaseTransform

# Decorators (placeholders)
def transform(fn): return fn
def token_transform(fn): return fn
def chain_transform(fn): return fn
def structure_transform(fn): return fn

__all__ = [
    # Core types
    "TransformLevel",
    "TransformResult",
    # Base classes
    "BaseTransform",
    "TokenTransform",
    "ResidueTransform",
    "ChainTransform",
    "StructureTransform",
    "SequenceTransform",
    "PairTransform",
    # Composition
    "TransformPipeline",
    "ParallelTransform",
    "BatchTransform",
    "CachedTransform",
    # Utilities
    "LambdaTransform",
    "FilterTransform",
    "MapTransform",
    "LiftToChain",
    "LiftToStructure",
    "LiftTokenToStructure",
    "AggregateChains",
    "AggregateTokens",
    # Decorators
    "transform",
    "token_transform",
    "chain_transform",
    "structure_transform",
    # Curation transforms
    "CurationConfig",
    "ChainLengthFilterTransform",
    "ChainSequenceExtractTransform",
    "ChainTypeFilterTransform",
    "DateFilterTransform",
    "ResolutionFilterTransform",
    "ChainCompositionFilterTransform",
    "ExclusionListFilterTransform",
    "LeavingAtomsRemovalTransform",
    "CurationFilterPipeline",
    "ClusterTransform",
    "InverseClusterWeightTransform",
    "DatasetWeightTransform",
    "SamplerTransform",
    "BatchFilterTransform",
    "CurationPipeline",
    "filter_structure",
    "filter_structures_batch",
]
