"""MSA search and processing modules.

This module provides MSA construction and processing following
AlphaFold3 Section 2.2-2.3:

- MSA data structures (MSA, MSASequence)
- Search runners (JackHMMER, HHblits, nhmmer)
- MSA processing (deduplication, block deletion removal)
- Species-based pairing for multi-chain complexes
- BFD search integration
- UniProt-to-UniRef90 mapping
- RNA MSA realignment
"""

from novadb.search.msa.hhblits import HHblitsRunner
from novadb.search.msa.jackhmmer import JackhmmerRunner
from novadb.search.msa.msa import MSA, MSASequence
from novadb.search.msa.msa_processor import MSAProcessor, PairedMSA
from novadb.search.msa.nhmmer import NhmmerRunner
from novadb.search.msa.processing import (
    BFDSearcher,
    BlockDeletionConfig,
    BlockDeletionRemover,
    DeduplicationConfig,
    MSADeduplicator,
    MSAProcessingPipeline,
    RNARealignmentConfig,
    RNARealigner,
    SpeciesPairer,
    SpeciesPairingConfig,
    UniRef90Mapper,
    UniRef90MappingConfig,
    create_processing_pipeline,
)

__all__ = [
    # Data structures
    "MSA",
    "MSASequence",
    "PairedMSA",
    # Search runners
    "HHblitsRunner",
    "JackhmmerRunner",
    "NhmmerRunner",
    # Processors
    "MSAProcessor",
    "MSADeduplicator",
    "BlockDeletionRemover",
    "SpeciesPairer",
    "UniRef90Mapper",
    "RNARealigner",
    "BFDSearcher",
    "MSAProcessingPipeline",
    # Configs
    "DeduplicationConfig",
    "BlockDeletionConfig",
    "SpeciesPairingConfig",
    "UniRef90MappingConfig",
    "RNARealignmentConfig",
    # Factory
    "create_processing_pipeline",
]
