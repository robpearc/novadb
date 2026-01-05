"""Dataset curation module for NovaDB.

Provides filtering, clustering, sampling, and distillation for training data
according to AlphaFold3 Section 2.5.

Components:
- Filtering: Date, resolution, chain composition filtering
- Clustering: Sequence and interface-based clustering
- Sampling: 5-dataset weighted sampling scheme
- Distillation: MGnify, Rfam, JASPAR distillation generators
- Transforms: Per-chain, per-structure, and batch transforms
- Bioassembly: Symmetry operations and assembly expansion
"""

from novadb.processing.curation.bioassembly import (
    SymmetryType,
    SymmetryOperation,
    AssemblyDefinition,
    ChainMapping,
    BioassemblyExpanderConfig,
    BioassemblyExpander,
    analyze_symmetry,
    compute_assembly_center,
    compute_assembly_radius,
    parse_operator_expression,
    get_assembly_statistics,
)
from novadb.processing.curation.filtering import (
    FilterResult,
    StructureFilter,
    ClusterFilter,
    InterfaceFilter,
)
from novadb.processing.curation.sampling import (
    DatasetEntry,
    SamplingConfig,
    DatasetSampler,
    create_distillation_entry,
)
from novadb.processing.curation.clustering import (
    ClusterResult,
    SequenceClusterer,
    IdentityClusterer,
    compute_sequence_identity,
    extract_sequences_from_structures,
    write_fasta,
)
from novadb.processing.curation.transforms import (
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
    CurationFilterPipeline,
    # Per-sequence transforms
    ClusterTransform,
    # Batch transforms
    InverseClusterWeightTransform,
    DatasetWeightTransform,
    SamplerTransform,
    BatchFilterTransform,
    CurationPipeline,
    # Utility functions
    filter_structure,
    filter_structures_batch,
)
from novadb.processing.curation.distillation import (
    DistillationSource,
    DistillationSample,
    DistillationConfig,
    BaseDistillationGenerator,
    MGnifyDistillationGenerator,
    RfamDistillationGenerator,
    JASPARDistillationGenerator,
    PDBDisorderedDistillationGenerator,
    ShortProteinDistillationGenerator,
    DistillationPipeline,
)
from novadb.processing.curation.interface_clustering import (
    InterfaceType,
    Interface,
    InterfaceCluster,
    InterfaceClusteringConfig,
    InterfaceDetector,
    InterfaceClusterer,
    LigandClusterer,
    ChainTypeClassifier,
    compute_interface_similarity,
    get_structure_interface_signature,
)

__all__ = [
    # Filtering
    "FilterResult",
    "StructureFilter",
    "ClusterFilter",
    "InterfaceFilter",
    # Sampling
    "DatasetEntry",
    "SamplingConfig",
    "DatasetSampler",
    "create_distillation_entry",
    # Clustering
    "ClusterResult",
    "SequenceClusterer",
    "IdentityClusterer",
    "compute_sequence_identity",
    "extract_sequences_from_structures",
    "write_fasta",
    # Config
    "CurationConfig",
    # Base transforms
    "BaseTransform",
    "ChainTransform",
    "StructureTransform",
    "BatchTransform",
    # Per-chain transforms
    "ChainLengthFilterTransform",
    "ChainSequenceExtractTransform",
    "ChainTypeFilterTransform",
    # Per-structure transforms
    "DateFilterTransform",
    "ResolutionFilterTransform",
    "ChainCompositionFilterTransform",
    "ExclusionListFilterTransform",
    "CurationFilterPipeline",
    # Per-sequence transforms
    "ClusterTransform",
    # Batch transforms
    "InverseClusterWeightTransform",
    "DatasetWeightTransform",
    "SamplerTransform",
    "BatchFilterTransform",
    "CurationPipeline",
    # Utility functions
    "filter_structure",
    "filter_structures_batch",
    # Distillation
    "DistillationSource",
    "DistillationSample",
    "DistillationConfig",
    "BaseDistillationGenerator",
    "MGnifyDistillationGenerator",
    "RfamDistillationGenerator",
    "JASPARDistillationGenerator",
    "PDBDisorderedDistillationGenerator",
    "ShortProteinDistillationGenerator",
    "DistillationPipeline",
    # Interface clustering
    "InterfaceType",
    "Interface",
    "InterfaceCluster",
    "InterfaceClusteringConfig",
    "InterfaceDetector",
    "InterfaceClusterer",
    "LigandClusterer",
    "ChainTypeClassifier",
    "compute_interface_similarity",
    "get_structure_interface_signature",
    # Bioassembly handling
    "SymmetryType",
    "SymmetryOperation",
    "AssemblyDefinition",
    "ChainMapping",
    "BioassemblyExpanderConfig",
    "BioassemblyExpander",
    "analyze_symmetry",
    "compute_assembly_center",
    "compute_assembly_radius",
    "parse_operator_expression",
    "get_assembly_statistics",
]
