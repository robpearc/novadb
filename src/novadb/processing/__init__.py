"""Processing modules for tokenization, features, cropping, and curation.

Provides both legacy class-based APIs and optimized transform-based APIs
organized by processing level:

Transform Hierarchy:
- TokenTransform: Per-token operations (residue encoding, position)
- ResidueTransform: Per-residue with atoms (coordinates, features)
- ChainTransform: Per-chain operations (sequence features, cropping)
- StructureTransform: Per-structure operations (tokenization, full features)
- SequenceTransform: Per-sequence operations (MSA processing)
- PairTransform: Pairwise relationships (distances, contacts)

Lifting Utilities:
- LiftToChain: Lift token transforms to chain level
- LiftToStructure: Lift chain transforms to structure level
- AggregateTokens/Chains: Aggregate results upward
"""

# Base transforms
from novadb.processing.transforms import (
    # Core types
    TransformLevel,
    TransformResult,
    # Base class
    BaseTransform,
    # Level-specific transforms
    TokenTransform,
    ResidueTransform,
    ChainTransform,
    StructureTransform,
    SequenceTransform,
    PairTransform,
    # Composition
    TransformPipeline,
    # Parallelization
    ParallelTransform,
    BatchTransform,
    # Caching
    CachedTransform,
    # Utilities
    LambdaTransform,
    FilterTransform,
    MapTransform,
    # Level lifting
    LiftToChain,
    LiftToStructure,
    LiftTokenToStructure,
    AggregateChains,
    AggregateTokens,
    # Decorators
    transform,
    token_transform,
    chain_transform,
    structure_transform,
)

# Tokenization
from novadb.processing.tokenization import (
    Token,
    TokenType,
    TokenizedStructure,
    Tokenizer,
)

# Features
from novadb.processing.features import (
    FeatureExtractor,
    InputFeatures,
    FeatureConfig,
    FeatureExtractionTransform,
    # Per-token
    ResidueTypeEncodeTransform,
    TokenTypeEncodeTransform,
    SingleTokenFeatureTransform,
    # Per-residue
    ResidueAtomFeatureTransform,
    PseudoBetaTransform,
    # Per-chain
    ChainTokenFeatureTransform,
    ChainAtomFeatureTransform,
    # Pair
    PairFeatureTransform,
    RelativePositionTransform,
    DistogramTransform,
    # Convenience
    extract_features_fast,
)

# Cropping
from novadb.processing.cropping import (
    # Configuration
    CropTransformConfig,
    CropConfig,
    # Data structures
    MoleculeInfo,
    CropResult,
    LegacyCropResult,
    # Enum
    CroppingStrategy,
    # Base class
    BaseCropTransform,
    # Preprocessing transforms
    MoleculeInfoTransform,
    TokenDistanceMatrixTransform,
    InterfaceTokenTransform,
    # Cropping transforms
    ContiguousCropTransform,
    SpatialCropTransform,
    CombinedCropTransform,
    ApplyCropTransform,
    # Legacy compatibility transforms
    CropTransform,
    CropToTokenLimitTransform,
    # Backward-compatible cropper classes
    Cropper,
    ContiguousCropper,
    SpatialCropper,
    SpatialInterfaceCropper,
    # Utility functions
    compute_distances_fast,
    identify_molecule_types,
    get_interface_tokens,
    check_atom_limit,
    # Pipeline factory
    create_crop_pipeline,
    apply_crop_pipeline,
)

# Curation
from novadb.processing.curation import (
    FilterResult,
    StructureFilter,
    ClusterFilter,
    DatasetEntry,
    DatasetSampler,
    CurationConfig,
    # Per-chain
    ChainLengthFilterTransform,
    ChainSequenceExtractTransform,
    ChainTypeFilterTransform,
    # Per-structure
    DateFilterTransform,
    ResolutionFilterTransform,
    ChainCompositionFilterTransform,
    ExclusionListFilterTransform,
    CurationFilterPipeline,
    # Per-sequence
    ClusterTransform,
    # Batch
    InverseClusterWeightTransform,
    DatasetWeightTransform,
    SamplerTransform,
    CurationPipeline,
    # Bioassembly handling
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

# Conformer generation
from novadb.processing.conformer import (
    # Configs
    ConformerGenerationConfig,
    CCDConfig,
    AlignmentConfig,
    # Data classes
    Conformer,
    ConformerEnsemble,
    # Generators
    RDKitConformerGenerator,
    CCDConformerExtractor,
    ConformerAligner,
    # Pipeline
    ReferenceConformerPipeline,
    create_conformer_pipeline,
    # Utilities
    kabsch_align,
    compute_kabsch_rotation,
    compute_rmsd,
)

# Validation
from novadb.processing.validation import (
    # Enums
    ValidationSeverity,
    # Data classes
    ValidationIssue,
    ValidationResult,
    StructureValidationConfig,
    # Validators
    StructureValidator,
    # Convenience functions
    validate_structure,
    is_structure_valid,
    create_strict_config,
    create_lenient_config,
    # Constants
    PROTEIN_BACKBONE_ATOMS,
    RNA_BACKBONE_ATOMS,
    DNA_BACKBONE_ATOMS,
    COVALENT_RADII,
)


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
    # Tokenization
    "Token",
    "TokenType",
    "TokenizedStructure",
    "Tokenizer",
    # Features
    "FeatureExtractor",
    "InputFeatures",
    "FeatureConfig",
    "FeatureExtractionTransform",
    "ResidueTypeEncodeTransform",
    "TokenTypeEncodeTransform",
    "SingleTokenFeatureTransform",
    "ResidueAtomFeatureTransform",
    "PseudoBetaTransform",
    "ChainTokenFeatureTransform",
    "ChainAtomFeatureTransform",
    "PairFeatureTransform",
    "RelativePositionTransform",
    "DistogramTransform",
    "extract_features_fast",
    # Cropping
    "CropTransformConfig",
    "CropConfig",
    "MoleculeInfo",
    "CropResult",
    "LegacyCropResult",
    "CroppingStrategy",
    "BaseCropTransform",
    "MoleculeInfoTransform",
    "TokenDistanceMatrixTransform",
    "InterfaceTokenTransform",
    "ContiguousCropTransform",
    "SpatialCropTransform",
    "CombinedCropTransform",
    "ApplyCropTransform",
    "CropTransform",
    "CropToTokenLimitTransform",
    "Cropper",
    "ContiguousCropper",
    "SpatialCropper",
    "SpatialInterfaceCropper",
    "compute_distances_fast",
    "identify_molecule_types",
    "get_interface_tokens",
    "check_atom_limit",
    "create_crop_pipeline",
    "apply_crop_pipeline",
    # Curation
    "FilterResult",
    "StructureFilter",
    "ClusterFilter",
    "DatasetEntry",
    "DatasetSampler",
    "CurationConfig",
    "ChainLengthFilterTransform",
    "ChainSequenceExtractTransform",
    "ChainTypeFilterTransform",
    "DateFilterTransform",
    "ResolutionFilterTransform",
    "ChainCompositionFilterTransform",
    "ExclusionListFilterTransform",
    "CurationFilterPipeline",
    "ClusterTransform",
    "InverseClusterWeightTransform",
    "DatasetWeightTransform",
    "SamplerTransform",
    "CurationPipeline",
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
    # Conformer generation
    "ConformerGenerationConfig",
    "CCDConfig",
    "AlignmentConfig",
    "Conformer",
    "ConformerEnsemble",
    "RDKitConformerGenerator",
    "CCDConformerExtractor",
    "ConformerAligner",
    "ReferenceConformerPipeline",
    "create_conformer_pipeline",
    "kabsch_align",
    "compute_kabsch_rotation",
    "compute_rmsd",
    # Validation
    "ValidationSeverity",
    "ValidationIssue",
    "ValidationResult",
    "StructureValidationConfig",
    "StructureValidator",
    "validate_structure",
    "is_structure_valid",
    "create_strict_config",
    "create_lenient_config",
    "PROTEIN_BACKBONE_ATOMS",
    "RNA_BACKBONE_ATOMS",
    "DNA_BACKBONE_ATOMS",
    "COVALENT_RADII",
]
