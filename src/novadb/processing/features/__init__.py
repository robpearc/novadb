"""Feature engineering module for NovaDB.

Provides feature extraction for biomolecular structures according to
AlphaFold3 Table 5 feature specifications, with hierarchical transforms.

Feature Categories (AF3 Table 5):
- Reference Conformer: ref_pos, ref_mask, ref_element, ref_charge, ref_atom_name_chars, ref_space_uid
- Atom-Token Mapping: atom_to_token
- MSA Features: msa_profile, deletion_mean, has_deletion, deletion_value
- Template Features: template_distogram, template_backbone_frame, template_unit_vector
- Bond Features: token_bonds, polymer_ligand_bonds

Transform Hierarchy:
- Per-Token: ResidueTypeEncodeTransform, TokenTypeEncodeTransform
- Per-Residue: ResidueAtomFeatureTransform, PseudoBetaTransform
- Per-Chain: ChainTokenFeatureTransform, ChainAtomFeatureTransform
- Per-Sequence: SequenceEncodeTransform, MSARowTransform
- Pair: RelativePositionTransform, DistogramTransform, PairFeatureTransform
- Per-Structure: TokenFeatureTransform, AtomFeatureTransform, FeatureExtractionTransform
"""

from novadb.processing.features.features import (
    FeatureExtractor,
    InputFeatures,
    compute_relative_position_encoding,
    compute_token_pair_features,
)

from novadb.processing.features.reference_features import (
    # Configs
    BondFeatureConfig,
    MSAProfileConfig,
    ReferenceConformerConfig,
    # Feature containers
    BondFeatures,
    MSAProfileFeatures,
    ReferenceConformerFeatures,
    TemplateFrameFeatures,
    # Extractors
    BondFeatureExtractor,
    MSAProfileExtractor,
    ReferenceConformerExtractor,
    TemplateFrameExtractor,
    # Pipeline
    FeatureEngineeringPipeline,
    create_feature_pipeline,
)

from novadb.processing.features.transforms import (
    # Base classes
    Transform,
    BaseTransform,
    Pipeline,
    # Configs
    FeatureConfig,
    DistogramConfig,
    MSAConfig,
    TemplateConfig,
    # Per-token transforms
    ResidueTypeEncodeTransform,
    TokenTypeEncodeTransform,
    TokenPositionTransform,
    SingleTokenFeatureTransform,
    # Per-residue transforms
    ResidueAtomFeatureTransform,
    PseudoBetaTransform,
    BackboneCompleteTransform,
    # Per-chain transforms
    ChainTokenFeatureTransform,
    ChainAtomFeatureTransform,
    ChainCenterCoordsTransform,
    # Per-sequence transforms
    SequenceEncodeTransform,
    MSARowTransform,
    # Pair transforms
    RelativePositionTransform,
    SameChainMaskTransform,
    SameEntityMaskTransform,
    DistogramTransform,
    PairFeatureTransform,
    # Per-structure transforms
    TokenFeatureTransform,
    AtomFeatureTransform,
    TemplateFeatureTransform,
    FeatureExtractionTransform,
    # Utility functions
    compute_distogram_fast,
    compute_relative_positions_fast,
    extract_features_fast,
    # Pipeline factory functions
    create_token_pipeline,
    create_atom_pipeline,
    create_msa_pipeline,
    create_template_pipeline,
    create_full_pipeline,
)

__all__ = [
    # Legacy classes
    "FeatureExtractor",
    "InputFeatures",
    "compute_relative_position_encoding",
    "compute_token_pair_features",
    # Reference feature configs
    "ReferenceConformerConfig",
    "MSAProfileConfig",
    "BondFeatureConfig",
    # Reference feature containers
    "ReferenceConformerFeatures",
    "MSAProfileFeatures",
    "TemplateFrameFeatures",
    "BondFeatures",
    # Reference feature extractors
    "ReferenceConformerExtractor",
    "MSAProfileExtractor",
    "TemplateFrameExtractor",
    "BondFeatureExtractor",
    # Feature pipeline (legacy)
    "FeatureEngineeringPipeline",
    "create_feature_pipeline",
    # Base classes
    "Transform",
    "BaseTransform",
    "Pipeline",
    # Configs
    "FeatureConfig",
    "DistogramConfig",
    "MSAConfig",
    "TemplateConfig",
    # Per-token transforms
    "ResidueTypeEncodeTransform",
    "TokenTypeEncodeTransform",
    "TokenPositionTransform",
    "SingleTokenFeatureTransform",
    # Per-residue transforms
    "ResidueAtomFeatureTransform",
    "PseudoBetaTransform",
    "BackboneCompleteTransform",
    # Per-chain transforms
    "ChainTokenFeatureTransform",
    "ChainAtomFeatureTransform",
    "ChainCenterCoordsTransform",
    # Per-sequence transforms
    "SequenceEncodeTransform",
    "MSARowTransform",
    # Pair transforms
    "RelativePositionTransform",
    "SameChainMaskTransform",
    "SameEntityMaskTransform",
    "DistogramTransform",
    "PairFeatureTransform",
    # Per-structure transforms
    "TokenFeatureTransform",
    "AtomFeatureTransform",
    "TemplateFeatureTransform",
    "FeatureExtractionTransform",
    # Utility functions
    "compute_distogram_fast",
    "compute_relative_positions_fast",
    "extract_features_fast",
    # Pipeline factory functions
    "create_token_pipeline",
    "create_atom_pipeline",
    "create_msa_pipeline",
    "create_template_pipeline",
    "create_full_pipeline",
]
