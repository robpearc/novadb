"""Cropping module for NovaDB.

Provides cropping strategies for training data according to
AlphaFold3 Section 2.7, with composable transform implementations.

Cropping Methods (Table 4):
- ContiguousCropping: Sample contiguous segments across chains (AF-multimer Algorithm 1)
- SpatialCropping: Select tokens nearest to a random reference token
- SpatialInterfaceCropping: Select tokens nearest to interface tokens
"""

from novadb.processing.cropping.transforms import (
    # Configuration
    CropTransformConfig,
    CropConfig,  # Backward compatibility
    # Data structures
    MoleculeInfo,
    CropResult,
    LegacyCropResult,  # Backward compatibility
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

__all__ = [
    # Configuration
    "CropTransformConfig",
    "CropConfig",
    # Data structures
    "MoleculeInfo",
    "CropResult",
    "LegacyCropResult",
    # Enum
    "CroppingStrategy",
    # Base class
    "BaseCropTransform",
    # Preprocessing transforms
    "MoleculeInfoTransform",
    "TokenDistanceMatrixTransform",
    "InterfaceTokenTransform",
    # Cropping transforms
    "ContiguousCropTransform",
    "SpatialCropTransform",
    "CombinedCropTransform",
    "ApplyCropTransform",
    # Legacy compatibility transforms
    "CropTransform",
    "CropToTokenLimitTransform",
    # Backward-compatible cropper classes
    "Cropper",
    "ContiguousCropper",
    "SpatialCropper",
    "SpatialInterfaceCropper",
    # Utility functions
    "compute_distances_fast",
    "identify_molecule_types",
    "get_interface_tokens",
    "check_atom_limit",
    # Pipeline factory
    "create_crop_pipeline",
    "apply_crop_pipeline",
]
