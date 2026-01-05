"""Cropping module for NovaDB.

Provides cropping strategies for training data according to
AlphaFold3 Section 2.7, with composable transform implementations.

Cropping Methods (Table 4):
- ContiguousCropping: Sample contiguous segments across chains (AF-multimer Algorithm 1)
- SpatialCropping: Select tokens nearest to a random reference token
- SpatialInterfaceCropping: Select tokens nearest to interface tokens
"""

from novadb.processing.cropping.cropping import (
    CroppingStrategy,
    CropConfig,
    CropResult as CropperResult,
    Cropper,
    ContiguousCropper,
    SpatialCropper,
    SpatialInterfaceCropper,
    check_atom_limit,
)

from novadb.processing.cropping.transforms import (
    # Configuration
    CropTransformConfig,
    # Data structures
    MoleculeInfo,
    CropResult,
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
    # Legacy compatibility
    CropTransform,
    CropToTokenLimitTransform,
    # Utility functions
    compute_distances_fast,
    identify_molecule_types,
    get_interface_tokens,
    # Pipeline factory
    create_crop_pipeline,
    apply_crop_pipeline,
)

__all__ = [
    # Legacy cropper classes
    "CroppingStrategy",
    "CropConfig",
    "CropperResult",
    "Cropper",
    "ContiguousCropper",
    "SpatialCropper",
    "SpatialInterfaceCropper",
    "check_atom_limit",
    # Configuration
    "CropTransformConfig",
    # Data structures
    "MoleculeInfo",
    "CropResult",
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
    # Legacy compatibility
    "CropTransform",
    "CropToTokenLimitTransform",
    # Utility functions
    "compute_distances_fast",
    "identify_molecule_types",
    "get_interface_tokens",
    # Pipeline factory
    "create_crop_pipeline",
    "apply_crop_pipeline",
]
