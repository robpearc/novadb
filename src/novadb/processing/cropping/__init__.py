"""Cropping module for NovaDB.

Provides cropping strategies for training data according to
AlphaFold3 Section 2.7, with hierarchical transform implementations.

Transform Hierarchy:
- Per-Chain: ChainContiguousCropTransform, ChainCenterCoordsTransform
- Per-Structure: ContiguousCropTransform, SpatialCropTransform, CropTransform
"""

from novadb.processing.cropping.cropping import (
    CroppingStrategy,
    CropConfig,
    CropResult,
    Cropper,
    ContiguousCropper,
    SpatialCropper,
    SpatialInterfaceCropper,
    check_atom_limit,
)

from novadb.processing.cropping.transforms import (
    StructureCache,
    # Per-chain transforms
    ChainContiguousCropTransform,
    ChainCenterCoordsTransform,
    ChainInterfaceDetectTransform,
    # Per-structure transforms
    ContiguousCropTransform,
    SpatialCropTransform,
    SpatialInterfaceCropTransform,
    CropTransform,
    CropToTokenLimitTransform,
    # Utility functions
    compute_distances_fast,
    find_interface_contacts_fast,
)

__all__ = [
    # Legacy classes
    "CroppingStrategy",
    "CropConfig",
    "CropResult",
    "Cropper",
    "ContiguousCropper",
    "SpatialCropper",
    "SpatialInterfaceCropper",
    "check_atom_limit",
    # Caching
    "StructureCache",
    # Per-chain transforms
    "ChainContiguousCropTransform",
    "ChainCenterCoordsTransform",
    "ChainInterfaceDetectTransform",
    # Per-structure transforms
    "ContiguousCropTransform",
    "SpatialCropTransform",
    "SpatialInterfaceCropTransform",
    "CropTransform",
    "CropToTokenLimitTransform",
    # Utility functions
    "compute_distances_fast",
    "find_interface_contacts_fast",
]
