"""Cropping transforms for NovaDB.

Provides composable transforms for structure cropping following AF3 Section 2.7.
Based on ByteDance/Protenix cropping implementation.

Cropping Methods (Table 4):
- ContiguousCropping: Sample contiguous segments across chains (AF-multimer Algorithm 1)
- SpatialCropping: Select tokens nearest to a random reference token
- SpatialInterfaceCropping: Select tokens nearest to interface tokens

Key Features:
- Complete ligand/non-standard residue handling (don't fragment molecules)
- Metal/ion removal option
- Interface token detection (tokens within 15A of another chain)
- ref_space_uid-based molecule boundary tracking
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

try:
    from scipy.spatial import cKDTree
    from scipy.spatial.distance import cdist
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# Type alias
FeatureDict = Dict[str, Any]


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class CropTransformConfig:
    """Configuration for cropping transforms.

    From AF3 Section 2.7 and Table 4.
    """
    max_tokens: int = 384
    max_atoms: int = 4608
    interface_distance: float = 15.0  # Distance threshold for interface detection

    # Method weights: [contiguous, spatial, spatial_interface]
    method_weights: Tuple[float, float, float] = (0.2, 0.4, 0.4)

    # Ligand handling
    crop_complete_ligand: bool = True  # Don't fragment ligands
    drop_last_incomplete: bool = False  # Drop incomplete molecules at boundary
    remove_metals: bool = False  # Remove single-atom metal ions


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class MoleculeInfo:
    """Information about molecule types and boundaries.

    Tracks ref_space_uid boundaries for complete molecule cropping.
    """
    is_metal: np.ndarray  # (N_tokens,) bool mask for metal ions
    first_indices: np.ndarray  # (N_tokens,) first token index for each uid
    last_indices: np.ndarray  # (N_tokens,) last token index for each uid
    atom_counts: np.ndarray  # (N_tokens,) atom count per uid


@dataclass
class CropResult:
    """Result of a cropping operation."""
    selected_indices: np.ndarray  # Selected token indices
    reference_token_idx: int = -1  # Reference token used (for spatial crops)
    method: str = ""  # Cropping method used


# =============================================================================
# Utility Functions
# =============================================================================


def compute_distances_fast(
    coords1: np.ndarray,
    coords2: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute pairwise distances efficiently.

    Args:
        coords1: First set of coordinates (N, 3)
        coords2: Optional second set (M, 3). If None, self-distances.

    Returns:
        Distance matrix (N, M) or (N, N)
    """
    if coords2 is None:
        coords2 = coords1

    if HAS_SCIPY:
        return cdist(coords1, coords2, metric='euclidean')
    else:
        # Fallback to broadcasting
        diff = coords1[:, np.newaxis, :] - coords2[np.newaxis, :, :]
        return np.sqrt(np.sum(diff ** 2, axis=-1))


def identify_molecule_types(
    ref_space_uid: np.ndarray,
    atom_counts: np.ndarray,
    chain_id: np.ndarray,
    chain_lengths: np.ndarray,
) -> MoleculeInfo:
    """Identify molecule types and track uid boundaries.

    Identifies metals (single atom in single-residue chain) and tracks
    first/last indices for each unique ref_space_uid.

    Args:
        ref_space_uid: Unique ID for each token's residue (N_tokens,)
        atom_counts: Number of atoms per token (N_tokens,)
        chain_id: Chain ID for each token (N_tokens,)
        chain_lengths: Length of each chain (N_chains,)

    Returns:
        MoleculeInfo with masks and indices
    """
    n_tokens = len(ref_space_uid)
    is_metal = np.zeros(n_tokens, dtype=bool)
    first_indices = np.zeros(n_tokens, dtype=np.int64)
    last_indices = np.zeros(n_tokens, dtype=np.int64)

    # Get unique uids and their counts
    unique_uids, inverse, counts = np.unique(
        ref_space_uid, return_inverse=True, return_counts=True
    )

    for i, (uid, count) in enumerate(zip(unique_uids, counts)):
        mask = ref_space_uid == uid
        indices = np.where(mask)[0]
        first_idx = indices[0]
        last_idx = indices[-1]

        first_indices[mask] = first_idx
        last_indices[mask] = last_idx

        # Metal: single occurrence, single atom, in single-residue chain
        if count == 1:
            token_chain = chain_id[mask][0]
            if token_chain < len(chain_lengths):
                if chain_lengths[int(token_chain)] == 1:
                    if atom_counts[mask][0] == 1:
                        is_metal[mask] = True

    return MoleculeInfo(
        is_metal=is_metal,
        first_indices=first_indices,
        last_indices=last_indices,
        atom_counts=atom_counts,
    )


def get_interface_tokens(
    chain_id: np.ndarray,
    reference_chain_ids: np.ndarray,
    token_distances: np.ndarray,
    distance_mask: np.ndarray,
    interface_distance: float = 15.0,
) -> np.ndarray:
    """Find tokens in contact with other chains.

    Args:
        chain_id: Chain ID for each token (N_tokens,)
        reference_chain_ids: Chain IDs to find interfaces for
        token_distances: Distance matrix (N_ref_tokens, N_all_tokens)
        distance_mask: Valid distance mask (N_ref_tokens, N_all_tokens)
        interface_distance: Distance threshold for interface

    Returns:
        Indices of interface tokens within reference chains
    """
    n_tokens = len(chain_id)
    n_ref = token_distances.shape[0]

    # Get reference token mask
    ref_mask = np.isin(chain_id, reference_chain_ids)
    ref_indices = np.where(ref_mask)[0]

    if len(ref_indices) == 0:
        return np.array([], dtype=np.int64)

    # Distance threshold mask
    close_mask = token_distances < interface_distance

    # Different chain mask (for each ref token, check if target is different chain)
    ref_chain_ids = chain_id[ref_indices]
    diff_chain_mask = chain_id[np.newaxis, :] != ref_chain_ids[:, np.newaxis]

    # Combine masks: close + different chain + valid distance
    interface_mask = close_mask & diff_chain_mask & distance_mask

    # Tokens with any interface contact
    has_interface = interface_mask.any(axis=1)
    interface_indices = ref_indices[has_interface]

    return interface_indices


# =============================================================================
# Base Transform
# =============================================================================


class BaseCropTransform:
    """Base class for cropping transforms."""

    def __init__(self, config: Optional[CropTransformConfig] = None):
        self.config = config or CropTransformConfig()

    def __call__(self, data: FeatureDict) -> FeatureDict:
        raise NotImplementedError

    def _get_token_coords(self, data: FeatureDict) -> np.ndarray:
        """Get representative coordinates for each token."""
        # Prefer centre_atom_coords if available
        if "centre_atom_coords" in data:
            return data["centre_atom_coords"]

        # Fall back to pseudo_beta
        if "pseudo_beta" in data:
            return data["pseudo_beta"]

        # Compute from atom positions
        atom_ref_pos = data.get("atom_ref_pos", np.zeros((0, 3)))
        atom_to_token = data.get("atom_to_token", np.array([]))
        n_tokens = data.get("num_tokens", 0)

        if len(atom_ref_pos) == 0 or n_tokens == 0:
            return np.zeros((n_tokens, 3))

        coords = np.zeros((n_tokens, 3))
        for i in range(n_tokens):
            atom_mask = atom_to_token == i
            if atom_mask.any():
                coords[i] = atom_ref_pos[atom_mask].mean(axis=0)

        return coords


# =============================================================================
# Molecule Info Transform
# =============================================================================


class MoleculeInfoTransform(BaseCropTransform):
    """Extract molecule type information for cropping.

    Computes ref_space_uid boundaries and identifies metals.
    """

    def __call__(self, data: FeatureDict) -> FeatureDict:
        ref_space_uid = data.get("ref_space_uid", np.array([]))
        chain_id = data.get("asym_id", np.array([]))
        n_tokens = data.get("num_tokens", len(chain_id))

        if n_tokens == 0:
            data["molecule_info"] = None
            return data

        # Compute atom counts per token
        atom_to_token = data.get("atom_to_token", np.array([]))
        atom_counts = np.zeros(n_tokens, dtype=np.int64)
        if len(atom_to_token) > 0:
            for i in range(n_tokens):
                atom_counts[i] = np.sum(atom_to_token == i)
        else:
            atom_counts[:] = 1  # Default

        # Get chain lengths
        unique_chains = np.unique(chain_id)
        chain_lengths = np.zeros(int(unique_chains.max()) + 1 if len(unique_chains) > 0 else 1)
        for c in unique_chains:
            chain_lengths[int(c)] = np.sum(chain_id == c)

        # Use token indices as ref_space_uid if not provided
        if len(ref_space_uid) == 0:
            ref_space_uid = np.arange(n_tokens)

        mol_info = identify_molecule_types(
            ref_space_uid=ref_space_uid,
            atom_counts=atom_counts,
            chain_id=chain_id,
            chain_lengths=chain_lengths,
        )

        data["molecule_info"] = mol_info
        data["ref_space_uid"] = ref_space_uid
        return data


# =============================================================================
# Distance Matrix Transform
# =============================================================================


class TokenDistanceMatrixTransform(BaseCropTransform):
    """Compute token-token distance matrix.

    Can compute full matrix or partial matrix for reference chains only.
    """

    def __init__(
        self,
        config: Optional[CropTransformConfig] = None,
        reference_chain_ids: Optional[np.ndarray] = None,
    ):
        super().__init__(config)
        self.reference_chain_ids = reference_chain_ids

    def __call__(self, data: FeatureDict) -> FeatureDict:
        coords = self._get_token_coords(data)
        chain_id = data.get("asym_id", np.array([]))
        resolved_mask = data.get("is_resolved", np.ones(len(coords), dtype=bool))

        n_tokens = len(coords)
        if n_tokens == 0:
            data["token_distances"] = np.zeros((0, 0))
            data["distance_mask"] = np.zeros((0, 0), dtype=bool)
            return data

        # Determine reference tokens
        if self.reference_chain_ids is not None:
            ref_mask = np.isin(chain_id, self.reference_chain_ids)
            ref_indices = np.where(ref_mask)[0]
        else:
            ref_indices = np.arange(n_tokens)

        if len(ref_indices) == 0:
            ref_indices = np.arange(n_tokens)

        # Compute distances from reference tokens to all tokens
        ref_coords = coords[ref_indices]
        distances = compute_distances_fast(ref_coords, coords)

        # Create distance mask (both tokens must be resolved)
        ref_resolved = resolved_mask[ref_indices]
        distance_mask = ref_resolved[:, np.newaxis] & resolved_mask[np.newaxis, :]

        data["token_distances"] = distances
        data["distance_mask"] = distance_mask
        data["reference_token_indices"] = ref_indices

        return data


# =============================================================================
# Interface Detection Transform
# =============================================================================


class InterfaceTokenTransform(BaseCropTransform):
    """Detect interface tokens between chains.

    Interface tokens are those within interface_distance of another chain.
    """

    def __call__(self, data: FeatureDict) -> FeatureDict:
        chain_id = data.get("asym_id", np.array([]))
        token_distances = data.get("token_distances", np.zeros((0, 0)))
        distance_mask = data.get("distance_mask", np.zeros((0, 0), dtype=bool))
        ref_indices = data.get("reference_token_indices", np.arange(len(chain_id)))

        if len(chain_id) == 0:
            data["interface_token_indices"] = np.array([], dtype=np.int64)
            return data

        # Get unique reference chain IDs
        ref_chain_ids = np.unique(chain_id[ref_indices])

        interface_indices = get_interface_tokens(
            chain_id=chain_id,
            reference_chain_ids=ref_chain_ids,
            token_distances=token_distances,
            distance_mask=distance_mask,
            interface_distance=self.config.interface_distance,
        )

        # If no interface tokens found, use all resolved reference tokens
        if len(interface_indices) == 0:
            resolved_ref = ref_indices[distance_mask.any(axis=1)]
            interface_indices = resolved_ref

        data["interface_token_indices"] = interface_indices
        return data


# =============================================================================
# Contiguous Cropping Transform
# =============================================================================


class ContiguousCropTransform(BaseCropTransform):
    """Apply contiguous cropping across chains.

    From AF-multimer Algorithm 1: Sample contiguous segments from each chain,
    shuffling chain order and respecting molecule boundaries.

    Features:
    - Shuffles chain processing order
    - Can preserve complete ligands/non-standard residues
    - Can remove metal ions
    """

    def __call__(self, data: FeatureDict) -> FeatureDict:
        chain_id = data.get("asym_id", np.array([]))
        n_tokens = data.get("num_tokens", len(chain_id))
        mol_info = data.get("molecule_info")
        ref_space_uid = data.get("ref_space_uid", np.arange(n_tokens))

        if n_tokens == 0:
            data["crop_result"] = CropResult(
                selected_indices=np.array([], dtype=np.int64),
                method="ContiguousCropping",
            )
            return data

        if n_tokens <= self.config.max_tokens:
            data["crop_result"] = CropResult(
                selected_indices=np.arange(n_tokens),
                method="ContiguousCropping",
            )
            return data

        # Get molecule info if not already computed
        if mol_info is None:
            mol_info_transform = MoleculeInfoTransform(self.config)
            data = mol_info_transform(data)
            mol_info = data["molecule_info"]

        selected_indices = self._contiguous_crop(
            n_tokens=n_tokens,
            chain_id=chain_id,
            ref_space_uid=ref_space_uid,
            mol_info=mol_info,
        )

        data["crop_result"] = CropResult(
            selected_indices=selected_indices,
            method="ContiguousCropping",
        )
        return data

    def _contiguous_crop(
        self,
        n_tokens: int,
        chain_id: np.ndarray,
        ref_space_uid: np.ndarray,
        mol_info: MoleculeInfo,
    ) -> np.ndarray:
        """Perform contiguous cropping across chains."""
        crop_size = self.config.max_tokens

        # Get chain info
        unique_chains = np.unique(chain_id)
        chain_lengths = {int(c): np.sum(chain_id == c) for c in unique_chains}
        chain_offsets = {}
        for c in unique_chains:
            chain_offsets[int(c)] = np.where(chain_id == c)[0][0]

        # Shuffle chain order
        shuffled_chains = np.random.permutation(unique_chains)

        selected_indices = []
        n_added = 0
        n_remaining = n_tokens

        if self.config.remove_metals and mol_info is not None:
            n_remaining -= mol_info.is_metal.sum()

        for chain in shuffled_chains:
            chain = int(chain)
            if n_added >= crop_size:
                break

            chain_offset = chain_offsets[chain]
            chain_length = chain_lengths[chain]

            # Skip metals if configured
            if self.config.remove_metals and mol_info is not None:
                if mol_info.is_metal[chain_offset]:
                    continue

            n_remaining -= chain_length

            # Determine crop size for this chain
            crop_size_min = min(
                chain_length,
                max(0, crop_size - (n_added + n_remaining))
            )
            crop_size_max = min(crop_size - n_added, chain_length)

            if crop_size_min > crop_size_max:
                continue

            chain_crop_size = np.random.randint(crop_size_min, crop_size_max + 1)
            chain_crop_start = np.random.randint(0, chain_length - chain_crop_size + 1)

            start_idx = chain_offset + chain_crop_start
            end_idx = chain_offset + chain_crop_start + chain_crop_size

            # Adjust for complete molecules if configured
            if self.config.crop_complete_ligand and mol_info is not None:
                start_idx, end_idx = self._adjust_for_complete_molecules(
                    start_idx, end_idx, crop_size_min, n_added,
                    mol_info, crop_size
                )
                chain_crop_size = end_idx - start_idx

            if start_idx < end_idx:
                selected_indices.extend(range(start_idx, end_idx))
                n_added += chain_crop_size

        selected_indices = np.array(sorted(selected_indices), dtype=np.int64)
        return selected_indices

    def _adjust_for_complete_molecules(
        self,
        start_idx: int,
        end_idx: int,
        crop_size_min: int,
        n_added: int,
        mol_info: MoleculeInfo,
        crop_size: int,
    ) -> Tuple[int, int]:
        """Adjust crop boundaries to keep molecules complete."""
        if start_idx >= end_idx:
            return start_idx, end_idx

        first_indices = mol_info.first_indices
        last_indices = mol_info.last_indices

        # Check if start is in middle of a molecule
        if first_indices[start_idx] != start_idx:
            # Move to molecule boundary
            left_start = first_indices[start_idx]
            right_start = last_indices[start_idx] + 1

            # Prefer starting at molecule boundary
            if right_start <= end_idx:
                start_idx = right_start
            else:
                start_idx = left_start

        # Check if end is in middle of a molecule
        if end_idx > 0 and last_indices[end_idx - 1] != end_idx - 1:
            left_end = first_indices[end_idx - 1]
            right_end = last_indices[end_idx - 1] + 1

            # Check which boundary works
            left_crop = left_end - start_idx
            right_crop = right_end - start_idx

            if left_crop >= crop_size_min and left_crop + n_added <= crop_size:
                if right_crop >= crop_size_min and right_crop + n_added <= crop_size:
                    # Both work, choose randomly
                    end_idx = left_end if np.random.random() < 0.5 else right_end
                else:
                    end_idx = left_end
            elif right_crop >= crop_size_min and right_crop + n_added <= crop_size:
                end_idx = right_end
            elif self.config.drop_last_incomplete:
                # Walk back to find complete molecule
                while end_idx > start_idx:
                    if last_indices[end_idx - 1] == end_idx - 1:
                        break
                    end_idx = first_indices[end_idx - 1]

        return start_idx, end_idx


# =============================================================================
# Spatial Cropping Transform
# =============================================================================


class SpatialCropTransform(BaseCropTransform):
    """Apply spatial cropping based on distance to reference token.

    Selects tokens nearest to a randomly chosen reference token.

    Features:
    - Random selection from resolved reference tokens
    - Ties broken by token index (stability)
    - Can preserve complete ligands
    """

    def __init__(
        self,
        config: Optional[CropTransformConfig] = None,
        use_interface: bool = False,
    ):
        super().__init__(config)
        self.use_interface = use_interface

    def __call__(self, data: FeatureDict) -> FeatureDict:
        n_tokens = data.get("num_tokens", 0)
        token_distances = data.get("token_distances", np.zeros((0, 0)))
        distance_mask = data.get("distance_mask", np.zeros((0, 0), dtype=bool))
        ref_indices = data.get("reference_token_indices", np.arange(n_tokens))
        ref_space_uid = data.get("ref_space_uid", np.arange(n_tokens))

        if n_tokens == 0:
            data["crop_result"] = CropResult(
                selected_indices=np.array([], dtype=np.int64),
                method="SpatialCropping" if not self.use_interface else "SpatialInterfaceCropping",
            )
            return data

        if n_tokens <= self.config.max_tokens:
            data["crop_result"] = CropResult(
                selected_indices=np.arange(n_tokens),
                method="SpatialCropping" if not self.use_interface else "SpatialInterfaceCropping",
            )
            return data

        # For interface cropping, use interface tokens as reference
        if self.use_interface:
            interface_indices = data.get("interface_token_indices", np.array([]))
            if len(interface_indices) > 0:
                # Map interface indices to distance matrix rows
                ref_to_row = {idx: i for i, idx in enumerate(ref_indices)}
                valid_rows = [ref_to_row[idx] for idx in interface_indices if idx in ref_to_row]
                if len(valid_rows) > 0:
                    reference_rows = np.array(valid_rows)
                else:
                    reference_rows = np.where(distance_mask.any(axis=1))[0]
            else:
                reference_rows = np.where(distance_mask.any(axis=1))[0]
        else:
            # Use all resolved reference tokens
            reference_rows = np.where(distance_mask.any(axis=1))[0]

        if len(reference_rows) == 0:
            # Fallback: use all tokens
            data["crop_result"] = CropResult(
                selected_indices=np.arange(min(n_tokens, self.config.max_tokens)),
                method="SpatialCropping" if not self.use_interface else "SpatialInterfaceCropping",
            )
            return data

        # Randomly select reference token
        ref_row = reference_rows[np.random.randint(len(reference_rows))]
        ref_token_idx = ref_indices[ref_row]

        # Get distances from reference
        distances = token_distances[ref_row].copy()

        # Add small noise to break ties (by index)
        noise = np.arange(len(distances)) * 1e-6
        distances = distances + noise

        # Mask unresolved tokens
        valid_mask = distance_mask[ref_row]
        distances[~valid_mask] = np.inf

        # Select nearest tokens
        crop_size = min(self.config.max_tokens, n_tokens)
        nearest_indices = np.argsort(distances)[:crop_size]
        selected_indices = np.sort(nearest_indices)

        # Remove incomplete molecules if configured
        if self.config.crop_complete_ligand:
            selected_indices = self._drop_incomplete_molecules(
                selected_indices, ref_space_uid
            )

        data["crop_result"] = CropResult(
            selected_indices=selected_indices,
            reference_token_idx=int(ref_token_idx),
            method="SpatialCropping" if not self.use_interface else "SpatialInterfaceCropping",
        )
        return data

    def _drop_incomplete_molecules(
        self,
        selected_indices: np.ndarray,
        ref_space_uid: np.ndarray,
    ) -> np.ndarray:
        """Remove tokens from incompletely selected molecules."""
        if len(selected_indices) == 0:
            return selected_indices

        selected_uids = ref_space_uid[selected_indices]

        # Find all uids in full array
        all_uids = set(ref_space_uid)

        # Find uids that are partially selected
        selected_set = set(selected_indices)
        incomplete_uids = set()

        for uid in np.unique(selected_uids):
            uid_indices = np.where(ref_space_uid == uid)[0]
            if not all(idx in selected_set for idx in uid_indices):
                incomplete_uids.add(uid)

        # Remove incomplete molecules
        if incomplete_uids:
            mask = ~np.isin(selected_uids, list(incomplete_uids))
            selected_indices = selected_indices[mask]

        return selected_indices


# =============================================================================
# Combined Cropping Transform
# =============================================================================


class CombinedCropTransform(BaseCropTransform):
    """Combined cropping with method selection.

    Randomly selects between cropping methods based on weights:
    - ContiguousCropping: [weight_0]
    - SpatialCropping: [weight_1]
    - SpatialInterfaceCropping: [weight_2]
    """

    def __init__(
        self,
        config: Optional[CropTransformConfig] = None,
        reference_chain_ids: Optional[np.ndarray] = None,
    ):
        super().__init__(config)
        self.reference_chain_ids = reference_chain_ids

        # Initialize sub-transforms
        self.mol_info_transform = MoleculeInfoTransform(config)
        self.distance_transform = TokenDistanceMatrixTransform(
            config, reference_chain_ids
        )
        self.interface_transform = InterfaceTokenTransform(config)
        self.contiguous_transform = ContiguousCropTransform(config)
        self.spatial_transform = SpatialCropTransform(config, use_interface=False)
        self.spatial_interface_transform = SpatialCropTransform(config, use_interface=True)

    def __call__(self, data: FeatureDict) -> FeatureDict:
        n_tokens = data.get("num_tokens", 0)

        if n_tokens <= self.config.max_tokens:
            data["crop_result"] = CropResult(
                selected_indices=np.arange(n_tokens),
                method="NoCrop",
            )
            return data

        # Randomly select method
        method = self._select_method()

        # Run preprocessing transforms
        data = self.mol_info_transform(data)

        if method == "ContiguousCropping":
            data = self.contiguous_transform(data)
        else:
            # Spatial methods need distance matrix
            data = self.distance_transform(data)

            if method == "SpatialInterfaceCropping":
                data = self.interface_transform(data)
                data = self.spatial_interface_transform(data)
            else:
                data = self.spatial_transform(data)

        return data

    def _select_method(self) -> str:
        """Select cropping method based on weights."""
        methods = ["ContiguousCropping", "SpatialCropping", "SpatialInterfaceCropping"]
        weights = self.config.method_weights

        # Normalize weights
        total = sum(weights)
        if total <= 0:
            return methods[0]

        probs = [w / total for w in weights]
        return np.random.choice(methods, p=probs)


# =============================================================================
# Apply Crop Transform
# =============================================================================


class ApplyCropTransform(BaseCropTransform):
    """Apply crop result to data features.

    Takes crop_result from previous transform and filters all features.
    """

    def __call__(self, data: FeatureDict) -> FeatureDict:
        crop_result = data.get("crop_result")
        if crop_result is None:
            return data

        selected_indices = crop_result.selected_indices
        if len(selected_indices) == 0:
            return data

        # Apply to token-level features
        token_features = [
            "token_index", "residue_index", "asym_id", "entity_id",
            "restype", "is_protein", "is_rna", "is_dna", "is_ligand",
            "pseudo_beta", "pseudo_beta_mask", "backbone_rigid_tensor",
            "backbone_rigid_mask", "ref_space_uid", "centre_atom_coords",
            "is_resolved",
        ]

        for key in token_features:
            if key in data and data[key] is not None:
                arr = data[key]
                if isinstance(arr, np.ndarray) and len(arr) > 0:
                    if arr.ndim >= 1 and arr.shape[0] > max(selected_indices):
                        data[key] = arr[selected_indices]

        # Apply to pair features
        pair_features = ["relative_position", "same_chain", "same_entity"]
        for key in pair_features:
            if key in data and data[key] is not None:
                arr = data[key]
                if isinstance(arr, np.ndarray) and arr.ndim == 2:
                    data[key] = arr[np.ix_(selected_indices, selected_indices)]

        # Apply to MSA features
        msa_features = ["msa", "msa_mask", "msa_deletion_value"]
        for key in msa_features:
            if key in data and data[key] is not None:
                arr = data[key]
                if isinstance(arr, np.ndarray) and arr.ndim >= 2:
                    data[key] = arr[:, selected_indices]

        # Apply to template features
        template_features = ["template_restype", "template_pseudo_beta",
                           "template_pseudo_beta_mask", "template_backbone_mask"]
        for key in template_features:
            if key in data and data[key] is not None:
                arr = data[key]
                if isinstance(arr, np.ndarray) and arr.ndim >= 2:
                    data[key] = arr[:, selected_indices]

        # Update metadata
        data["num_tokens"] = len(selected_indices)
        data["crop_method"] = crop_result.method
        data["reference_token_idx"] = crop_result.reference_token_idx

        return data


# =============================================================================
# Pipeline Factory
# =============================================================================


def create_crop_pipeline(
    config: Optional[CropTransformConfig] = None,
    reference_chain_ids: Optional[np.ndarray] = None,
) -> List[BaseCropTransform]:
    """Create a standard cropping pipeline.

    Args:
        config: Cropping configuration
        reference_chain_ids: Chain IDs for reference (spatial cropping)

    Returns:
        List of transforms to apply in order
    """
    config = config or CropTransformConfig()

    return [
        CombinedCropTransform(config, reference_chain_ids),
        ApplyCropTransform(config),
    ]


def apply_crop_pipeline(
    data: FeatureDict,
    pipeline: List[BaseCropTransform],
) -> FeatureDict:
    """Apply a cropping pipeline to data.

    Args:
        data: Feature dictionary
        pipeline: List of transforms

    Returns:
        Transformed data
    """
    for transform in pipeline:
        data = transform(data)
    return data


# =============================================================================
# Legacy Compatibility
# =============================================================================


class CropTransform(BaseCropTransform):
    """Legacy combined cropping transform.

    Maintained for backward compatibility.
    """

    def __init__(
        self,
        max_tokens: int = 384,
        max_atoms: int = 4608,
        spatial_crop_prob: float = 0.5,
    ):
        config = CropTransformConfig(
            max_tokens=max_tokens,
            max_atoms=max_atoms,
            method_weights=(1 - spatial_crop_prob, spatial_crop_prob, 0.0),
        )
        super().__init__(config)
        self._combined = CombinedCropTransform(config)
        self._apply = ApplyCropTransform(config)

    def __call__(self, data: FeatureDict) -> FeatureDict:
        data = self._combined(data)
        data = self._apply(data)
        return data


class CropToTokenLimitTransform(BaseCropTransform):
    """Ensure structure fits within token limit.

    Final transform to guarantee size constraints.
    """

    def __call__(self, data: FeatureDict) -> FeatureDict:
        n_tokens = data.get("num_tokens", 0)

        if n_tokens <= self.config.max_tokens:
            return data

        # Simple truncation if no crop result
        if "crop_result" not in data:
            data["crop_result"] = CropResult(
                selected_indices=np.arange(self.config.max_tokens),
                method="Truncation",
            )

        # Apply crop
        apply_transform = ApplyCropTransform(self.config)
        return apply_transform(data)


# =============================================================================
# Backward Compatibility with cropping.py API
# =============================================================================

from enum import Enum, auto


class CroppingStrategy(Enum):
    """Cropping strategy selection (backward compatibility)."""
    CONTIGUOUS = auto()
    SPATIAL = auto()
    SPATIAL_INTERFACE = auto()


@dataclass
class CropConfig:
    """Configuration for cropping (backward compatibility).

    From AF3 Table 4 and Section 2.7.
    """
    max_tokens: int = 384
    max_atoms: int = 4608
    contiguous_weight: float = 0.2
    spatial_weight: float = 0.4
    spatial_interface_weight: float = 0.4
    contiguous_crop_size: int = 128
    spatial_radius: float = 24.0
    interface_distance: float = 15.0

    def to_transform_config(self) -> CropTransformConfig:
        """Convert to CropTransformConfig."""
        return CropTransformConfig(
            max_tokens=self.max_tokens,
            max_atoms=self.max_atoms,
            interface_distance=self.interface_distance,
            method_weights=(
                self.contiguous_weight,
                self.spatial_weight,
                self.spatial_interface_weight,
            ),
        )


class Cropper:
    """Main cropper with backward-compatible API.

    Wraps the new transform-based implementation while providing
    the same interface as the original Cropper class.
    """

    def __init__(self, config: Optional[CropConfig] = None):
        self.config = config or CropConfig()
        self._transform_config = self.config.to_transform_config()

    def crop(
        self,
        tokenized: Any,  # TokenizedStructure
        rng: Optional[np.random.Generator] = None,
        strategy: Optional[CroppingStrategy] = None,
    ) -> "LegacyCropResult":
        """Crop a tokenized structure.

        Args:
            tokenized: Structure to crop
            rng: Random number generator
            strategy: Specific strategy to use, or None to sample

        Returns:
            LegacyCropResult with selected token indices
        """
        if rng is None:
            rng = np.random.default_rng()

        # Set seed for reproducibility
        np.random.seed(rng.integers(0, 2**31))

        tokens = tokenized.tokens
        n_tokens = len(tokens)

        if n_tokens <= self.config.max_tokens:
            return LegacyCropResult(
                token_indices=np.arange(n_tokens, dtype=np.int32),
                strategy=strategy or CroppingStrategy.CONTIGUOUS,
            )

        # Build feature dict from tokenized structure
        data = self._tokenized_to_feature_dict(tokenized)

        # Select strategy
        if strategy is None:
            strategy = self._sample_strategy(rng)

        # Configure transform based on strategy
        if strategy == CroppingStrategy.CONTIGUOUS:
            method_weights = (1.0, 0.0, 0.0)
        elif strategy == CroppingStrategy.SPATIAL:
            method_weights = (0.0, 1.0, 0.0)
        else:
            method_weights = (0.0, 0.0, 1.0)

        config = CropTransformConfig(
            max_tokens=self.config.max_tokens,
            max_atoms=self.config.max_atoms,
            interface_distance=self.config.interface_distance,
            method_weights=method_weights,
        )

        # Run transform pipeline
        combined = CombinedCropTransform(config)
        data = combined(data)

        crop_result = data.get("crop_result")
        if crop_result is None:
            return LegacyCropResult(
                token_indices=np.arange(min(n_tokens, self.config.max_tokens), dtype=np.int32),
                strategy=strategy,
            )

        return LegacyCropResult(
            token_indices=crop_result.selected_indices.astype(np.int32),
            strategy=strategy,
            center_token_idx=crop_result.reference_token_idx if crop_result.reference_token_idx >= 0 else None,
        )

    def _tokenized_to_feature_dict(self, tokenized: Any) -> FeatureDict:
        """Convert TokenizedStructure to feature dictionary."""
        tokens = tokenized.tokens
        n_tokens = len(tokens)

        # Extract chain IDs
        chain_ids = np.zeros(n_tokens, dtype=np.int64)
        chain_id_map = {}
        for i, token in enumerate(tokens):
            if token.chain_id not in chain_id_map:
                chain_id_map[token.chain_id] = len(chain_id_map)
            chain_ids[i] = chain_id_map[token.chain_id]

        # Extract coordinates
        coords = np.zeros((n_tokens, 3), dtype=np.float32)
        for i, token in enumerate(tokens):
            if token.center_coords is not None:
                coords[i] = token.center_coords

        # Extract ref_space_uid (use residue indices)
        ref_space_uid = np.zeros(n_tokens, dtype=np.int64)
        for i, token in enumerate(tokens):
            ref_space_uid[i] = getattr(token, 'residue_index', i)

        return {
            "num_tokens": n_tokens,
            "asym_id": chain_ids,
            "centre_atom_coords": coords,
            "ref_space_uid": ref_space_uid,
            "is_resolved": np.ones(n_tokens, dtype=bool),
        }

    def _sample_strategy(self, rng: np.random.Generator) -> CroppingStrategy:
        """Sample a cropping strategy based on configured weights."""
        weights = np.array([
            self.config.contiguous_weight,
            self.config.spatial_weight,
            self.config.spatial_interface_weight,
        ])
        weights = weights / weights.sum()

        strategies = [
            CroppingStrategy.CONTIGUOUS,
            CroppingStrategy.SPATIAL,
            CroppingStrategy.SPATIAL_INTERFACE,
        ]

        idx = rng.choice(len(strategies), p=weights)
        return strategies[idx]

    def contiguous_crop(
        self,
        tokenized: Any,
        max_tokens: int,
        rng: Optional[np.random.Generator] = None,
    ) -> Any:
        """Apply contiguous cropping strategy."""
        result = self.crop(tokenized, rng, strategy=CroppingStrategy.CONTIGUOUS)
        return self._apply_crop_result(tokenized, result)

    def spatial_crop(
        self,
        tokenized: Any,
        max_tokens: int,
        rng: Optional[np.random.Generator] = None,
    ) -> Any:
        """Apply spatial cropping strategy."""
        result = self.crop(tokenized, rng, strategy=CroppingStrategy.SPATIAL)
        return self._apply_crop_result(tokenized, result)

    def crop_to_token_limit(
        self,
        tokenized: Any,
        max_tokens: int,
        rng: Optional[np.random.Generator] = None,
    ) -> Any:
        """Crop and return new tokenized structure."""
        if len(tokenized.tokens) <= max_tokens:
            return tokenized

        # Create temporary config
        temp_config = CropConfig(
            max_tokens=max_tokens,
            contiguous_weight=self.config.contiguous_weight,
            spatial_weight=self.config.spatial_weight,
            spatial_interface_weight=self.config.spatial_interface_weight,
        )
        temp_cropper = Cropper(temp_config)
        result = temp_cropper.crop(tokenized, rng)

        return self._apply_crop_result(tokenized, result)

    def _apply_crop_result(
        self,
        tokenized: Any,
        result: "LegacyCropResult",
    ) -> Any:
        """Apply a crop result to create new TokenizedStructure."""
        from novadb.processing.tokenization.tokenizer import TokenizedStructure

        selected = set(result.token_indices)
        new_tokens = []
        for i, token in enumerate(tokenized.tokens):
            if i in selected:
                new_tokens.append(token)

        # Renumber token indices
        for i, token in enumerate(new_tokens):
            token.token_index = i

        return TokenizedStructure(
            tokens=new_tokens,
            pdb_id=tokenized.pdb_id,
            chain_id_to_index=tokenized.chain_id_to_index,
            entity_id_map=tokenized.entity_id_map,
        )


@dataclass
class LegacyCropResult:
    """Result of cropping operation (backward compatibility)."""
    token_indices: np.ndarray
    strategy: CroppingStrategy
    center_token_idx: Optional[int] = None
    interface_chains: Optional[Tuple[str, str]] = None

    @property
    def num_tokens(self) -> int:
        return len(self.token_indices)


# Backward-compatible croppers
class ContiguousCropper:
    """Contiguous cropping strategy (backward compatibility)."""

    def __init__(self, config: CropConfig):
        self.config = config

    def crop(self, tokenized: Any, rng: np.random.Generator) -> LegacyCropResult:
        cropper = Cropper(self.config)
        return cropper.crop(tokenized, rng, strategy=CroppingStrategy.CONTIGUOUS)


class SpatialCropper:
    """Spatial cropping strategy (backward compatibility)."""

    def __init__(self, config: CropConfig):
        self.config = config

    def crop(self, tokenized: Any, rng: np.random.Generator) -> LegacyCropResult:
        cropper = Cropper(self.config)
        return cropper.crop(tokenized, rng, strategy=CroppingStrategy.SPATIAL)


class SpatialInterfaceCropper:
    """Spatial interface cropping strategy (backward compatibility)."""

    def __init__(self, config: CropConfig):
        self.config = config

    def crop(self, tokenized: Any, rng: np.random.Generator) -> LegacyCropResult:
        cropper = Cropper(self.config)
        return cropper.crop(tokenized, rng, strategy=CroppingStrategy.SPATIAL_INTERFACE)


def check_atom_limit(
    tokens: List[Any],
    selected_indices: np.ndarray,
    max_atoms: int,
) -> np.ndarray:
    """Ensure selected tokens don't exceed atom limit.

    From AF3: Both token and atom limits must be respected.
    """
    total_atoms = 0
    valid_indices = []

    for idx in selected_indices:
        token = tokens[idx]
        num_atoms = getattr(token, 'num_atoms', 1)
        if total_atoms + num_atoms <= max_atoms:
            valid_indices.append(idx)
            total_atoms += num_atoms
        else:
            break

    return np.array(valid_indices, dtype=np.int32)
