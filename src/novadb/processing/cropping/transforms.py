"""Cropping transforms for NovaDB.

Provides composable transforms for structure cropping following AF3 Section 2.7.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# Type alias
FeatureDict = Dict[str, Any]


@dataclass
class StructureCache:
    """Cache for structure data during cropping.

    Stores precomputed data to avoid redundant calculations.
    """

    chain_centers: Optional[Dict[str, np.ndarray]] = None
    chain_coords: Optional[Dict[str, np.ndarray]] = None
    interface_pairs: Optional[List[Tuple[str, str]]] = None
    kdtrees: Optional[Dict[str, Any]] = None

    def clear(self) -> None:
        """Clear all cached data."""
        self.chain_centers = None
        self.chain_coords = None
        self.interface_pairs = None
        self.kdtrees = None


class BaseCropTransform:
    """Base class for cropping transforms."""

    def __init__(self, max_tokens: int = 384, max_atoms: int = 16384):
        self.max_tokens = max_tokens
        self.max_atoms = max_atoms

    def __call__(self, data: FeatureDict) -> FeatureDict:
        raise NotImplementedError


class ChainContiguousCropTransform(BaseCropTransform):
    """Crop contiguous segments from chains.

    From AF3 Section 2.7: Sample contiguous crop from each chain.
    """

    def __call__(self, data: FeatureDict) -> FeatureDict:
        # Get chain information
        chain_ids = data.get("chain_ids", [])
        token_indices = data.get("token_index", np.array([]))

        if len(token_indices) <= self.max_tokens:
            data["crop_mask"] = np.ones(len(token_indices), dtype=bool)
            return data

        # Simple contiguous crop - take first max_tokens
        crop_mask = np.zeros(len(token_indices), dtype=bool)
        crop_mask[:self.max_tokens] = True

        data["crop_mask"] = crop_mask
        return data


class ChainCenterCoordsTransform(BaseCropTransform):
    """Compute center coordinates for each chain."""

    def __call__(self, data: FeatureDict) -> FeatureDict:
        atom_ref_pos = data.get("atom_ref_pos", np.zeros((0, 3)))
        atom_to_token = data.get("atom_to_token", np.array([]))
        asym_id = data.get("asym_id", np.array([]))

        chain_centers = {}

        if len(asym_id) > 0 and len(atom_ref_pos) > 0:
            unique_chains = np.unique(asym_id)

            for chain_idx in unique_chains:
                # Get tokens in this chain
                token_mask = asym_id == chain_idx
                token_indices = np.where(token_mask)[0]

                if len(token_indices) == 0:
                    continue

                # Get atoms for these tokens
                atom_mask = np.isin(atom_to_token, token_indices)
                chain_coords = atom_ref_pos[atom_mask]

                if len(chain_coords) > 0:
                    chain_centers[int(chain_idx)] = chain_coords.mean(axis=0)

        data["chain_centers"] = chain_centers
        return data


class ChainInterfaceDetectTransform(BaseCropTransform):
    """Detect interfaces between chains.

    From AF3 Section 2.7: Identify chain pairs with contacts.
    """

    def __init__(
        self,
        max_tokens: int = 384,
        max_atoms: int = 16384,
        contact_threshold: float = 8.0,
    ):
        super().__init__(max_tokens, max_atoms)
        self.contact_threshold = contact_threshold

    def __call__(self, data: FeatureDict) -> FeatureDict:
        chain_centers = data.get("chain_centers", {})

        interface_pairs = []

        chain_ids = sorted(chain_centers.keys())
        for i, chain_i in enumerate(chain_ids):
            for chain_j in chain_ids[i+1:]:
                center_i = chain_centers[chain_i]
                center_j = chain_centers[chain_j]

                dist = np.linalg.norm(center_i - center_j)

                # Use a larger threshold for center-based detection
                if dist < self.contact_threshold * 5:
                    interface_pairs.append((chain_i, chain_j))

        data["interface_pairs"] = interface_pairs
        return data


class ContiguousCropTransform(BaseCropTransform):
    """Apply contiguous cropping to structure.

    From AF3 Section 2.7: Sample start points uniformly.
    """

    def __call__(self, data: FeatureDict) -> FeatureDict:
        n_tokens = data.get("num_tokens", 0)

        if n_tokens <= self.max_tokens:
            data["crop_start"] = 0
            data["crop_end"] = n_tokens
            return data

        # Random start point
        max_start = n_tokens - self.max_tokens
        start = np.random.randint(0, max_start + 1)

        data["crop_start"] = start
        data["crop_end"] = start + self.max_tokens
        return data


class SpatialCropTransform(BaseCropTransform):
    """Apply spatial cropping to structure.

    From AF3 Section 2.7: Crop based on spatial proximity.
    """

    def __init__(
        self,
        max_tokens: int = 384,
        max_atoms: int = 16384,
        seed_selection: str = "random",
    ):
        super().__init__(max_tokens, max_atoms)
        self.seed_selection = seed_selection

    def __call__(self, data: FeatureDict) -> FeatureDict:
        pseudo_beta = data.get("pseudo_beta", np.zeros((0, 3)))
        pseudo_beta_mask = data.get("pseudo_beta_mask", np.array([]))
        n_tokens = len(pseudo_beta)

        if n_tokens <= self.max_tokens:
            data["crop_indices"] = np.arange(n_tokens)
            return data

        # Select seed token
        valid_indices = np.where(pseudo_beta_mask > 0)[0]
        if len(valid_indices) == 0:
            valid_indices = np.arange(n_tokens)

        if self.seed_selection == "random":
            seed_idx = np.random.choice(valid_indices)
        else:
            seed_idx = valid_indices[0]

        # Compute distances from seed
        seed_pos = pseudo_beta[seed_idx]
        distances = np.linalg.norm(pseudo_beta - seed_pos, axis=1)

        # Select closest tokens
        sorted_indices = np.argsort(distances)
        crop_indices = sorted_indices[:self.max_tokens]

        data["crop_indices"] = np.sort(crop_indices)
        return data


class SpatialInterfaceCropTransform(BaseCropTransform):
    """Crop around interface regions.

    From AF3 Section 2.7: Focus on chain-chain interfaces.
    """

    def __call__(self, data: FeatureDict) -> FeatureDict:
        interface_pairs = data.get("interface_pairs", [])
        chain_centers = data.get("chain_centers", {})
        pseudo_beta = data.get("pseudo_beta", np.zeros((0, 3)))
        asym_id = data.get("asym_id", np.array([]))

        n_tokens = len(pseudo_beta)

        if n_tokens <= self.max_tokens or not interface_pairs:
            data["crop_indices"] = np.arange(min(n_tokens, self.max_tokens))
            return data

        # Select random interface
        chain_i, chain_j = interface_pairs[np.random.randint(len(interface_pairs))]

        # Get interface center
        if chain_i in chain_centers and chain_j in chain_centers:
            interface_center = (chain_centers[chain_i] + chain_centers[chain_j]) / 2
        else:
            interface_center = pseudo_beta.mean(axis=0)

        # Compute distances
        distances = np.linalg.norm(pseudo_beta - interface_center, axis=1)

        # Select closest tokens
        sorted_indices = np.argsort(distances)
        crop_indices = sorted_indices[:self.max_tokens]

        data["crop_indices"] = np.sort(crop_indices)
        return data


class CropTransform(BaseCropTransform):
    """Combined cropping transform.

    Selects between contiguous and spatial cropping based on structure.
    """

    def __init__(
        self,
        max_tokens: int = 384,
        max_atoms: int = 16384,
        spatial_crop_prob: float = 0.5,
    ):
        super().__init__(max_tokens, max_atoms)
        self.spatial_crop_prob = spatial_crop_prob
        self.contiguous = ContiguousCropTransform(max_tokens, max_atoms)
        self.spatial = SpatialCropTransform(max_tokens, max_atoms)

    def __call__(self, data: FeatureDict) -> FeatureDict:
        if np.random.random() < self.spatial_crop_prob:
            return self.spatial(data)
        else:
            return self.contiguous(data)


class CropToTokenLimitTransform(BaseCropTransform):
    """Ensure structure fits within token limit.

    Final transform to guarantee size constraints.
    """

    def __call__(self, data: FeatureDict) -> FeatureDict:
        n_tokens = data.get("num_tokens", 0)

        if n_tokens <= self.max_tokens:
            return data

        # Apply crop indices if available
        if "crop_indices" in data:
            crop_indices = data["crop_indices"][:self.max_tokens]
            data["crop_indices"] = crop_indices
            data["num_tokens"] = len(crop_indices)
        else:
            # Simple truncation
            data["crop_indices"] = np.arange(self.max_tokens)
            data["num_tokens"] = self.max_tokens

        return data


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
        from scipy.spatial.distance import cdist
        return cdist(coords1, coords2, metric='euclidean')
    else:
        # Fallback to broadcasting
        diff = coords1[:, np.newaxis, :] - coords2[np.newaxis, :, :]
        return np.sqrt(np.sum(diff ** 2, axis=-1))


def find_interface_contacts_fast(
    coords1: np.ndarray,
    coords2: np.ndarray,
    threshold: float = 8.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Find contacts between two sets of coordinates.

    Args:
        coords1: First coordinates (N, 3)
        coords2: Second coordinates (M, 3)
        threshold: Contact distance threshold

    Returns:
        Tuple of (indices1, indices2) for contacting pairs
    """
    if len(coords1) == 0 or len(coords2) == 0:
        return np.array([]), np.array([])

    if HAS_SCIPY:
        tree = cKDTree(coords2)
        contacts = tree.query_ball_point(coords1, threshold)

        idx1 = []
        idx2 = []
        for i, contact_list in enumerate(contacts):
            for j in contact_list:
                idx1.append(i)
                idx2.append(j)

        return np.array(idx1), np.array(idx2)
    else:
        # Fallback
        distances = compute_distances_fast(coords1, coords2)
        idx1, idx2 = np.where(distances < threshold)
        return idx1, idx2
