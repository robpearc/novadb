"""Optimized operations for NovaDB processing.

This module provides performance-optimized implementations of common operations:
- KD-tree based spatial queries for bond detection and clash detection
- Vectorized pairwise distance computations
- Optimized distogram and pair feature computations
- Efficient contact detection between chains

These implementations replace O(n²) to O(n⁴) nested loops with:
- scipy.spatial.KDTree for O(n log n) spatial queries
- numpy broadcasting for vectorized operations
- scipy.spatial.distance for optimized distance matrices
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union
import numpy as np

try:
    from scipy.spatial import KDTree, cKDTree
    from scipy.spatial.distance import cdist
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Use cKDTree if available (faster C implementation)
SpatialTree = cKDTree if HAS_SCIPY else None


# =============================================================================
# Optimized Distance Computations
# =============================================================================


def compute_pairwise_distances_fast(
    coords1: np.ndarray,
    coords2: Optional[np.ndarray] = None,
    metric: str = "euclidean",
) -> np.ndarray:
    """Compute pairwise distances using scipy.cdist.

    This is significantly faster than nested loops or broadcasting
    for large coordinate arrays.

    Args:
        coords1: First set of coordinates (N, 3).
        coords2: Second set of coordinates (M, 3). If None, computes
            self-distances for coords1.
        metric: Distance metric (default: euclidean).

    Returns:
        Distance matrix (N, M) or (N, N) if coords2 is None.
    """
    if len(coords1) == 0:
        if coords2 is None:
            return np.zeros((0, 0), dtype=np.float32)
        return np.zeros((0, len(coords2)), dtype=np.float32)

    if coords2 is None:
        coords2 = coords1

    if len(coords2) == 0:
        return np.zeros((len(coords1), 0), dtype=np.float32)

    if HAS_SCIPY:
        return cdist(coords1, coords2, metric=metric).astype(np.float32)
    else:
        # Fallback to broadcasting
        diff = coords1[:, np.newaxis, :] - coords2[np.newaxis, :, :]
        return np.sqrt(np.sum(diff ** 2, axis=-1)).astype(np.float32)


def compute_min_distances_kdtree(
    coords1: np.ndarray,
    coords2: np.ndarray,
    k: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute minimum distances using KD-tree.

    Much faster than brute force for finding nearest neighbors.

    Args:
        coords1: Query coordinates (N, 3).
        coords2: Reference coordinates to search (M, 3).
        k: Number of nearest neighbors.

    Returns:
        Tuple of (distances, indices) for k nearest neighbors.
        distances: (N, k) array of distances.
        indices: (N, k) array of indices into coords2.
    """
    if len(coords1) == 0 or len(coords2) == 0:
        return (
            np.zeros((len(coords1), k), dtype=np.float32),
            np.zeros((len(coords1), k), dtype=np.int32),
        )

    if HAS_SCIPY:
        tree = cKDTree(coords2)
        distances, indices = tree.query(coords1, k=k)
        if k == 1:
            distances = distances.reshape(-1, 1)
            indices = indices.reshape(-1, 1)
        return distances.astype(np.float32), indices.astype(np.int32)
    else:
        # Fallback to brute force
        dist_matrix = compute_pairwise_distances_fast(coords1, coords2)
        indices = np.argsort(dist_matrix, axis=1)[:, :k]
        distances = np.take_along_axis(dist_matrix, indices, axis=1)
        return distances, indices


def find_pairs_within_distance(
    coords1: np.ndarray,
    coords2: np.ndarray,
    threshold: float,
) -> List[Tuple[int, int, float]]:
    """Find all pairs within a distance threshold using KD-tree.

    This is O(n log n + k) where k is the number of pairs found,
    compared to O(n²) for brute force.

    Args:
        coords1: First set of coordinates (N, 3).
        coords2: Second set of coordinates (M, 3).
        threshold: Distance threshold.

    Returns:
        List of (idx1, idx2, distance) tuples for pairs within threshold.
    """
    if len(coords1) == 0 or len(coords2) == 0:
        return []

    pairs = []

    if HAS_SCIPY:
        tree2 = cKDTree(coords2)
        # Query all points in coords1 for neighbors in coords2
        for i, coord in enumerate(coords1):
            indices = tree2.query_ball_point(coord, threshold)
            for j in indices:
                dist = np.linalg.norm(coord - coords2[j])
                pairs.append((i, j, float(dist)))
    else:
        # Fallback to vectorized approach
        dist_matrix = compute_pairwise_distances_fast(coords1, coords2)
        i_indices, j_indices = np.where(dist_matrix <= threshold)
        for i, j in zip(i_indices, j_indices):
            pairs.append((int(i), int(j), float(dist_matrix[i, j])))

    return pairs


def find_all_pairs_within_distance_symmetric(
    coords: np.ndarray,
    threshold: float,
) -> List[Tuple[int, int, float]]:
    """Find all pairs within threshold for a single set of coordinates.

    Uses KD-tree query_pairs for optimal performance.

    Args:
        coords: Coordinates (N, 3).
        threshold: Distance threshold.

    Returns:
        List of (idx1, idx2, distance) tuples for pairs within threshold.
    """
    if len(coords) == 0:
        return []

    pairs = []

    if HAS_SCIPY:
        tree = cKDTree(coords)
        # query_pairs returns set of (i, j) with i < j
        pair_set = tree.query_pairs(threshold, output_type='set')
        for i, j in pair_set:
            dist = np.linalg.norm(coords[i] - coords[j])
            pairs.append((i, j, float(dist)))
    else:
        dist_matrix = compute_pairwise_distances_fast(coords, coords)
        i_indices, j_indices = np.where(
            (dist_matrix <= threshold) & (np.triu(np.ones_like(dist_matrix), k=1) > 0)
        )
        for i, j in zip(i_indices, j_indices):
            pairs.append((int(i), int(j), float(dist_matrix[i, j])))

    return pairs


# =============================================================================
# Optimized Clash Detection
# =============================================================================


def compute_clash_fraction_fast(
    query_coords: np.ndarray,
    reference_coords: np.ndarray,
    threshold: float = 1.7,
) -> float:
    """Compute fraction of query atoms clashing with reference atoms.

    Uses KD-tree for O(n log m) complexity instead of O(n*m).

    Args:
        query_coords: Coordinates to check for clashes (N, 3).
        reference_coords: Reference coordinates (M, 3).
        threshold: Clash distance threshold.

    Returns:
        Fraction of query atoms within threshold of any reference atom.
    """
    if len(query_coords) == 0:
        return 0.0
    if len(reference_coords) == 0:
        return 0.0

    if HAS_SCIPY:
        tree = cKDTree(reference_coords)
        # Count how many query points have at least one neighbor within threshold
        distances, _ = tree.query(query_coords, k=1)
        clash_count = np.sum(distances <= threshold)
        return float(clash_count) / len(query_coords)
    else:
        dist_matrix = compute_pairwise_distances_fast(query_coords, reference_coords)
        min_distances = np.min(dist_matrix, axis=1)
        clash_count = np.sum(min_distances <= threshold)
        return float(clash_count) / len(query_coords)


def detect_clashing_chains_fast(
    chain_coords: Dict[str, np.ndarray],
    distance_threshold: float = 1.7,
    fraction_threshold: float = 0.30,
) -> List[str]:
    """Detect chains with excessive atomic clashes.

    Uses KD-tree for efficient clash detection.

    Args:
        chain_coords: Dictionary mapping chain_id to coordinates.
        distance_threshold: Distance for clash (1.7Å).
        fraction_threshold: Fraction constituting clash (0.30).

    Returns:
        List of chain IDs that should be removed due to clashes.
    """
    chain_ids = list(chain_coords.keys())
    chains_to_remove = []

    # Build KD-tree for all coordinates with chain labels
    all_coords = []
    coord_chain_map = []
    for chain_id in chain_ids:
        coords = chain_coords[chain_id]
        if len(coords) > 0:
            all_coords.append(coords)
            coord_chain_map.extend([chain_id] * len(coords))

    if not all_coords:
        return []

    all_coords_arr = np.concatenate(all_coords)
    coord_chain_map = np.array(coord_chain_map)

    if HAS_SCIPY:
        tree = cKDTree(all_coords_arr)

        for chain_id in chain_ids:
            coords = chain_coords[chain_id]
            if len(coords) == 0:
                continue

            # Find all atoms within threshold
            distances, indices = tree.query(coords, k=min(10, len(all_coords_arr)))

            # Count clashes with OTHER chains
            clash_count = 0
            for i, (dists, idxs) in enumerate(zip(distances, indices)):
                for d, idx in zip(dists, idxs):
                    if d <= distance_threshold and coord_chain_map[idx] != chain_id:
                        clash_count += 1
                        break  # Only count each query atom once

            clash_fraction = clash_count / len(coords)
            if clash_fraction >= fraction_threshold:
                chains_to_remove.append(chain_id)
    else:
        # Fallback for each chain
        for chain_id in chain_ids:
            coords = chain_coords[chain_id]
            if len(coords) == 0:
                continue

            # Get other chain coords
            other_coords = [
                chain_coords[cid] for cid in chain_ids
                if cid != chain_id and len(chain_coords[cid]) > 0
            ]
            if not other_coords:
                continue

            other_coords_arr = np.concatenate(other_coords)
            clash_frac = compute_clash_fraction_fast(
                coords, other_coords_arr, distance_threshold
            )
            if clash_frac >= fraction_threshold:
                chains_to_remove.append(chain_id)

    return chains_to_remove


# =============================================================================
# Optimized Bond Detection
# =============================================================================


@dataclass
class BondCandidate:
    """Candidate bond between atoms."""
    chain1_id: str
    residue1_idx: int
    atom1_idx: int
    atom1_name: str
    chain2_id: str
    residue2_idx: int
    atom2_idx: int
    atom2_name: str
    distance: float


def detect_bonds_kdtree(
    atom_coords: np.ndarray,
    atom_info: List[Tuple[str, int, str]],  # (chain_id, residue_idx, atom_name)
    threshold: float = 2.4,
    same_residue: bool = False,
) -> List[BondCandidate]:
    """Detect bonds using KD-tree for O(n log n) complexity.

    Args:
        atom_coords: All atom coordinates (N, 3).
        atom_info: List of (chain_id, residue_idx, atom_name) for each atom.
        threshold: Bond distance threshold.
        same_residue: If True, include intra-residue bonds.

    Returns:
        List of BondCandidate objects.
    """
    if len(atom_coords) == 0:
        return []

    bonds = []

    if HAS_SCIPY:
        tree = cKDTree(atom_coords)
        pairs = tree.query_pairs(threshold, output_type='set')

        for i, j in pairs:
            chain1, res1, name1 = atom_info[i]
            chain2, res2, name2 = atom_info[j]

            # Skip intra-residue unless requested
            if not same_residue and chain1 == chain2 and res1 == res2:
                continue

            dist = np.linalg.norm(atom_coords[i] - atom_coords[j])
            bonds.append(BondCandidate(
                chain1_id=chain1,
                residue1_idx=res1,
                atom1_idx=i,
                atom1_name=name1,
                chain2_id=chain2,
                residue2_idx=res2,
                atom2_idx=j,
                atom2_name=name2,
                distance=dist,
            ))
    else:
        # Fallback to vectorized distance matrix
        dist_matrix = compute_pairwise_distances_fast(atom_coords, atom_coords)
        i_indices, j_indices = np.where(
            (dist_matrix <= threshold) & (np.triu(np.ones_like(dist_matrix), k=1) > 0)
        )

        for i, j in zip(i_indices, j_indices):
            chain1, res1, name1 = atom_info[i]
            chain2, res2, name2 = atom_info[j]

            if not same_residue and chain1 == chain2 and res1 == res2:
                continue

            bonds.append(BondCandidate(
                chain1_id=chain1,
                residue1_idx=res1,
                atom1_idx=int(i),
                atom1_name=name1,
                chain2_id=chain2,
                residue2_idx=res2,
                atom2_idx=int(j),
                atom2_name=name2,
                distance=float(dist_matrix[i, j]),
            ))

    return bonds


def detect_cross_chain_bonds_fast(
    coords1: np.ndarray,
    coords2: np.ndarray,
    info1: List[Tuple[int, str]],  # (residue_idx, atom_name)
    info2: List[Tuple[int, str]],
    chain1_id: str,
    chain2_id: str,
    threshold: float = 2.4,
) -> List[BondCandidate]:
    """Detect bonds between two chains using KD-tree.

    Replaces O(n*m) nested loop with O(n log m) KD-tree query.

    Args:
        coords1: Coordinates of first chain (N, 3).
        coords2: Coordinates of second chain (M, 3).
        info1: (residue_idx, atom_name) for each atom in chain1.
        info2: (residue_idx, atom_name) for each atom in chain2.
        chain1_id: ID of first chain.
        chain2_id: ID of second chain.
        threshold: Bond distance threshold.

    Returns:
        List of BondCandidate objects.
    """
    if len(coords1) == 0 or len(coords2) == 0:
        return []

    bonds = []

    if HAS_SCIPY:
        tree2 = cKDTree(coords2)

        for i, coord in enumerate(coords1):
            indices = tree2.query_ball_point(coord, threshold)
            res1, name1 = info1[i]

            for j in indices:
                res2, name2 = info2[j]
                dist = np.linalg.norm(coord - coords2[j])

                bonds.append(BondCandidate(
                    chain1_id=chain1_id,
                    residue1_idx=res1,
                    atom1_idx=i,
                    atom1_name=name1,
                    chain2_id=chain2_id,
                    residue2_idx=res2,
                    atom2_idx=j,
                    atom2_name=name2,
                    distance=dist,
                ))
    else:
        pairs = find_pairs_within_distance(coords1, coords2, threshold)
        for i, j, dist in pairs:
            res1, name1 = info1[i]
            res2, name2 = info2[j]
            bonds.append(BondCandidate(
                chain1_id=chain1_id,
                residue1_idx=res1,
                atom1_idx=i,
                atom1_name=name1,
                chain2_id=chain2_id,
                residue2_idx=res2,
                atom2_idx=j,
                atom2_name=name2,
                distance=dist,
            ))

    return bonds


# =============================================================================
# Optimized Distogram Computation
# =============================================================================


def compute_distogram_vectorized(
    positions: np.ndarray,
    mask: np.ndarray,
    bin_edges: np.ndarray,
) -> np.ndarray:
    """Compute distogram using fully vectorized operations.

    Replaces nested loop binning with vectorized digitize.

    Args:
        positions: Positions (N, 3) or (batch, N, 3).
        mask: Validity mask (N,) or (batch, N).
        bin_edges: Bin edges for histogram (num_bins + 1,).

    Returns:
        Distogram (N, N, num_bins) or (batch, N, N, num_bins).
    """
    # Handle both batched and non-batched inputs
    if positions.ndim == 2:
        positions = positions[np.newaxis, ...]
        mask = mask[np.newaxis, ...]
        squeeze = True
    else:
        squeeze = False

    batch_size, n_pos, _ = positions.shape
    n_bins = len(bin_edges) - 1

    # Compute pairwise distances for all batches at once
    # (batch, N, 1, 3) - (batch, 1, N, 3) -> (batch, N, N, 3)
    diff = positions[:, :, np.newaxis, :] - positions[:, np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff ** 2, axis=-1))  # (batch, N, N)

    # Compute pair mask
    pair_mask = mask[:, :, np.newaxis] * mask[:, np.newaxis, :]  # (batch, N, N)

    # Digitize distances to get bin indices
    # np.digitize returns 0 for values below first edge, n_bins for above last
    bin_indices = np.digitize(distances, bin_edges) - 1  # 0 to n_bins-1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    # Create one-hot encoding using advanced indexing
    distogram = np.zeros((batch_size, n_pos, n_pos, n_bins), dtype=np.float32)
    batch_idx = np.arange(batch_size)[:, np.newaxis, np.newaxis]
    i_idx = np.arange(n_pos)[np.newaxis, :, np.newaxis]
    j_idx = np.arange(n_pos)[np.newaxis, np.newaxis, :]

    # Set the appropriate bin to 1
    np.put_along_axis(
        distogram.reshape(batch_size, n_pos, n_pos, n_bins),
        bin_indices[:, :, :, np.newaxis],
        pair_mask[:, :, :, np.newaxis],
        axis=-1,
    )

    # Apply mask
    distogram = distogram * pair_mask[:, :, :, np.newaxis]

    if squeeze:
        distogram = distogram[0]

    return distogram


def compute_distogram_fast(
    positions: np.ndarray,
    mask: np.ndarray,
    num_bins: int = 39,
    min_dist: float = 3.25,
    max_dist: float = 50.75,
) -> np.ndarray:
    """Compute distogram with optimized binning.

    Args:
        positions: Positions (N, 3) or (batch, N, 3).
        mask: Validity mask (N,) or (batch, N).
        num_bins: Number of distance bins.
        min_dist: Minimum distance.
        max_dist: Maximum distance.

    Returns:
        Distogram (N, N, num_bins) or (batch, N, N, num_bins).
    """
    bin_edges = np.linspace(min_dist, max_dist, num_bins + 1)
    return compute_distogram_vectorized(positions, mask, bin_edges)


# =============================================================================
# Optimized Pair Features
# =============================================================================


def compute_relative_position_vectorized(
    residue_indices: np.ndarray,
    chain_ids: np.ndarray,
    max_relative_idx: int = 32,
) -> np.ndarray:
    """Compute relative position encoding using vectorization.

    Replaces O(n²) nested loop with vectorized operations.

    Args:
        residue_indices: Residue index for each token (N,).
        chain_ids: Chain ID (as integer) for each token (N,).
        max_relative_idx: Maximum relative index before clipping.

    Returns:
        Relative position matrix (N, N).
    """
    n = len(residue_indices)

    # Compute pairwise differences
    diff = residue_indices[np.newaxis, :] - residue_indices[:, np.newaxis]

    # Clip to range
    diff = np.clip(diff, -max_relative_idx, max_relative_idx)

    # Offset to make non-negative
    rel_pos = diff + max_relative_idx

    # Set different chains to special value
    same_chain = chain_ids[:, np.newaxis] == chain_ids[np.newaxis, :]
    rel_pos = np.where(same_chain, rel_pos, 2 * max_relative_idx + 1)

    return rel_pos.astype(np.int32)


def compute_pair_masks_vectorized(
    chain_ids: np.ndarray,
    entity_ids: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute same_chain and same_entity masks vectorized.

    Args:
        chain_ids: Chain ID (as integer) for each token (N,).
        entity_ids: Optional entity ID for each token (N,).

    Returns:
        Tuple of (same_chain, same_entity) masks, each (N, N).
    """
    same_chain = (chain_ids[:, np.newaxis] == chain_ids[np.newaxis, :]).astype(np.float32)

    if entity_ids is not None:
        same_entity = (entity_ids[:, np.newaxis] == entity_ids[np.newaxis, :]).astype(np.float32)
    else:
        same_entity = same_chain.copy()

    return same_chain, same_entity


def compute_unit_vectors_vectorized(
    positions: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Compute unit vectors between all position pairs.

    Args:
        positions: Positions (N, 3) or (batch, N, 3).
        mask: Validity mask (N,) or (batch, N).

    Returns:
        Unit vectors (N, N, 3) or (batch, N, N, 3).
    """
    # Handle batched input
    if positions.ndim == 2:
        positions = positions[np.newaxis, ...]
        mask = mask[np.newaxis, ...]
        squeeze = True
    else:
        squeeze = False

    # Compute direction vectors
    # (batch, N, 1, 3) - (batch, 1, N, 3) -> (batch, N, N, 3)
    diff = positions[:, :, np.newaxis, :] - positions[:, np.newaxis, :, :]

    # Compute distances
    distances = np.sqrt(np.sum(diff ** 2, axis=-1, keepdims=True))
    distances = np.maximum(distances, 1e-8)

    # Normalize
    unit_vectors = diff / distances

    # Apply mask
    pair_mask = mask[:, :, np.newaxis] * mask[:, np.newaxis, :]
    unit_vectors = unit_vectors * pair_mask[:, :, :, np.newaxis]

    if squeeze:
        unit_vectors = unit_vectors[0]

    return unit_vectors.astype(np.float32)


# =============================================================================
# Optimized Contact Detection
# =============================================================================


def detect_contacts_kdtree(
    coords1: np.ndarray,
    coords2: np.ndarray,
    threshold: float = 8.0,
) -> Tuple[List[int], List[int], float]:
    """Detect contacts between two sets of coordinates.

    Uses KD-tree for efficient contact detection.

    Args:
        coords1: First set of coordinates (N, 3).
        coords2: Second set of coordinates (M, 3).
        threshold: Contact distance threshold.

    Returns:
        Tuple of (contact_indices_1, contact_indices_2, min_distance).
    """
    if len(coords1) == 0 or len(coords2) == 0:
        return [], [], float('inf')

    contact_indices_1 = set()
    contact_indices_2 = set()
    min_dist = float('inf')

    if HAS_SCIPY:
        tree2 = cKDTree(coords2)

        for i, coord in enumerate(coords1):
            distances, indices = tree2.query(coord, k=1)
            if distances <= threshold:
                contact_indices_1.add(i)
                contact_indices_2.add(int(indices))
                min_dist = min(min_dist, distances)

            # Also find all neighbors for complete contact list
            all_indices = tree2.query_ball_point(coord, threshold)
            if all_indices:
                contact_indices_1.add(i)
                contact_indices_2.update(all_indices)
    else:
        dist_matrix = compute_pairwise_distances_fast(coords1, coords2)
        min_dist = float(np.min(dist_matrix)) if dist_matrix.size > 0 else float('inf')

        i_indices, j_indices = np.where(dist_matrix <= threshold)
        contact_indices_1 = set(i_indices.tolist())
        contact_indices_2 = set(j_indices.tolist())

    return list(contact_indices_1), list(contact_indices_2), min_dist


def compute_interface_contacts_fast(
    chain_coords: Dict[str, np.ndarray],
    threshold: float = 8.0,
) -> Dict[Tuple[str, str], Tuple[List[int], List[int], float]]:
    """Detect all inter-chain contacts efficiently.

    Uses combined KD-tree for all chains.

    Args:
        chain_coords: Dictionary mapping chain_id to heavy atom coordinates.
        threshold: Contact distance threshold.

    Returns:
        Dictionary mapping (chain1_id, chain2_id) to contact info.
    """
    chain_ids = list(chain_coords.keys())
    contacts = {}

    if not HAS_SCIPY or len(chain_ids) < 2:
        # Fallback to pairwise
        for i, cid1 in enumerate(chain_ids):
            for cid2 in chain_ids[i + 1:]:
                coords1 = chain_coords[cid1]
                coords2 = chain_coords[cid2]
                if len(coords1) > 0 and len(coords2) > 0:
                    c1, c2, min_d = detect_contacts_kdtree(coords1, coords2, threshold)
                    if c1:
                        contacts[(cid1, cid2)] = (c1, c2, min_d)
        return contacts

    # Build combined KD-tree
    all_coords = []
    coord_to_chain = []
    coord_to_idx = []

    for chain_id in chain_ids:
        coords = chain_coords[chain_id]
        for i, coord in enumerate(coords):
            all_coords.append(coord)
            coord_to_chain.append(chain_id)
            coord_to_idx.append(i)

    if not all_coords:
        return contacts

    all_coords_arr = np.array(all_coords)
    tree = cKDTree(all_coords_arr)

    # Find all pairs within threshold
    pairs = tree.query_pairs(threshold, output_type='set')

    # Group by chain pairs
    chain_contacts: Dict[Tuple[str, str], Tuple[Set[int], Set[int], float]] = {}

    for i, j in pairs:
        chain_i = coord_to_chain[i]
        chain_j = coord_to_chain[j]

        if chain_i == chain_j:
            continue

        # Normalize chain pair order
        if chain_i > chain_j:
            chain_i, chain_j = chain_j, chain_i
            i, j = j, i

        key = (chain_i, chain_j)
        dist = np.linalg.norm(all_coords_arr[i] - all_coords_arr[j])

        if key not in chain_contacts:
            chain_contacts[key] = (set(), set(), float('inf'))

        c1, c2, min_d = chain_contacts[key]
        c1.add(coord_to_idx[i] if coord_to_chain[i] == chain_i else coord_to_idx[j])
        c2.add(coord_to_idx[j] if coord_to_chain[j] == chain_j else coord_to_idx[i])
        chain_contacts[key] = (c1, c2, min(min_d, dist))

    # Convert sets to lists
    for key, (c1, c2, min_d) in chain_contacts.items():
        contacts[key] = (list(c1), list(c2), min_d)

    return contacts


# =============================================================================
# Optimized MSA Operations
# =============================================================================


def compute_msa_profile_vectorized(
    msa: np.ndarray,
    num_classes: int = 32,
    pseudocount: float = 1e-8,
) -> np.ndarray:
    """Compute MSA profile using vectorized operations.

    Args:
        msa: MSA array (num_seqs, seq_len) with integer residue types.
        num_classes: Number of residue type classes.
        pseudocount: Pseudocount for smoothing.

    Returns:
        Profile (seq_len, num_classes) with normalized frequencies.
    """
    if msa.size == 0:
        return np.zeros((0, num_classes), dtype=np.float32)

    n_seqs, seq_len = msa.shape

    # Use bincount for each position
    profile = np.zeros((seq_len, num_classes), dtype=np.float32)

    for j in range(seq_len):
        counts = np.bincount(msa[:, j], minlength=num_classes)[:num_classes]
        profile[j] = counts

    # Normalize with pseudocount
    row_sums = profile.sum(axis=1, keepdims=True) + pseudocount
    profile = profile / row_sums

    return profile


def deduplicate_msa_fast(
    msa: np.ndarray,
    threshold: float = 0.9,
) -> np.ndarray:
    """Deduplicate MSA sequences by identity.

    Uses vectorized sequence identity computation.

    Args:
        msa: MSA array (num_seqs, seq_len).
        threshold: Identity threshold for deduplication.

    Returns:
        Deduplicated MSA.
    """
    if len(msa) <= 1:
        return msa

    n_seqs, seq_len = msa.shape
    keep = [True] * n_seqs
    keep[0] = True  # Always keep query

    for i in range(1, n_seqs):
        if not keep[i]:
            continue

        # Compute identity with all remaining sequences
        remaining_indices = [j for j in range(i + 1, n_seqs) if keep[j]]
        if not remaining_indices:
            break

        remaining = msa[remaining_indices]
        query = msa[i:i+1]

        # Vectorized identity computation
        matches = (remaining == query).sum(axis=1)
        identities = matches / seq_len

        # Mark duplicates
        for idx, j in enumerate(remaining_indices):
            if identities[idx] >= threshold:
                keep[j] = False

    return msa[keep]


# =============================================================================
# Utility Functions
# =============================================================================


def batch_compute_distances(
    coords_list: List[np.ndarray],
    batch_size: int = 1000,
) -> List[np.ndarray]:
    """Compute distance matrices in batches to manage memory.

    Args:
        coords_list: List of coordinate arrays.
        batch_size: Number of pairs to process at once.

    Returns:
        List of distance matrices.
    """
    results = []

    for coords in coords_list:
        if len(coords) <= batch_size:
            results.append(compute_pairwise_distances_fast(coords))
        else:
            # Process in batches for large arrays
            n = len(coords)
            dist_matrix = np.zeros((n, n), dtype=np.float32)

            for i in range(0, n, batch_size):
                end_i = min(i + batch_size, n)
                for j in range(0, n, batch_size):
                    end_j = min(j + batch_size, n)
                    dist_matrix[i:end_i, j:end_j] = compute_pairwise_distances_fast(
                        coords[i:end_i], coords[j:end_j]
                    )

            results.append(dist_matrix)

    return results


def precompute_chain_kdtrees(
    chain_coords: Dict[str, np.ndarray],
) -> Dict[str, "cKDTree"]:
    """Pre-build KD-trees for all chains.

    Useful when multiple queries will be performed.

    Args:
        chain_coords: Dictionary mapping chain_id to coordinates.

    Returns:
        Dictionary mapping chain_id to KD-tree.
    """
    if not HAS_SCIPY:
        return {}

    trees = {}
    for chain_id, coords in chain_coords.items():
        if len(coords) > 0:
            trees[chain_id] = cKDTree(coords)

    return trees
