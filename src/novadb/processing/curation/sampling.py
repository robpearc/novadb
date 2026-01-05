"""Dataset sampling and weighting.

Implements the 5-dataset sampling scheme from AlphaFold3 Section 2.5:
1. Weighted PDB (0.5)
2. Disordered PDB distillation (0.02)
3. Protein monomer distillation (0.495)
4. Short protein distillation (0.005)
5. RNA distillation (0.05)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from novadb.data.parsers.structure import Structure, ChainType
from novadb.config import SamplingWeightsConfig


@dataclass
class DatasetEntry:
    """Entry in the training dataset."""
    structure_id: str
    structure_path: str
    cluster_id: Optional[str] = None
    weight: float = 1.0
    dataset_type: str = "pdb"
    resolution: Optional[float] = None
    release_date: Optional[str] = None
    num_chains: int = 0
    num_residues: int = 0
    chain_types: List[str] = field(default_factory=list)
    interface_types: List[str] = field(default_factory=list)


@dataclass
class SamplingConfig:
    """Configuration for dataset sampling.
    
    From AF3 Section 2.5 and Table 4.
    """
    # Dataset weights
    pdb_weight: float = 0.5
    disordered_distillation_weight: float = 0.02
    monomer_distillation_weight: float = 0.495
    short_protein_distillation_weight: float = 0.005
    rna_distillation_weight: float = 0.05

    # Per-sample weights
    # Chain-based sample weight
    sample_weight_chain: bool = True

    # Resolution-based weight (higher weight for better resolution)
    sample_weight_resolution: bool = True
    resolution_weight_max: float = 9.0

    # Interface-based weight
    sample_weight_interface: bool = True
    interface_weight_multiplier: float = 2.0


class DatasetSampler:
    """Sample from training datasets according to AF3 weighting.
    
    From AF3 Section 2.5:
    - 5 datasets with specified weights
    - Inverse-cluster-size weighting for PDB
    - Chain-based weighting
    """

    def __init__(
        self,
        config: Optional[SamplingConfig] = None,
        seed: int = 42,
    ):
        self.config = config or SamplingConfig()
        self.rng = np.random.default_rng(seed)

        # Dataset storage
        self.datasets: Dict[str, List[DatasetEntry]] = {
            "pdb": [],
            "disordered_distillation": [],
            "monomer_distillation": [],
            "short_protein_distillation": [],
            "rna_distillation": [],
        }

        # Cluster information for weighting
        self.cluster_sizes: Dict[str, int] = {}

    def add_entry(
        self,
        entry: DatasetEntry,
        dataset_name: str = "pdb",
    ) -> None:
        """Add an entry to a dataset.
        
        Args:
            entry: Dataset entry to add
            dataset_name: Which dataset to add to
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        entry.dataset_type = dataset_name
        self.datasets[dataset_name].append(entry)

        # Update cluster size tracking
        if entry.cluster_id:
            self.cluster_sizes[entry.cluster_id] = (
                self.cluster_sizes.get(entry.cluster_id, 0) + 1
            )

    def load_cluster_sizes(self, cluster_file: str) -> None:
        """Load cluster size information.
        
        Args:
            cluster_file: Path to cluster size file
                         (TSV: cluster_id, size)
        """
        with open(cluster_file, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    self.cluster_sizes[parts[0]] = int(parts[1])

    def compute_weights(self) -> Dict[str, np.ndarray]:
        """Compute sampling weights for all datasets.
        
        Returns:
            Dictionary of dataset_name -> weight array
        """
        weights = {}

        for name, entries in self.datasets.items():
            if not entries:
                weights[name] = np.array([])
                continue

            entry_weights = np.ones(len(entries), dtype=np.float64)

            for i, entry in enumerate(entries):
                w = 1.0

                # Inverse cluster size weighting (for PDB)
                if name == "pdb" and entry.cluster_id:
                    cluster_size = self.cluster_sizes.get(entry.cluster_id, 1)
                    w *= 1.0 / cluster_size

                # Chain-based weighting
                if self.config.sample_weight_chain:
                    w *= self._compute_chain_weight(entry)

                # Resolution-based weighting
                if self.config.sample_weight_resolution and entry.resolution:
                    w *= self._compute_resolution_weight(entry.resolution)

                # Interface-based weighting
                if self.config.sample_weight_interface:
                    w *= self._compute_interface_weight(entry)

                entry_weights[i] = w

            # Normalize
            if entry_weights.sum() > 0:
                entry_weights /= entry_weights.sum()

            weights[name] = entry_weights

        return weights

    def _compute_chain_weight(self, entry: DatasetEntry) -> float:
        """Compute weight based on chain composition.
        
        From AF3: Weight by chain composition to balance
        different molecular types.
        """
        # Simplified: weight by number of interface types
        if entry.interface_types:
            return 1.0 + 0.5 * len(entry.interface_types)
        return 1.0

    def _compute_resolution_weight(self, resolution: float) -> float:
        """Compute weight based on resolution.
        
        Higher weight for better (lower) resolution.
        """
        max_res = self.config.resolution_weight_max
        if resolution >= max_res:
            return 0.1
        # Linear scaling: 1.0 at 0Å, 0.1 at max_res
        return 1.0 - 0.9 * (resolution / max_res)

    def _compute_interface_weight(self, entry: DatasetEntry) -> float:
        """Compute weight based on interface presence.
        
        From AF3: Upweight structures with meaningful interfaces.
        """
        if "protein-ligand" in entry.interface_types:
            return self.config.interface_weight_multiplier
        if "protein-nucleic" in entry.interface_types:
            return self.config.interface_weight_multiplier
        return 1.0

    def sample(
        self,
        batch_size: int = 1,
    ) -> List[DatasetEntry]:
        """Sample entries from datasets.
        
        Args:
            batch_size: Number of entries to sample
            
        Returns:
            List of sampled entries
        """
        # First, select which dataset to sample from
        dataset_weights = np.array([
            self.config.pdb_weight,
            self.config.disordered_distillation_weight,
            self.config.monomer_distillation_weight,
            self.config.short_protein_distillation_weight,
            self.config.rna_distillation_weight,
        ])
        dataset_names = list(self.datasets.keys())

        # Filter to non-empty datasets
        non_empty = []
        non_empty_weights = []
        for i, name in enumerate(dataset_names):
            if self.datasets[name]:
                non_empty.append(name)
                non_empty_weights.append(dataset_weights[i])

        if not non_empty:
            return []

        non_empty_weights = np.array(non_empty_weights)
        non_empty_weights /= non_empty_weights.sum()

        # Compute per-entry weights
        entry_weights = self.compute_weights()

        samples = []
        for _ in range(batch_size):
            # Sample dataset
            dataset_idx = self.rng.choice(len(non_empty), p=non_empty_weights)
            dataset_name = non_empty[dataset_idx]

            # Sample entry from dataset
            weights = entry_weights[dataset_name]
            if len(weights) == 0:
                continue

            entry_idx = self.rng.choice(len(weights), p=weights)
            samples.append(self.datasets[dataset_name][entry_idx])

        return samples

    def sample_balanced(
        self,
        batch_size: int,
        interface_types: Optional[List[str]] = None,
    ) -> List[DatasetEntry]:
        """Sample with balanced interface type representation.
        
        Args:
            batch_size: Number of entries to sample
            interface_types: Specific interface types to balance
            
        Returns:
            List of sampled entries
        """
        if interface_types is None:
            return self.sample(batch_size)

        # Group entries by interface type
        type_entries: Dict[str, List[Tuple[str, int]]] = {
            t: [] for t in interface_types
        }

        for dataset_name, entries in self.datasets.items():
            for i, entry in enumerate(entries):
                for itype in entry.interface_types:
                    if itype in type_entries:
                        type_entries[itype].append((dataset_name, i))

        # Sample equally from each type
        samples_per_type = batch_size // len(interface_types)
        samples = []

        for itype, candidates in type_entries.items():
            if not candidates:
                continue

            n_samples = min(samples_per_type, len(candidates))
            indices = self.rng.choice(len(candidates), size=n_samples, replace=False)

            for idx in indices:
                dataset_name, entry_idx = candidates[idx]
                samples.append(self.datasets[dataset_name][entry_idx])

        return samples

    def get_dataset_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for each dataset.
        
        Returns:
            Dictionary of dataset_name -> statistics
        """
        stats = {}

        for name, entries in self.datasets.items():
            if not entries:
                stats[name] = {"count": 0}
                continue

            resolutions = [e.resolution for e in entries if e.resolution]
            num_residues = [e.num_residues for e in entries]

            stats[name] = {
                "count": len(entries),
                "mean_resolution": np.mean(resolutions) if resolutions else 0,
                "mean_residues": np.mean(num_residues),
                "median_residues": np.median(num_residues),
                "num_clusters": len(set(e.cluster_id for e in entries if e.cluster_id)),
            }

        return stats


def create_distillation_entry(
    structure: Structure,
    prediction_path: str,
    source_model: str = "alphafold2",
) -> DatasetEntry:
    """Create a distillation dataset entry.
    
    From AF3 Section 2.5: Distillation data is generated by
    predicting structures with AF2 or similar.
    
    Args:
        structure: Template structure for metadata
        prediction_path: Path to predicted structure
        source_model: Model used for prediction
        
    Returns:
        DatasetEntry for distillation dataset
    """
    chain_types = [
        chain.chain_type.name if chain.chain_type else "unknown"
        for chain in structure.chains.values()
    ]

    total_residues = sum(
        len(chain.residues) for chain in structure.chains.values()
    )

    return DatasetEntry(
        structure_id=f"{structure.pdb_id}_distill_{source_model}",
        structure_path=prediction_path,
        cluster_id=None,  # Distillation doesn't use clustering
        weight=1.0,
        dataset_type="distillation",
        resolution=None,  # Predictions don't have resolution
        release_date=None,
        num_chains=len(structure.chains),
        num_residues=total_residues,
        chain_types=chain_types,
        interface_types=[],
    )
