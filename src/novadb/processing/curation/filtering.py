"""Dataset curation and filtering.

Implements data filtering from AlphaFold3 Section 2.5:
- Date-based filtering (training cutoff: 2021-09-30)
- Resolution filtering (≤9Å for training)
- Chain composition filtering
- Exclusion lists (crystallization aids, common ligands)
- Max 20 chains cropping for large bioassemblies (Section 2.5.4)
- Clash detection (>30% atoms within 1.7Å)
- Consecutive Cα distance filtering (>10Å indicates break)

Performance optimizations:
- Uses KD-tree for O(n log n) clash detection instead of O(n²)
- Vectorized chain contact detection
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from novadb.data.parsers.structure import (
    Structure,
    Chain,
    ChainType,
    CRYSTALLIZATION_AIDS,
    LIGAND_EXCLUSION_LIST,
    ION_CCD_CODES,
)
from novadb.config import FilteringConfig, DateConfig
from novadb.processing.optimized import (
    detect_clashing_chains_fast,
    compute_interface_contacts_fast,
)

# AF3 Section 2.5.4 constants
MAX_CHAINS_AFTER_CROPPING = 20
CLASH_DISTANCE_THRESHOLD = 1.7  # Å
CLASH_FRACTION_THRESHOLD = 0.30  # 30% of atoms
MAX_CA_DISTANCE = 10.0  # Å for consecutive Cα atoms


@dataclass
class FilterResult:
    """Result of filtering operation."""
    passed: bool
    reason: str = ""
    metrics: Dict[str, float] = None

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}


class StructureFilter:
    """Filter structures based on AF3 criteria.
    
    From AF3 Section 2.5:
    - Training data cutoff: 2021-09-30
    - Resolution ≤9Å for X-ray, electron microscopy, neutron
    - Only polymer chains (protein, nucleic acid) required
    - Exclude certain crystallization aids and common ligands
    """

    def __init__(
        self,
        filtering_config: Optional[FilteringConfig] = None,
        date_config: Optional[DateConfig] = None,
    ):
        self.filtering = filtering_config or FilteringConfig()
        self.dates = date_config or DateConfig()

        # Build exclusion sets
        self.crystallization_aids = CRYSTALLIZATION_AIDS
        self.excluded_ligands = LIGAND_EXCLUSION_LIST

    def filter(self, structure: Structure) -> FilterResult:
        """Apply all filters to a structure.
        
        Args:
            structure: Structure to filter
            
        Returns:
            FilterResult indicating pass/fail and reason
        """
        # Check release date
        result = self._check_date(structure)
        if not result.passed:
            return result

        # Check resolution
        result = self._check_resolution(structure)
        if not result.passed:
            return result

        # Check chain composition
        result = self._check_chain_composition(structure)
        if not result.passed:
            return result

        # Check minimum size
        result = self._check_minimum_size(structure)
        if not result.passed:
            return result

        return FilterResult(
            passed=True,
            reason="passed all filters",
            metrics=self._compute_metrics(structure),
        )

    def _check_date(self, structure: Structure) -> FilterResult:
        """Check if structure is within date cutoff."""
        if structure.release_date is None:
            return FilterResult(
                passed=False,
                reason="missing release date",
            )

        try:
            if isinstance(structure.release_date, str):
                release = datetime.strptime(
                    structure.release_date, "%Y-%m-%d"
                ).date()
            else:
                release = structure.release_date
        except ValueError:
            return FilterResult(
                passed=False,
                reason=f"invalid date format: {structure.release_date}",
            )

        if release > self.dates.training_cutoff:
            return FilterResult(
                passed=False,
                reason=f"release date {release} after cutoff {self.dates.training_cutoff}",
            )

        return FilterResult(passed=True)

    def _check_resolution(self, structure: Structure) -> FilterResult:
        """Check if resolution is acceptable."""
        if structure.resolution is None:
            # NMR structures may not have resolution
            if structure.method and "NMR" in structure.method.upper():
                return FilterResult(passed=True)
            return FilterResult(
                passed=False,
                reason="missing resolution",
            )

        if structure.resolution > self.filtering.max_resolution:
            return FilterResult(
                passed=False,
                reason=f"resolution {structure.resolution}Å > {self.filtering.max_resolution}Å",
            )

        return FilterResult(passed=True)

    def _check_chain_composition(self, structure: Structure) -> FilterResult:
        """Check if structure has required chain types."""
        chain_types = set()
        for chain in structure.chains.values():
            chain_types.add(chain.chain_type)

        # Must have at least one polymer chain
        polymer_types = {ChainType.PROTEIN, ChainType.RNA, ChainType.DNA}
        if not chain_types.intersection(polymer_types):
            return FilterResult(
                passed=False,
                reason="no polymer chains found",
            )

        return FilterResult(passed=True)

    def _check_minimum_size(self, structure: Structure) -> FilterResult:
        """Check if structure meets minimum size requirements."""
        total_residues = 0
        for chain in structure.chains.values():
            total_residues += len(chain.residues)

        if total_residues < self.filtering.min_residues:
            return FilterResult(
                passed=False,
                reason=f"too few residues: {total_residues} < {self.filtering.min_residues}",
            )

        return FilterResult(passed=True)

    def _compute_metrics(self, structure: Structure) -> Dict[str, float]:
        """Compute metrics for the structure."""
        metrics = {}

        # Count chains by type
        type_counts = {}
        total_residues = 0
        for chain in structure.chains.values():
            chain_type = chain.chain_type.name if chain.chain_type else "unknown"
            type_counts[chain_type] = type_counts.get(chain_type, 0) + 1
            total_residues += len(chain.residues)

        metrics["num_chains"] = len(structure.chains)
        metrics["num_residues"] = total_residues
        metrics["resolution"] = structure.resolution or 0.0

        for ctype, count in type_counts.items():
            metrics[f"num_{ctype.lower()}_chains"] = count

        return metrics

    def filter_ligands(self, structure: Structure) -> Structure:
        """Filter out crystallization aids and excluded ligands.
        
        From AF3 Section 2.5: Remove crystallization aids (Table 9)
        and certain common ligands (Table 10).
        
        Args:
            structure: Structure to filter
            
        Returns:
            New structure with filtered ligands
        """
        filtered_chains = {}

        for chain_id, chain in structure.chains.items():
            if chain.chain_type == ChainType.LIGAND:
                # Check if all residues are excluded
                excluded = True
                for residue in chain.residues:
                    if residue.name not in self.crystallization_aids:
                        if residue.name not in self.excluded_ligands:
                            excluded = False
                            break

                if excluded:
                    continue

            filtered_chains[chain_id] = chain

        return Structure(
            pdb_id=structure.pdb_id,
            chains=filtered_chains,
            release_date=structure.release_date,
            resolution=structure.resolution,
            method=structure.method,
            bonds=structure.bonds,
        )

    def filter_by_interface(
        self,
        structure: Structure,
        min_interface_residues: int = 1,
    ) -> bool:
        """Check if structure has meaningful interfaces.

        For training on protein-ligand, protein-nucleic acid, etc.
        interactions.

        Args:
            structure: Structure to check
            min_interface_residues: Minimum interface residues

        Returns:
            True if structure has interfaces
        """
        interfaces = structure.get_interfaces(threshold=5.0)  # AF3 uses 5Å
        return len(interfaces) >= min_interface_residues

    def crop_to_max_chains(
        self,
        structure: Structure,
        max_chains: int = MAX_CHAINS_AFTER_CROPPING,
        seed: Optional[int] = None,
    ) -> Structure:
        """Crop structure to maximum number of chains.

        From AF3 Section 2.5.4: For bioassemblies with >20 chains, crop
        to 20 chains while preserving inter-chain contacts.

        The algorithm:
        1. Start with a random seed chain
        2. Iteratively add chains that have contacts with already-selected chains
        3. Stop when max_chains is reached or no more contacting chains

        Args:
            structure: Structure to crop
            max_chains: Maximum number of chains (default 20)
            seed: Random seed for reproducibility

        Returns:
            New structure with at most max_chains chains
        """
        if len(structure.chains) <= max_chains:
            return structure

        rng = np.random.default_rng(seed)
        chain_ids = list(structure.chains.keys())

        # Find all chain-chain contacts
        contacts = self._get_chain_contacts(structure)

        # Start with a random chain
        selected = [rng.choice(chain_ids)]
        remaining = set(chain_ids) - set(selected)

        # Greedily add chains that contact already-selected chains
        while len(selected) < max_chains and remaining:
            best_chain = None
            best_contacts = 0

            for chain_id in remaining:
                num_contacts = sum(
                    1 for sel in selected
                    if (sel, chain_id) in contacts or (chain_id, sel) in contacts
                )
                if num_contacts > best_contacts:
                    best_contacts = num_contacts
                    best_chain = chain_id

            if best_chain is None:
                # No more contacting chains, pick random
                best_chain = rng.choice(list(remaining))

            selected.append(best_chain)
            remaining.remove(best_chain)

        # Build new structure with selected chains
        new_chains = {cid: structure.chains[cid] for cid in selected}

        return Structure(
            pdb_id=structure.pdb_id,
            chains=new_chains,
            release_date=structure.release_date,
            resolution=structure.resolution,
            method=structure.method,
            bonds=[b for b in structure.bonds
                   if b.chain1_id in selected and b.chain2_id in selected],
        )

    def _get_chain_contacts(
        self,
        structure: Structure,
        threshold: float = 5.0,
    ) -> Set[Tuple[str, str]]:
        """Get set of chain pairs that have contacts within threshold.

        Uses optimized KD-tree based contact detection.
        """
        chain_ids = list(structure.chains.keys())

        # Collect coordinates for all chains
        chain_coords = {}
        for chain_id in chain_ids:
            chain = structure.chains[chain_id]
            coords = chain.get_all_heavy_atom_coords()
            chain_coords[chain_id] = coords if len(coords) > 0 else np.zeros((0, 3))

        # Use optimized interface contact detection
        interface_contacts = compute_interface_contacts_fast(chain_coords, threshold)

        # Convert to set of chain pairs
        contacts = set(interface_contacts.keys())

        return contacts

    def remove_clashing_chains(
        self,
        structure: Structure,
        distance_threshold: float = CLASH_DISTANCE_THRESHOLD,
        fraction_threshold: float = CLASH_FRACTION_THRESHOLD,
    ) -> Structure:
        """Remove chains with excessive atomic clashes.

        From AF3 Section 2.5.4: Remove chains where >30% of atoms are
        within 1.7Å of atoms from other chains.

        Uses KD-tree for O(n log n) clash detection instead of O(n²).

        Args:
            structure: Structure to filter
            distance_threshold: Distance for clash detection (1.7Å)
            fraction_threshold: Fraction of atoms that constitutes clash (0.30)

        Returns:
            New structure with clashing chains removed
        """
        chain_ids = list(structure.chains.keys())

        # Collect coordinates for all chains
        chain_coords = {}
        for chain_id in chain_ids:
            chain = structure.chains[chain_id]
            coords = chain.get_all_heavy_atom_coords()
            chain_coords[chain_id] = coords if len(coords) > 0 else np.zeros((0, 3))

        # Use optimized KD-tree based clash detection
        chains_to_remove = detect_clashing_chains_fast(
            chain_coords,
            distance_threshold=distance_threshold,
            fraction_threshold=fraction_threshold,
        )

        chains_to_keep = [cid for cid in chain_ids if cid not in chains_to_remove]

        # Build new structure
        new_chains = {cid: structure.chains[cid] for cid in chains_to_keep}

        return Structure(
            pdb_id=structure.pdb_id,
            chains=new_chains,
            release_date=structure.release_date,
            resolution=structure.resolution,
            method=structure.method,
            bonds=[b for b in structure.bonds
                   if b.chain1_id in chains_to_keep and b.chain2_id in chains_to_keep],
        )

    def filter_broken_chains(
        self,
        structure: Structure,
        max_ca_distance: float = MAX_CA_DISTANCE,
    ) -> Structure:
        """Filter protein chains with broken backbone.

        From AF3 Section 2.5.4: Remove protein chains where consecutive
        Cα atoms are >10Å apart (indicating missing residues or breaks).

        Args:
            structure: Structure to filter
            max_ca_distance: Maximum allowed Cα-Cα distance (10Å)

        Returns:
            New structure with broken chains removed
        """
        chains_to_keep = []

        for chain_id, chain in structure.chains.items():
            # Non-protein chains are kept
            if chain.chain_type != ChainType.PROTEIN:
                chains_to_keep.append(chain_id)
                continue

            # Check Cα-Cα distances
            is_valid = True
            prev_ca = None

            for residue in chain.residues:
                ca = residue.get_ca()
                if ca is None:
                    continue

                if prev_ca is not None:
                    distance = np.linalg.norm(ca.coords - prev_ca.coords)
                    if distance > max_ca_distance:
                        is_valid = False
                        break

                prev_ca = ca

            if is_valid:
                chains_to_keep.append(chain_id)

        # Build new structure
        new_chains = {cid: structure.chains[cid] for cid in chains_to_keep}

        return Structure(
            pdb_id=structure.pdb_id,
            chains=new_chains,
            release_date=structure.release_date,
            resolution=structure.resolution,
            method=structure.method,
            bonds=[b for b in structure.bonds
                   if b.chain1_id in chains_to_keep and b.chain2_id in chains_to_keep],
        )


class ClusterFilter:
    """Filter based on sequence clustering.
    
    From AF3 Section 2.5:
    - Protein chains: 40% sequence identity clustering
    - Nucleic acids: 100% sequence identity clustering
    """

    def __init__(
        self,
        protein_threshold: float = 0.4,
        nucleic_acid_threshold: float = 1.0,
    ):
        self.protein_threshold = protein_threshold
        self.nucleic_acid_threshold = nucleic_acid_threshold
        self.cluster_assignments: Dict[str, str] = {}

    def load_clusters(self, cluster_file: str) -> None:
        """Load pre-computed cluster assignments.
        
        Args:
            cluster_file: Path to cluster assignment file
                         (TSV: sequence_id, cluster_id)
        """
        with open(cluster_file, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    self.cluster_assignments[parts[0]] = parts[1]

    def get_cluster(self, sequence_id: str) -> Optional[str]:
        """Get cluster ID for a sequence."""
        return self.cluster_assignments.get(sequence_id)

    def filter_by_cluster(
        self,
        structures: List[Structure],
        target_cluster: str,
    ) -> List[Structure]:
        """Filter structures to exclude those in target cluster.
        
        Used for test set holdout.
        
        Args:
            structures: List of structures
            target_cluster: Cluster to exclude
            
        Returns:
            Filtered list of structures
        """
        filtered = []
        for structure in structures:
            exclude = False
            for chain_id, chain in structure.chains.items():
                seq_id = f"{structure.pdb_id}_{chain_id}"
                cluster = self.get_cluster(seq_id)
                if cluster == target_cluster:
                    exclude = True
                    break

            if not exclude:
                filtered.append(structure)

        return filtered


class InterfaceFilter:
    """Filter structures based on interface properties.

    From AF3 Section 2.5.1: Filter for meaningful biological interfaces.
    Interfaces defined by minimum heavy atom separation < 5Å.
    """

    def __init__(
        self,
        min_interface_area: float = 100.0,  # Å²
        min_contacts: int = 5,
        contact_distance: float = 5.0,  # Å (AF3 Section 2.5.1)
    ):
        self.min_interface_area = min_interface_area
        self.min_contacts = min_contacts
        self.contact_distance = contact_distance

    def has_valid_interface(
        self,
        structure: Structure,
        chain_types: Optional[Tuple[ChainType, ChainType]] = None,
    ) -> bool:
        """Check if structure has valid interfaces.
        
        Args:
            structure: Structure to check
            chain_types: Optional specific interface types to require
            
        Returns:
            True if valid interfaces exist
        """
        interfaces = structure.get_interfaces(
            threshold=self.contact_distance
        )

        if not interfaces:
            return False

        if chain_types is not None:
            # Check for specific interface type
            type1, type2 = chain_types
            for chain1_id, chain2_id in interfaces:
                chain1 = structure.chains[chain1_id]
                chain2 = structure.chains[chain2_id]
                if (
                    (chain1.chain_type == type1 and chain2.chain_type == type2)
                    or (chain1.chain_type == type2 and chain2.chain_type == type1)
                ):
                    if len(interfaces[(chain1_id, chain2_id)]) >= self.min_contacts:
                        return True
            return False

        # Check any interface has enough contacts
        for (chain1_id, chain2_id), contact_pairs in interfaces.items():
            if len(contact_pairs) >= self.min_contacts:
                return True

        return False

    def get_interface_chains(
        self,
        structure: Structure,
    ) -> List[Tuple[str, str, int]]:
        """Get list of interface chain pairs with contact counts.
        
        Args:
            structure: Structure to analyze
            
        Returns:
            List of (chain1_id, chain2_id, num_contacts)
        """
        interfaces = structure.get_interfaces(
            threshold=self.contact_distance
        )

        result = []
        for (chain1_id, chain2_id), contact_pairs in interfaces.items():
            result.append((chain1_id, chain2_id, len(contact_pairs)))

        return sorted(result, key=lambda x: -x[2])
