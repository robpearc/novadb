"""Curation transforms for dataset processing.

Implements the transform-based curation pipeline from AlphaFold3 Section 2.5.
Transforms operate at different levels:
- Per-chain: Filter/process individual chains
- Per-structure: Filter/process complete structures
- Per-sequence: Clustering operations
- Batch: Sampling and weighting
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import numpy as np

from novadb.data.parsers.structure import (
    Structure,
    Chain,
    ChainType,
    CRYSTALLIZATION_AIDS,
    LIGAND_EXCLUSION_LIST,
)


@dataclass
class CurationConfig:
    """Configuration for curation transforms.
    
    From AF3 Section 2.5 and Table 4.
    """
    # Date cutoffs
    training_cutoff: date = field(default_factory=lambda: date(2021, 9, 30))
    template_cutoff: date = field(default_factory=lambda: date(2021, 9, 30))
    
    # Resolution
    max_resolution: float = 9.0
    
    # Chain length limits
    min_chain_length: int = 4
    max_chain_length: int = 10000
    
    # Minimum residues per structure
    min_residues: int = 10
    
    # Clustering thresholds
    protein_identity_threshold: float = 0.4
    nucleic_identity_threshold: float = 1.0
    peptide_identity_threshold: float = 1.0
    ligand_cluster_by_ccd: bool = True
    
    # Dataset weights (Table 4)
    pdb_weight: float = 0.5
    disordered_distillation_weight: float = 0.02
    monomer_distillation_weight: float = 0.495
    short_protein_weight: float = 0.005
    rna_distillation_weight: float = 0.05
    
    # Interface requirements
    min_interface_contacts: int = 5
    interface_distance_threshold: float = 8.0


# =============================================================================
# Base Transform Classes
# =============================================================================

class BaseTransform(ABC):
    """Base class for all transforms."""
    
    @abstractmethod
    def __call__(self, data: Any) -> Any:
        """Apply the transform."""
        pass
    
    @property
    def name(self) -> str:
        """Name of the transform."""
        return self.__class__.__name__


class ChainTransform(BaseTransform):
    """Transform that operates on individual chains."""
    
    @abstractmethod
    def __call__(self, chain: Chain) -> Optional[Chain]:
        """Apply transform to a chain.
        
        Returns None if chain should be filtered out.
        """
        pass


class StructureTransform(BaseTransform):
    """Transform that operates on complete structures."""
    
    @abstractmethod
    def __call__(self, structure: Structure) -> Optional[Structure]:
        """Apply transform to a structure.
        
        Returns None if structure should be filtered out.
        """
        pass


class BatchTransform(BaseTransform):
    """Transform that operates on batches of structures."""
    
    @abstractmethod
    def __call__(self, structures: List[Structure]) -> List[Structure]:
        """Apply transform to a batch of structures."""
        pass


# =============================================================================
# Per-Chain Transforms
# =============================================================================

class ChainLengthFilterTransform(ChainTransform):
    """Filter chains by length.
    
    From AF3 Section 2.5: Filter short chains.
    """
    
    def __init__(
        self,
        min_length: int = 4,
        max_length: int = 10000,
    ):
        self.min_length = min_length
        self.max_length = max_length
    
    def __call__(self, chain: Chain) -> Optional[Chain]:
        length = len(chain.residues)
        if length < self.min_length or length > self.max_length:
            return None
        return chain


class ChainSequenceExtractTransform(ChainTransform):
    """Extract sequence from chain for clustering."""
    
    def __init__(self):
        self.sequences: Dict[str, str] = {}
    
    def __call__(self, chain: Chain) -> Optional[Chain]:
        seq = chain.sequence
        if seq:
            self.sequences[chain.chain_id] = seq
        return chain
    
    def get_sequences(self) -> Dict[str, str]:
        """Get all extracted sequences."""
        return self.sequences


class ChainTypeFilterTransform(ChainTransform):
    """Filter chains by type.
    
    Args:
        allowed_types: Set of allowed chain types
        required_types: At least one chain of these types must exist
    """
    
    def __init__(
        self,
        allowed_types: Optional[Set[ChainType]] = None,
        exclude_types: Optional[Set[ChainType]] = None,
    ):
        self.allowed_types = allowed_types
        self.exclude_types = exclude_types or set()
    
    def __call__(self, chain: Chain) -> Optional[Chain]:
        if chain.chain_type in self.exclude_types:
            return None
        if self.allowed_types and chain.chain_type not in self.allowed_types:
            return None
        return chain


# =============================================================================
# Per-Structure Transforms
# =============================================================================

class DateFilterTransform(StructureTransform):
    """Filter structures by release date.
    
    From AF3 Section 2.5: Training cutoff 2021-09-30.
    """
    
    def __init__(
        self,
        cutoff_date: Union[date, str],
        mode: str = "before",  # "before", "after", "on"
    ):
        if isinstance(cutoff_date, str):
            cutoff_date = datetime.strptime(cutoff_date, "%Y-%m-%d").date()
        self.cutoff_date = cutoff_date
        self.mode = mode
    
    def __call__(self, structure: Structure) -> Optional[Structure]:
        if structure.release_date is None:
            return None
        
        if isinstance(structure.release_date, str):
            try:
                release = datetime.strptime(
                    structure.release_date, "%Y-%m-%d"
                ).date()
            except ValueError:
                return None
        else:
            release = structure.release_date
        
        if self.mode == "before":
            if release > self.cutoff_date:
                return None
        elif self.mode == "after":
            if release < self.cutoff_date:
                return None
        elif self.mode == "on":
            if release != self.cutoff_date:
                return None
        
        return structure


class ResolutionFilterTransform(StructureTransform):
    """Filter structures by resolution.
    
    From AF3 Section 2.5: Resolution ≤9Å for training.
    """
    
    def __init__(
        self,
        max_resolution: float = 9.0,
        allow_nmr: bool = True,
    ):
        self.max_resolution = max_resolution
        self.allow_nmr = allow_nmr
    
    def __call__(self, structure: Structure) -> Optional[Structure]:
        # NMR structures may not have resolution
        if structure.resolution is None:
            if self.allow_nmr and structure.method:
                if "NMR" in structure.method.upper():
                    return structure
            return None
        
        if structure.resolution > self.max_resolution:
            return None
        
        return structure


class ChainCompositionFilterTransform(StructureTransform):
    """Filter by chain composition.
    
    From AF3 Section 2.5: Require at least one polymer chain.
    """
    
    def __init__(
        self,
        require_protein: bool = False,
        require_nucleic: bool = False,
        require_ligand: bool = False,
        min_chains: int = 1,
        max_chains: int = 1000,
    ):
        self.require_protein = require_protein
        self.require_nucleic = require_nucleic
        self.require_ligand = require_ligand
        self.min_chains = min_chains
        self.max_chains = max_chains
    
    def __call__(self, structure: Structure) -> Optional[Structure]:
        chain_types = set()
        for chain in structure.chains.values():
            if chain.chain_type:
                chain_types.add(chain.chain_type)
        
        # Check chain count
        num_chains = len(structure.chains)
        if num_chains < self.min_chains or num_chains > self.max_chains:
            return None
        
        # Must have at least one polymer chain
        polymer_types = {ChainType.PROTEIN, ChainType.RNA, ChainType.DNA}
        if not chain_types.intersection(polymer_types):
            return None
        
        # Check specific requirements
        if self.require_protein and ChainType.PROTEIN not in chain_types:
            return None
        if self.require_nucleic:
            if ChainType.RNA not in chain_types and ChainType.DNA not in chain_types:
                return None
        if self.require_ligand and ChainType.LIGAND not in chain_types:
            return None
        
        return structure


class LeavingAtomsRemovalTransform(StructureTransform):
    """Remove leaving atoms from covalent ligands.

    From AF3 Section 2.6: When a ligand forms a covalent bond to a polymer,
    the "leaving atoms" (typically hydrogen or hydroxyl groups) are removed
    from both the ligand and the polymer.

    This is determined from the CCD (Chemical Component Dictionary) which
    defines which atoms are leaving groups for each covalent modification.
    """

    # Common leaving atoms for covalent modifications
    # These are atoms that leave when a covalent bond forms
    COMMON_LEAVING_ATOMS: Set[str] = {
        # Hydrogen atoms that leave during bond formation
        "H", "HG", "HG1", "HN", "HO", "HE", "HZ",
        # Hydroxyl groups that may leave
        "OXT",  # C-terminal oxygen
    }

    # Mapping of bond type to leaving atom patterns
    # Format: (residue_name, atom_name) -> leaving atoms on the ligand side
    LEAVING_ATOM_PATTERNS: Dict[Tuple[str, str], Set[str]] = {
        # Cysteine SG modifications (e.g., drug conjugates)
        ("CYS", "SG"): {"HG"},
        # Serine/Threonine modifications
        ("SER", "OG"): {"HG"},
        ("THR", "OG1"): {"HG1"},
        # Lysine modifications
        ("LYS", "NZ"): {"HZ1", "HZ2", "HZ3"},
        # N-terminal modifications
        ("ALA", "N"): {"H", "H2", "H3"},
        ("GLY", "N"): {"H", "H2", "H3"},
    }

    def __init__(
        self,
        ccd_leaving_atoms: Optional[Dict[str, Set[str]]] = None,
    ):
        """Initialize the transform.

        Args:
            ccd_leaving_atoms: Optional dictionary mapping CCD codes to
                their leaving atoms. If not provided, uses common patterns.
        """
        self.ccd_leaving_atoms = ccd_leaving_atoms or {}

    def __call__(self, structure: Structure) -> Optional[Structure]:
        """Remove leaving atoms from covalent bonds.

        Args:
            structure: Structure to process

        Returns:
            Structure with leaving atoms removed
        """
        if not structure.bonds:
            return structure

        # Find atoms to remove based on covalent bonds
        atoms_to_remove: Dict[str, Set[str]] = {}  # chain_id -> set of atom_names

        for bond in structure.bonds:
            # Check if this is a cross-chain or ligand-polymer bond
            chain1 = structure.chains.get(bond.chain1_id)
            chain2 = structure.chains.get(bond.chain2_id)

            if chain1 is None or chain2 is None:
                continue

            # Find leaving atoms for each side of the bond
            leaving1 = self._get_leaving_atoms(
                chain1, bond.res1_seq_id, bond.atom1_name
            )
            leaving2 = self._get_leaving_atoms(
                chain2, bond.res2_seq_id, bond.atom2_name
            )

            # Add to removal set
            if leaving1:
                key = (bond.chain1_id, bond.res1_seq_id)
                if key not in atoms_to_remove:
                    atoms_to_remove[key] = set()
                atoms_to_remove[key].update(leaving1)

            if leaving2:
                key = (bond.chain2_id, bond.res2_seq_id)
                if key not in atoms_to_remove:
                    atoms_to_remove[key] = set()
                atoms_to_remove[key].update(leaving2)

        # If no atoms to remove, return original structure
        if not atoms_to_remove:
            return structure

        # Create new structure with leaving atoms removed
        new_chains = {}
        for chain_id, chain in structure.chains.items():
            new_residues = []
            for residue in chain.residues:
                key = (chain_id, residue.seq_id)
                if key in atoms_to_remove:
                    # Remove specified atoms
                    new_atoms = {
                        name: atom
                        for name, atom in residue.atoms.items()
                        if name not in atoms_to_remove[key]
                    }
                    from novadb.data.parsers.structure import Residue
                    new_residue = Residue(
                        name=residue.name,
                        seq_id=residue.seq_id,
                        atoms=new_atoms,
                        insertion_code=residue.insertion_code,
                        is_standard=residue.is_standard,
                    )
                    new_residues.append(new_residue)
                else:
                    new_residues.append(residue)

            from novadb.data.parsers.structure import Chain as ChainClass
            new_chain = ChainClass(
                chain_id=chain.chain_id,
                residues=new_residues,
                entity_id=chain.entity_id,
                chain_type=chain.chain_type,
            )
            new_chains[chain_id] = new_chain

        return Structure(
            pdb_id=structure.pdb_id,
            chains=new_chains,
            release_date=structure.release_date,
            resolution=structure.resolution,
            method=structure.method,
            bonds=structure.bonds,
        )

    def _get_leaving_atoms(
        self,
        chain: Chain,
        res_seq_id: int,
        bond_atom_name: str,
    ) -> Set[str]:
        """Get leaving atoms for a specific bonding atom.

        Args:
            chain: Chain containing the residue
            res_seq_id: Residue sequence ID
            bond_atom_name: Name of the atom forming the bond

        Returns:
            Set of atom names to remove
        """
        leaving = set()

        # Find the residue
        residue = None
        for res in chain.residues:
            if res.seq_id == res_seq_id:
                residue = res
                break

        if residue is None:
            return leaving

        # Check CCD-defined leaving atoms
        if residue.name in self.ccd_leaving_atoms:
            leaving.update(self.ccd_leaving_atoms[residue.name])
            return leaving

        # Check pattern-based leaving atoms
        pattern_key = (residue.name, bond_atom_name)
        if pattern_key in self.LEAVING_ATOM_PATTERNS:
            leaving.update(self.LEAVING_ATOM_PATTERNS[pattern_key])
            return leaving

        # For ligands, look for common leaving atoms
        if chain.chain_type == ChainType.LIGAND:
            # Find hydrogen atoms bonded to the bonding atom
            # This is a simplified heuristic - real implementation would
            # use CCD bond information
            for atom_name, atom in residue.atoms.items():
                if atom.is_hydrogen and atom_name.startswith(bond_atom_name[0]):
                    leaving.add(atom_name)

        return leaving


class ExclusionListFilterTransform(StructureTransform):
    """Remove crystallization aids and excluded ligands.

    From AF3 Section 2.5, Tables 9-10.
    """

    def __init__(
        self,
        crystallization_aids: Optional[Set[str]] = None,
        excluded_ligands: Optional[Set[str]] = None,
        remove_water: bool = True,
        remove_ions: bool = False,
    ):
        self.crystallization_aids = crystallization_aids or CRYSTALLIZATION_AIDS
        self.excluded_ligands = excluded_ligands or LIGAND_EXCLUSION_LIST
        self.remove_water = remove_water
        self.remove_ions = remove_ions

    def __call__(self, structure: Structure) -> Optional[Structure]:
        filtered_chains = {}
        
        for chain_id, chain in structure.chains.items():
            # Check water removal
            if self.remove_water and chain.chain_type == ChainType.WATER:
                continue
            
            # Check ion removal
            if self.remove_ions and chain.chain_type == ChainType.ION:
                continue
            
            # Check ligand exclusions
            if chain.chain_type == ChainType.LIGAND:
                excluded = True
                for residue in chain.residues:
                    if residue.name not in self.crystallization_aids:
                        if residue.name not in self.excluded_ligands:
                            excluded = False
                            break
                if excluded:
                    continue
            
            filtered_chains[chain_id] = chain
        
        if not filtered_chains:
            return None
        
        return Structure(
            pdb_id=structure.pdb_id,
            chains=filtered_chains,
            release_date=structure.release_date,
            resolution=structure.resolution,
            method=structure.method,
            bonds=getattr(structure, 'bonds', []),
        )


class CurationFilterPipeline(StructureTransform):
    """Pipeline of structure filters.
    
    Applies multiple filters in sequence.
    """
    
    def __init__(
        self,
        transforms: Optional[List[StructureTransform]] = None,
    ):
        self.transforms = transforms or []
    
    def add(self, transform: StructureTransform) -> "CurationFilterPipeline":
        """Add a transform to the pipeline."""
        self.transforms.append(transform)
        return self
    
    def __call__(self, structure: Structure) -> Optional[Structure]:
        result = structure
        for transform in self.transforms:
            if result is None:
                return None
            result = transform(result)
        return result


# =============================================================================
# Per-Sequence Transforms (Clustering)
# =============================================================================

class ClusterTransform(BatchTransform):
    """Assign cluster IDs to structures based on sequences.
    
    From AF3 Section 2.5:
    - Proteins: 40% sequence identity
    - Nucleic acids: 100% sequence identity
    """
    
    def __init__(
        self,
        cluster_file: Optional[str] = None,
        protein_threshold: float = 0.4,
        nucleic_threshold: float = 1.0,
    ):
        self.cluster_assignments: Dict[str, str] = {}
        self.protein_threshold = protein_threshold
        self.nucleic_threshold = nucleic_threshold
        
        if cluster_file:
            self.load_clusters(cluster_file)
    
    def load_clusters(self, cluster_file: str) -> None:
        """Load cluster assignments from file."""
        with open(cluster_file, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    self.cluster_assignments[parts[0]] = parts[1]
    
    def __call__(self, structures: List[Structure]) -> List[Structure]:
        """Assign cluster IDs to all structures."""
        for structure in structures:
            for chain_id, chain in structure.chains.items():
                seq_id = f"{structure.pdb_id}_{chain_id}"
                cluster_id = self.cluster_assignments.get(seq_id)
                if cluster_id:
                    # Store cluster ID in chain metadata
                    if not hasattr(chain, 'metadata'):
                        chain.metadata = {}
                    chain.metadata['cluster_id'] = cluster_id
        return structures
    
    def get_cluster(self, seq_id: str) -> Optional[str]:
        """Get cluster ID for a sequence."""
        return self.cluster_assignments.get(seq_id)


# =============================================================================
# Batch Transforms (Sampling and Weighting)
# =============================================================================

class InverseClusterWeightTransform(BatchTransform):
    """Apply inverse cluster size weighting.
    
    From AF3 Section 2.5: Structures in larger clusters get lower weight.
    """
    
    def __init__(
        self,
        cluster_sizes: Optional[Dict[str, int]] = None,
    ):
        self.cluster_sizes = cluster_sizes or {}
    
    def load_cluster_sizes(self, cluster_file: str) -> None:
        """Load cluster sizes from file."""
        with open(cluster_file, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    self.cluster_sizes[parts[0]] = int(parts[1])
    
    def compute_weight(self, structure: Structure) -> float:
        """Compute inverse cluster weight for structure."""
        min_cluster_size = float('inf')
        
        for chain_id, chain in structure.chains.items():
            seq_id = f"{structure.pdb_id}_{chain_id}"
            cluster_id = getattr(chain, 'metadata', {}).get('cluster_id')
            if cluster_id:
                size = self.cluster_sizes.get(cluster_id, 1)
                min_cluster_size = min(min_cluster_size, size)
        
        if min_cluster_size == float('inf'):
            return 1.0
        
        return 1.0 / min_cluster_size
    
    def __call__(self, structures: List[Structure]) -> List[Structure]:
        """Assign weights to all structures."""
        for structure in structures:
            weight = self.compute_weight(structure)
            if not hasattr(structure, 'metadata'):
                structure.metadata = {}
            structure.metadata['cluster_weight'] = weight
        return structures


class DatasetWeightTransform(BatchTransform):
    """Apply dataset-level weights.
    
    From AF3 Table 4:
    - Weighted PDB: 0.5
    - Disordered PDB distillation: 0.02
    - Protein monomer distillation: 0.495
    - Short protein distillation: 0.005
    - RNA distillation: 0.05
    """
    
    def __init__(
        self,
        dataset_weights: Optional[Dict[str, float]] = None,
    ):
        self.dataset_weights = dataset_weights or {
            "pdb": 0.5,
            "disordered_distillation": 0.02,
            "monomer_distillation": 0.495,
            "short_protein_distillation": 0.005,
            "rna_distillation": 0.05,
        }
    
    def __call__(self, structures: List[Structure]) -> List[Structure]:
        """Apply dataset weights to structures."""
        for structure in structures:
            if not hasattr(structure, 'metadata'):
                structure.metadata = {}
            
            dataset = structure.metadata.get('dataset', 'pdb')
            weight = self.dataset_weights.get(dataset, 1.0)
            structure.metadata['dataset_weight'] = weight
        
        return structures


class SamplerTransform(BatchTransform):
    """Sample structures according to weights.
    
    Combines cluster weights, dataset weights, and other factors.
    """
    
    def __init__(
        self,
        seed: int = 42,
        replacement: bool = True,
    ):
        self.rng = np.random.default_rng(seed)
        self.replacement = replacement
    
    def __call__(
        self,
        structures: List[Structure],
        n_samples: Optional[int] = None,
    ) -> List[Structure]:
        """Sample structures according to weights."""
        if not structures:
            return []
        
        n_samples = n_samples or len(structures)
        
        # Compute combined weights
        weights = []
        for structure in structures:
            meta = getattr(structure, 'metadata', {})
            cluster_weight = meta.get('cluster_weight', 1.0)
            dataset_weight = meta.get('dataset_weight', 1.0)
            weights.append(cluster_weight * dataset_weight)
        
        weights = np.array(weights)
        weights /= weights.sum()
        
        # Sample
        indices = self.rng.choice(
            len(structures),
            size=n_samples,
            replace=self.replacement,
            p=weights,
        )
        
        return [structures[i] for i in indices]


class BatchFilterTransform(BatchTransform):
    """Filter a batch of structures.
    
    Applies structure-level filtering to a batch.
    """
    
    def __init__(
        self,
        structure_filter: Optional[StructureTransform] = None,
    ):
        self.structure_filter = structure_filter
    
    def __call__(self, structures: List[Structure]) -> List[Structure]:
        """Filter batch of structures."""
        if self.structure_filter is None:
            return structures
        
        filtered = []
        for structure in structures:
            result = self.structure_filter(structure)
            if result is not None:
                filtered.append(result)
        
        return filtered


# =============================================================================
# Complete Curation Pipeline
# =============================================================================

class CurationPipeline:
    """Complete curation pipeline.
    
    Combines filtering, clustering, and sampling.
    
    Example:
        ```python
        config = CurationConfig(
            training_cutoff=date(2021, 9, 30),
            max_resolution=9.0,
        )
        
        pipeline = CurationPipeline(config)
        
        # Load cluster information
        pipeline.load_clusters("clusters.tsv")
        
        # Process structures
        curated = pipeline.process(structures)
        
        # Sample for training
        batch = pipeline.sample(curated, batch_size=32)
        ```
    """
    
    def __init__(
        self,
        config: Optional[CurationConfig] = None,
    ):
        self.config = config or CurationConfig()
        
        # Build filter pipeline
        self.filter_pipeline = CurationFilterPipeline([
            DateFilterTransform(self.config.training_cutoff),
            ResolutionFilterTransform(self.config.max_resolution),
            ChainCompositionFilterTransform(),
            ExclusionListFilterTransform(),
        ])
        
        # Clustering
        self.cluster_transform = ClusterTransform(
            protein_threshold=self.config.protein_identity_threshold,
            nucleic_threshold=self.config.nucleic_identity_threshold,
        )
        
        # Weighting
        self.cluster_weight_transform = InverseClusterWeightTransform()
        self.dataset_weight_transform = DatasetWeightTransform({
            "pdb": self.config.pdb_weight,
            "disordered_distillation": self.config.disordered_distillation_weight,
            "monomer_distillation": self.config.monomer_distillation_weight,
            "short_protein_distillation": self.config.short_protein_weight,
            "rna_distillation": self.config.rna_distillation_weight,
        })
        
        # Sampling
        self.sampler = SamplerTransform()
    
    def load_clusters(self, cluster_file: str) -> None:
        """Load cluster assignments."""
        self.cluster_transform.load_clusters(cluster_file)
    
    def load_cluster_sizes(self, size_file: str) -> None:
        """Load cluster sizes for weighting."""
        self.cluster_weight_transform.load_cluster_sizes(size_file)
    
    def process(
        self,
        structures: List[Structure],
        apply_weights: bool = True,
    ) -> List[Structure]:
        """Process structures through full curation pipeline.
        
        Args:
            structures: List of structures to process
            apply_weights: Whether to compute sampling weights
            
        Returns:
            Curated list of structures
        """
        # Filter
        filtered = []
        for structure in structures:
            result = self.filter_pipeline(structure)
            if result is not None:
                filtered.append(result)
        
        # Assign clusters
        filtered = self.cluster_transform(filtered)
        
        # Apply weights
        if apply_weights:
            filtered = self.cluster_weight_transform(filtered)
            filtered = self.dataset_weight_transform(filtered)
        
        return filtered
    
    def sample(
        self,
        structures: List[Structure],
        batch_size: int,
    ) -> List[Structure]:
        """Sample a batch from curated structures."""
        return self.sampler(structures, n_samples=batch_size)


# =============================================================================
# Utility Functions
# =============================================================================

def filter_structure(
    structure: Structure,
    config: Optional[CurationConfig] = None,
) -> Optional[Structure]:
    """Apply standard filtering to a single structure.
    
    Args:
        structure: Structure to filter
        config: Curation configuration
        
    Returns:
        Filtered structure or None if filtered out
    """
    config = config or CurationConfig()
    pipeline = CurationFilterPipeline([
        DateFilterTransform(config.training_cutoff),
        ResolutionFilterTransform(config.max_resolution),
        ChainCompositionFilterTransform(),
        ExclusionListFilterTransform(),
    ])
    return pipeline(structure)


def filter_structures_batch(
    structures: List[Structure],
    config: Optional[CurationConfig] = None,
) -> List[Structure]:
    """Apply standard filtering to a batch of structures.
    
    Args:
        structures: List of structures to filter
        config: Curation configuration
        
    Returns:
        List of filtered structures
    """
    config = config or CurationConfig()
    batch_filter = BatchFilterTransform(
        CurationFilterPipeline([
            DateFilterTransform(config.training_cutoff),
            ResolutionFilterTransform(config.max_resolution),
            ChainCompositionFilterTransform(),
            ExclusionListFilterTransform(),
        ])
    )
    return batch_filter(structures)
