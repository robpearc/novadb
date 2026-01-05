"""Interface-based clustering for dataset curation.

Implements interface clustering from AlphaFold3 Section 2.5:
- Cluster structures by interface composition
- Identify protein-protein, protein-ligand, protein-nucleic interfaces
- Group similar interfaces for balanced sampling
"""

from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple
import numpy as np

from novadb.data.parsers.structure import Structure, Chain, ChainType

logger = logging.getLogger(__name__)


@dataclass
class InterfaceType:
    """Description of an interface type."""
    chain_type_1: ChainType
    chain_type_2: ChainType
    
    def __hash__(self) -> int:
        # Order-independent hash
        types = tuple(sorted([self.chain_type_1.name, self.chain_type_2.name]))
        return hash(types)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, InterfaceType):
            return False
        return {self.chain_type_1, self.chain_type_2} == {other.chain_type_1, other.chain_type_2}
    
    @property
    def name(self) -> str:
        """Human-readable name."""
        types = sorted([self.chain_type_1.name, self.chain_type_2.name])
        return f"{types[0].lower()}-{types[1].lower()}"


@dataclass
class Interface:
    """Represents an interface between two chains."""
    chain_id_1: str
    chain_id_2: str
    chain_type_1: ChainType
    chain_type_2: ChainType
    contact_residues_1: List[int]  # Residue indices
    contact_residues_2: List[int]
    contact_pairs: List[Tuple[int, int]]  # (res1_idx, res2_idx)
    min_distance: float = 0.0
    buried_surface_area: float = 0.0
    
    @property
    def interface_type(self) -> InterfaceType:
        return InterfaceType(self.chain_type_1, self.chain_type_2)
    
    @property
    def num_contacts(self) -> int:
        return len(self.contact_pairs)
    
    @property
    def size(self) -> int:
        """Total number of residues involved in interface."""
        return len(set(self.contact_residues_1)) + len(set(self.contact_residues_2))


@dataclass
class InterfaceCluster:
    """A cluster of similar interfaces."""
    cluster_id: str
    interface_type: InterfaceType
    members: List[Tuple[str, str, str]]  # (structure_id, chain1, chain2)
    representative: Optional[Tuple[str, str, str]] = None
    
    @property
    def size(self) -> int:
        return len(self.members)


@dataclass
class InterfaceClusteringConfig:
    """Configuration for interface clustering."""
    
    # Contact detection
    contact_distance: float = 8.0  # Å
    min_contacts: int = 5
    
    # Clustering
    cluster_by_type: bool = True
    cluster_by_size: bool = True
    size_bins: List[int] = field(default_factory=lambda: [10, 25, 50, 100, 200])
    
    # Ligand clustering
    cluster_ligands_by_ccd: bool = True
    
    # Interface composition fingerprint
    use_residue_composition: bool = True
    composition_similarity_threshold: float = 0.5


class InterfaceDetector:
    """Detect interfaces between chains in a structure."""
    
    def __init__(
        self,
        contact_distance: float = 8.0,
        min_contacts: int = 5,
    ):
        self.contact_distance = contact_distance
        self.min_contacts = min_contacts
    
    def detect_interfaces(self, structure: Structure) -> List[Interface]:
        """Detect all interfaces in a structure.
        
        Args:
            structure: Structure to analyze
            
        Returns:
            List of detected interfaces
        """
        interfaces = []
        chain_ids = list(structure.chains.keys())
        
        for i, chain_id_1 in enumerate(chain_ids):
            for chain_id_2 in chain_ids[i + 1:]:
                chain1 = structure.chains[chain_id_1]
                chain2 = structure.chains[chain_id_2]
                
                interface = self._detect_interface(
                    chain_id_1, chain1,
                    chain_id_2, chain2,
                )
                
                if interface is not None:
                    interfaces.append(interface)
        
        return interfaces
    
    def _detect_interface(
        self,
        chain_id_1: str,
        chain1: Chain,
        chain_id_2: str,
        chain2: Chain,
    ) -> Optional[Interface]:
        """Detect interface between two chains."""
        contact_residues_1 = []
        contact_residues_2 = []
        contact_pairs = []
        min_dist = float('inf')
        
        for i, res1 in enumerate(chain1.residues):
            coords1 = res1.heavy_atom_coords
            if len(coords1) == 0:
                continue
            
            for j, res2 in enumerate(chain2.residues):
                coords2 = res2.heavy_atom_coords
                if len(coords2) == 0:
                    continue
                
                # Compute minimum distance
                dist_matrix = np.linalg.norm(
                    coords1[:, None, :] - coords2[None, :, :],
                    axis=-1
                )
                dist = dist_matrix.min()
                
                if dist < self.contact_distance:
                    contact_residues_1.append(i)
                    contact_residues_2.append(j)
                    contact_pairs.append((i, j))
                    min_dist = min(min_dist, dist)
        
        if len(contact_pairs) < self.min_contacts:
            return None
        
        return Interface(
            chain_id_1=chain_id_1,
            chain_id_2=chain_id_2,
            chain_type_1=chain1.chain_type or ChainType.UNKNOWN,
            chain_type_2=chain2.chain_type or ChainType.UNKNOWN,
            contact_residues_1=contact_residues_1,
            contact_residues_2=contact_residues_2,
            contact_pairs=contact_pairs,
            min_distance=min_dist if min_dist != float('inf') else 0.0,
        )
    
    def get_interface_types(self, structure: Structure) -> Set[InterfaceType]:
        """Get all interface types present in a structure."""
        interfaces = self.detect_interfaces(structure)
        return {iface.interface_type for iface in interfaces}
    
    def has_interface_type(
        self,
        structure: Structure,
        interface_type: InterfaceType,
    ) -> bool:
        """Check if structure has a specific interface type."""
        interfaces = self.detect_interfaces(structure)
        for iface in interfaces:
            if iface.interface_type == interface_type:
                return True
        return False


class InterfaceClusterer:
    """Cluster structures by interface composition.
    
    From AF3 Section 2.5: Cluster by interface type to ensure
    balanced sampling of different interaction types.
    
    Example:
        ```python
        config = InterfaceClusteringConfig(
            contact_distance=8.0,
            min_contacts=5,
        )
        
        clusterer = InterfaceClusterer(config)
        
        # Add structures
        for structure in structures:
            clusterer.add_structure(structure)
        
        # Get clusters
        clusters = clusterer.get_clusters()
        
        # Get cluster for a structure
        cluster_id = clusterer.get_cluster_id(structure)
        ```
    """
    
    def __init__(self, config: Optional[InterfaceClusteringConfig] = None):
        self.config = config or InterfaceClusteringConfig()
        self.detector = InterfaceDetector(
            contact_distance=self.config.contact_distance,
            min_contacts=self.config.min_contacts,
        )
        
        # Storage
        self.structures: Dict[str, Structure] = {}
        self.structure_interfaces: Dict[str, List[Interface]] = {}
        self.clusters: Dict[str, InterfaceCluster] = {}
        self.structure_to_cluster: Dict[str, str] = {}
    
    def add_structure(self, structure: Structure) -> None:
        """Add a structure for clustering.
        
        Args:
            structure: Structure to add
        """
        self.structures[structure.pdb_id] = structure
        interfaces = self.detector.detect_interfaces(structure)
        self.structure_interfaces[structure.pdb_id] = interfaces
    
    def cluster_all(self) -> Dict[str, InterfaceCluster]:
        """Cluster all added structures.
        
        Returns:
            Dictionary of cluster_id -> InterfaceCluster
        """
        # Group by interface type composition
        type_groups: Dict[FrozenSet[str], List[str]] = defaultdict(list)
        
        for struct_id, interfaces in self.structure_interfaces.items():
            # Create interface type signature
            if interfaces:
                type_set = frozenset(iface.interface_type.name for iface in interfaces)
            else:
                type_set = frozenset(["no_interface"])
            
            type_groups[type_set].append(struct_id)
        
        # Create clusters
        self.clusters = {}
        cluster_idx = 0
        
        for type_set, struct_ids in type_groups.items():
            if self.config.cluster_by_size:
                # Sub-cluster by interface size
                size_groups = self._group_by_size(struct_ids)
                for size_bin, bin_struct_ids in size_groups.items():
                    cluster_id = f"interface_cluster_{cluster_idx}"
                    
                    # Determine representative interface type
                    if type_set and "no_interface" not in type_set:
                        type_name = sorted(type_set)[0]
                        type_parts = type_name.split("-")
                        itype = InterfaceType(
                            ChainType[type_parts[0].upper()],
                            ChainType[type_parts[1].upper()],
                        )
                    else:
                        itype = InterfaceType(ChainType.UNKNOWN, ChainType.UNKNOWN)
                    
                    cluster = InterfaceCluster(
                        cluster_id=cluster_id,
                        interface_type=itype,
                        members=[(sid, "", "") for sid in bin_struct_ids],
                        representative=(bin_struct_ids[0], "", "") if bin_struct_ids else None,
                    )
                    
                    self.clusters[cluster_id] = cluster
                    for sid in bin_struct_ids:
                        self.structure_to_cluster[sid] = cluster_id
                    
                    cluster_idx += 1
            else:
                # Single cluster per type set
                cluster_id = f"interface_cluster_{cluster_idx}"
                
                if type_set and "no_interface" not in type_set:
                    type_name = sorted(type_set)[0]
                    type_parts = type_name.split("-")
                    itype = InterfaceType(
                        ChainType[type_parts[0].upper()],
                        ChainType[type_parts[1].upper()],
                    )
                else:
                    itype = InterfaceType(ChainType.UNKNOWN, ChainType.UNKNOWN)
                
                cluster = InterfaceCluster(
                    cluster_id=cluster_id,
                    interface_type=itype,
                    members=[(sid, "", "") for sid in struct_ids],
                    representative=(struct_ids[0], "", "") if struct_ids else None,
                )
                
                self.clusters[cluster_id] = cluster
                for sid in struct_ids:
                    self.structure_to_cluster[sid] = cluster_id
                
                cluster_idx += 1
        
        logger.info(f"Created {len(self.clusters)} interface clusters from {len(self.structures)} structures")
        return self.clusters
    
    def _group_by_size(
        self,
        struct_ids: List[str],
    ) -> Dict[int, List[str]]:
        """Group structures by total interface size."""
        size_groups: Dict[int, List[str]] = defaultdict(list)
        
        for struct_id in struct_ids:
            interfaces = self.structure_interfaces.get(struct_id, [])
            total_size = sum(iface.num_contacts for iface in interfaces)
            
            # Find appropriate bin
            bin_idx = 0
            for i, threshold in enumerate(self.config.size_bins):
                if total_size >= threshold:
                    bin_idx = i + 1
            
            size_groups[bin_idx].append(struct_id)
        
        return size_groups
    
    def get_cluster_id(self, structure: Structure) -> Optional[str]:
        """Get cluster ID for a structure.
        
        Args:
            structure: Structure to look up
            
        Returns:
            Cluster ID or None if not found
        """
        return self.structure_to_cluster.get(structure.pdb_id)
    
    def get_cluster(self, cluster_id: str) -> Optional[InterfaceCluster]:
        """Get cluster by ID."""
        return self.clusters.get(cluster_id)
    
    def get_cluster_sizes(self) -> Dict[str, int]:
        """Get sizes of all clusters."""
        return {cid: cluster.size for cid, cluster in self.clusters.items()}
    
    def get_clusters_by_type(
        self,
        interface_type: InterfaceType,
    ) -> List[InterfaceCluster]:
        """Get all clusters with a specific interface type."""
        return [
            cluster for cluster in self.clusters.values()
            if cluster.interface_type == interface_type
        ]
    
    def compute_interface_fingerprint(
        self,
        structure: Structure,
    ) -> str:
        """Compute a fingerprint for the interface composition.
        
        Args:
            structure: Structure to fingerprint
            
        Returns:
            Hex string fingerprint
        """
        interfaces = self.structure_interfaces.get(structure.pdb_id, [])
        
        if not interfaces:
            interfaces = self.detector.detect_interfaces(structure)
        
        # Build fingerprint components
        components = []
        
        for iface in sorted(interfaces, key=lambda x: x.interface_type.name):
            comp = f"{iface.interface_type.name}:{iface.num_contacts}"
            
            if self.config.use_residue_composition:
                # Add residue composition
                chain1 = structure.chains.get(iface.chain_id_1)
                chain2 = structure.chains.get(iface.chain_id_2)
                
                if chain1 and chain2:
                    res_types_1 = [
                        chain1.residues[i].name
                        for i in set(iface.contact_residues_1)
                        if i < len(chain1.residues)
                    ]
                    res_types_2 = [
                        chain2.residues[i].name
                        for i in set(iface.contact_residues_2)
                        if i < len(chain2.residues)
                    ]
                    
                    comp += f":{sorted(res_types_1)}:{sorted(res_types_2)}"
            
            components.append(comp)
        
        # Hash the fingerprint
        fingerprint_str = "|".join(components)
        return hashlib.md5(fingerprint_str.encode()).hexdigest()


class LigandClusterer:
    """Cluster ligands by CCD code.
    
    From AF3 Section 2.5: Ligands clustered by CCD code
    rather than sequence identity.
    """
    
    def __init__(self):
        self.ccd_to_structures: Dict[str, Set[str]] = defaultdict(set)
        self.structure_to_ccds: Dict[str, Set[str]] = defaultdict(set)
    
    def add_structure(self, structure: Structure) -> None:
        """Add a structure for ligand clustering."""
        struct_id = structure.pdb_id
        
        for chain in structure.chains.values():
            if chain.chain_type == ChainType.LIGAND:
                for residue in chain.residues:
                    ccd_code = residue.name
                    self.ccd_to_structures[ccd_code].add(struct_id)
                    self.structure_to_ccds[struct_id].add(ccd_code)
    
    def get_ligand_cluster(self, ccd_code: str) -> Set[str]:
        """Get all structures containing a ligand."""
        return self.ccd_to_structures.get(ccd_code, set())
    
    def get_structure_ligands(self, struct_id: str) -> Set[str]:
        """Get all ligand CCD codes in a structure."""
        return self.structure_to_ccds.get(struct_id, set())
    
    def get_ligand_counts(self) -> Dict[str, int]:
        """Get count of structures per ligand type."""
        return {ccd: len(structs) for ccd, structs in self.ccd_to_structures.items()}


class ChainTypeClassifier:
    """Classify chain types for proper clustering.
    
    From AF3 Section 2.5: Different clustering thresholds
    for different chain types.
    """
    
    # Extended set of standard amino acids including modified
    PROTEIN_RESIDUES = {
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
        "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
        "MSE", "SEC", "PYL", "UNK",  # Modified/unknown
    }
    
    RNA_RESIDUES = {"A", "G", "C", "U", "N", "I"}  # Including inosine
    DNA_RESIDUES = {"DA", "DG", "DC", "DT", "DN", "DI"}
    
    # Common ions
    IONS = {
        "AG", "AL", "AU", "BA", "BR", "CA", "CD", "CE", "CL", "CO", "CR", "CS",
        "CU", "DY", "EU", "F", "FE", "GA", "GD", "HG", "IN", "IR", "K", "LA",
        "LI", "LU", "MG", "MN", "MO", "NA", "NI", "OS", "PB", "PD", "PR", "PT",
        "RB", "RU", "SB", "SM", "SR", "TB", "TH", "TL", "V", "W", "Y", "YB", "ZN", "ZR"
    }
    
    @classmethod
    def classify_chain(cls, chain: Chain) -> ChainType:
        """Classify chain type based on residue composition.
        
        Args:
            chain: Chain to classify
            
        Returns:
            Classified chain type
        """
        if not chain.residues:
            return ChainType.UNKNOWN
        
        # Count residue types
        protein_count = 0
        rna_count = 0
        dna_count = 0
        ion_count = 0
        water_count = 0
        other_count = 0
        
        for res in chain.residues:
            name = res.name.upper()
            
            if name in cls.PROTEIN_RESIDUES:
                protein_count += 1
            elif name in cls.RNA_RESIDUES:
                rna_count += 1
            elif name in cls.DNA_RESIDUES:
                dna_count += 1
            elif name in cls.IONS:
                ion_count += 1
            elif name == "HOH" or name == "WAT":
                water_count += 1
            else:
                other_count += 1
        
        total = len(chain.residues)
        
        # Determine type by majority
        if protein_count > total * 0.8:
            return ChainType.PROTEIN
        elif rna_count > total * 0.8:
            return ChainType.RNA
        elif dna_count > total * 0.8:
            return ChainType.DNA
        elif ion_count == total:
            return ChainType.ION
        elif water_count == total:
            return ChainType.WATER
        elif other_count > 0 and (protein_count + rna_count + dna_count) == 0:
            return ChainType.LIGAND
        else:
            # Mixed or unknown
            return ChainType.UNKNOWN
    
    @classmethod
    def is_polymer(cls, chain_type: ChainType) -> bool:
        """Check if chain type is a polymer."""
        return chain_type in {ChainType.PROTEIN, ChainType.RNA, ChainType.DNA}
    
    @classmethod
    def is_small_molecule(cls, chain_type: ChainType) -> bool:
        """Check if chain type is a small molecule."""
        return chain_type in {ChainType.LIGAND, ChainType.ION}
    
    @classmethod
    def get_clustering_threshold(cls, chain_type: ChainType) -> float:
        """Get sequence identity threshold for clustering.
        
        From AF3 Section 2.5:
        - Proteins: 40%
        - Nucleic acids: 100%
        - Peptides (< 10 residues): 100%
        """
        if chain_type == ChainType.PROTEIN:
            return 0.4
        elif chain_type in {ChainType.RNA, ChainType.DNA}:
            return 1.0
        else:
            return 1.0


def compute_interface_similarity(
    interface1: Interface,
    interface2: Interface,
) -> float:
    """Compute similarity between two interfaces.
    
    Args:
        interface1: First interface
        interface2: Second interface
        
    Returns:
        Similarity score (0-1)
    """
    # Must be same type
    if interface1.interface_type != interface2.interface_type:
        return 0.0
    
    # Compare sizes
    size_ratio = min(interface1.num_contacts, interface2.num_contacts) / \
                 max(interface1.num_contacts, interface2.num_contacts)
    
    # Could add more sophisticated comparison (e.g., residue composition)
    return size_ratio


def get_structure_interface_signature(
    structure: Structure,
    detector: Optional[InterfaceDetector] = None,
) -> str:
    """Get a string signature of structure's interfaces.
    
    Args:
        structure: Structure to analyze
        detector: Interface detector (created if None)
        
    Returns:
        String signature
    """
    detector = detector or InterfaceDetector()
    interfaces = detector.detect_interfaces(structure)
    
    if not interfaces:
        return "no_interface"
    
    types = sorted(set(iface.interface_type.name for iface in interfaces))
    total_contacts = sum(iface.num_contacts for iface in interfaces)
    
    return f"{'+'.join(types)}:{total_contacts}"
